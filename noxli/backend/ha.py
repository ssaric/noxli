"""Home Assistant Supervisor API client for stream source discovery."""

import json
import os

import httpx

SUPERVISOR_URL = "http://supervisor/core/api"
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

# Entity domains that can provide a live audio stream
STREAM_DOMAINS = ("camera.", "media_player.")

# Attributes that may contain a stream URL
STREAM_ATTRS = ("stream_source", "rtsp_url", "media_content_id", "url")


def _log(msg: str) -> None:
    print(f"[noxli] {msg}", flush=True)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
        "Content-Type": "application/json",
    }


async def get_stream_entities() -> list[dict]:
    """Fetch entities that may provide a live audio stream."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SUPERVISOR_URL}/states", headers=_headers()
            )
            resp.raise_for_status()
            states = resp.json()

        results = []
        for s in states:
            eid = s["entity_id"]
            if not any(eid.startswith(d) for d in STREAM_DOMAINS):
                continue
            attrs = s.get("attributes", {})
            has_stream = any(attrs.get(a) for a in STREAM_ATTRS)
            results.append({
                "entity_id": eid,
                "name": attrs.get("friendly_name", eid),
                "state": s["state"],
                "domain": eid.split(".")[0],
                "has_stream_url": has_stream,
            })
        return results
    except Exception:
        return []


async def _try_expose_integration(entity_id: str) -> str | None:
    """Try hass-expose-camera-stream-source custom integration."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{SUPERVISOR_URL}/camera_stream_source/{entity_id}",
                headers=_headers(),
            )
            if resp.status_code == 200:
                url = resp.text.strip()
                if url:
                    return url
        return None
    except Exception:
        return None


GO2RTC_RTSP = "rtsp://homeassistant:8554"


async def _try_go2rtc(entity_id: str) -> str | None:
    """Try to resolve a stream URL via go2rtc.

    Prefer the camera's direct RTSP source if available.
    Otherwise use go2rtc's RTSP proxy which includes audio
    (unlike HA's HLS stream which is often video-only).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{SUPERVISOR_URL}/go2rtc/api/streams",
                headers=_headers(),
            )
            resp.raise_for_status()
            streams = resp.json()

        # If go2rtc knows about this camera, check for a direct RTSP source
        stream = streams.get(entity_id)
        if stream:
            producers = stream.get("producers", [])
            for p in producers:
                url = p.get("url", "")
                if url.startswith(("rtsp://", "rtsps://")):
                    _log(f"go2rtc has direct RTSP source for {entity_id}")
                    return url

        # go2rtc is available — use its RTSP proxy (includes audio)
        # go2rtc creates streams on-demand for HA camera entities
        proxy_url = f"{GO2RTC_RTSP}/{entity_id}"
        _log(f"Using go2rtc RTSP proxy: {proxy_url}")
        return proxy_url

    except Exception as exc:
        _log(f"go2rtc not available: {exc}")
        return None


async def _try_ws_stream(entity_id: str) -> str | None:
    """Request an HLS stream URL via the HA WebSocket API."""
    try:
        import websockets  # shipped with uvicorn[standard]
    except ImportError:
        _log("websockets library not available, skipping WS method")
        return None

    try:
        async with websockets.connect("ws://supervisor/core/websocket") as ws:
            # auth handshake
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_required":
                return None

            await ws.send(json.dumps({
                "type": "auth",
                "access_token": SUPERVISOR_TOKEN,
            }))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                return None

            # request camera stream
            await ws.send(json.dumps({
                "id": 1,
                "type": "camera/stream",
                "entity_id": entity_id,
                "format": "hls",
            }))
            msg = json.loads(await ws.recv())

            if msg.get("success"):
                hls_path = msg.get("result", {}).get("url", "")
                if hls_path:
                    return f"http://supervisor/core{hls_path}"

        return None
    except Exception as exc:
        _log(f"WebSocket camera/stream failed for {entity_id}: {exc}")
        return None


async def get_stream_source(entity_id: str) -> str | None:
    """Try every available method to resolve a stream URL."""

    # 1 — entity state attributes
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SUPERVISOR_URL}/states/{entity_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            state = resp.json()
            attrs = state.get("attributes", {})

            for attr in STREAM_ATTRS:
                val = attrs.get(attr)
                if val and isinstance(val, str):
                    _log(f"Resolved {entity_id} via attribute '{attr}'")
                    return val

            _log(f"No stream URL in attributes for {entity_id} "
                 f"(keys: {list(attrs.keys())})")
    except Exception as exc:
        _log(f"Failed to fetch state for {entity_id}: {exc}")

    # 2 — expose-camera-stream-source custom integration
    url = await _try_expose_integration(entity_id)
    if url:
        _log(f"Resolved {entity_id} via expose-camera-stream-source")
        return url

    # 3 — go2rtc
    url = await _try_go2rtc(entity_id)
    if url:
        _log(f"Resolved {entity_id} via go2rtc")
        return url

    # 4 — WebSocket camera/stream (returns HLS URL)
    url = await _try_ws_stream(entity_id)
    if url:
        _log(f"Resolved {entity_id} via WebSocket HLS stream")
        return url

    _log(f"All resolution methods failed for {entity_id}")
    return None
