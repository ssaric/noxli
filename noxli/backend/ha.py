"""Home Assistant Supervisor API client for stream source discovery."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SUPERVISOR_URL = "http://supervisor/core/api"
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

# Entity domains that can provide a live audio stream
STREAM_DOMAINS = ("camera.", "media_player.")

# Attributes that may contain a stream URL
STREAM_ATTRS = ("stream_source", "rtsp_url", "media_content_id", "url")


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
            # Check if entity has any stream-related attribute
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


async def _try_go2rtc(entity_id: str) -> str | None:
    """Try to resolve a stream URL via go2rtc's REST API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{SUPERVISOR_URL}/go2rtc/api/streams",
                headers=_headers(),
            )
            resp.raise_for_status()
            streams = resp.json()

        stream = streams.get(entity_id)
        if not stream:
            return None

        producers = stream.get("producers", [])
        for p in producers:
            url = p.get("url", "")
            if url.startswith(("rtsp://", "rtsps://")):
                return url

        # Fall back to any producer URL
        if producers:
            return producers[0].get("url")

        return None
    except Exception as exc:
        logger.debug("go2rtc API not available: %s", exc)
        return None


async def get_stream_source(entity_id: str) -> str | None:
    """Resolve a stream URL from entity attributes, then go2rtc."""

    # 1 — check entity state attributes
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
                    logger.info("Found stream URL in attribute '%s' for %s", attr, entity_id)
                    return val

            logger.info(
                "No stream URL in attributes for %s (attrs: %s)",
                entity_id,
                list(attrs.keys()),
            )
    except Exception as exc:
        logger.warning("Failed to fetch state for %s: %s", entity_id, exc)

    # 2 — try go2rtc
    url = await _try_go2rtc(entity_id)
    if url:
        logger.info("Resolved stream URL via go2rtc for %s", entity_id)
        return url

    return None
