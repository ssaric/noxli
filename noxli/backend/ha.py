"""Home Assistant Supervisor API client for stream source discovery."""

import os

import httpx

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


async def get_stream_source(entity_id: str) -> str | None:
    """Resolve a stream URL from an entity's attributes."""
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
                    return val
            return None
    except Exception:
        return None
