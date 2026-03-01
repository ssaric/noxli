"""Noxli companion integration — exposes camera RTSP URLs to the Noxli addon."""

from homeassistant.components.camera import Camera
from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_component import EntityComponent

from .const import DOMAIN


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Noxli from a config entry."""
    hass.http.register_view(NoxliStreamView())
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Noxli config entry."""
    return True


class NoxliStreamView(HomeAssistantView):
    """Expose camera stream source via /api/noxli/stream/{entity_id}."""

    url = "/api/noxli/stream/{entity_id}"
    name = "api:noxli:stream"
    requires_auth = True

    async def get(self, request, entity_id: str):
        """Return the RTSP URL for the given camera entity."""
        from aiohttp import web

        hass = request.app["hass"]
        component: EntityComponent[Camera] = hass.data.get("camera")
        if component is None:
            return web.Response(status=404, text="Camera component not loaded")

        camera = component.get_entity(entity_id)
        if camera is None:
            return web.Response(status=404, text=f"Camera {entity_id} not found")

        try:
            source = await camera.stream_source()
        except Exception:
            source = None

        if not source:
            return web.Response(status=404, text=f"No stream source for {entity_id}")

        return web.Response(text=source, content_type="text/plain")
