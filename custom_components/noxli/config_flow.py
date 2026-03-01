"""Config flow for Noxli — zero-config, immediately creates entry."""

from homeassistant.config_entries import ConfigFlow

from .const import DOMAIN


class NoxliConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Noxli."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Create entry immediately — no configuration needed."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        return self.async_create_entry(title="Noxli", data={})
