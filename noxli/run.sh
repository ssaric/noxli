#!/usr/bin/with-contenv bashio

# Discover MQTT broker from Supervisor
if bashio::services "mqtt"; then
    export MQTT_HOST=$(bashio::services mqtt "host")
    export MQTT_PORT=$(bashio::services mqtt "port")
    export MQTT_USER=$(bashio::services mqtt "username")
    export MQTT_PASS=$(bashio::services mqtt "password")
    bashio::log.info "MQTT broker discovered at ${MQTT_HOST}:${MQTT_PORT}"
else
    bashio::log.warning "No MQTT service found — detection events will only be logged locally"
fi

# Ingress entry point for HA panel
export INGRESS_ENTRY=$(bashio::addon.ingress_entry)

# Ensure data directories exist
mkdir -p /data/db /data/models

bashio::log.info "Starting Noxli backend..."
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8099
