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

# Download YAMNet model files if not present
MODELS_DIR="/data/models"
if [ ! -f "${MODELS_DIR}/yamnet.tflite" ] && [ ! -f "${MODELS_DIR}/yamnet.onnx" ]; then
    bashio::log.info "Downloading YAMNet model..."
    curl -sL "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" \
        -o "${MODELS_DIR}/yamnet.tflite" \
        && bashio::log.info "YAMNet model downloaded" \
        || bashio::log.warning "Failed to download YAMNet model — try placing yamnet.tflite in /data/models/ manually"
fi
if [ ! -f "${MODELS_DIR}/yamnet_class_map.csv" ]; then
    bashio::log.info "Downloading YAMNet class map..."
    curl -sL "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv" \
        -o "${MODELS_DIR}/yamnet_class_map.csv" \
        || bashio::log.warning "Failed to download class map"
fi

bashio::log.info "Starting Noxli backend..."
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8099
