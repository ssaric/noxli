ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    tflite-runtime \
    numpy \
    soundfile \
    paho-mqtt \
    fastapi \
    "uvicorn[standard]" \
    httpx

# Copy application code
COPY backend/ /backend/
COPY frontend/ /frontend/

# Copy rootfs overlay
COPY rootfs /

# Copy entrypoint
COPY run.sh /
RUN chmod a+x /run.sh

CMD ["/run.sh"]
