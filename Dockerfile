ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x for future frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    tflite-runtime \
    numpy \
    soundfile \
    paho-mqtt \
    fastapi \
    "uvicorn[standard]"

# Copy rootfs overlay
COPY rootfs /

# Copy entrypoint
COPY run.sh /
RUN chmod a+x /run.sh

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8099/api/health || exit 1

CMD ["/run.sh"]
