# Changelog

## 0.5.3 (2026-02-28)

- fix: cache resolved HLS URLs and stop audio endpoint spam


## 0.5.2 (2026-02-28)

- fix: re-resolve expired HLS stream URLs on detection start


## 0.5.1 (2026-02-28)

- fix: pass Supervisor auth token to ffmpeg for HLS streams


## 0.5.0 (2026-02-28)

- feat: add detection monitor UI with live feed and audio preview


## 0.4.2 (2026-02-28)

- fix: replace release-please with direct tag-and-release workflow


## 0.4.1

- Fix YAMNet model download URL (use TF Hub instead of Kaggle redirect)
- Add HLS stream support for HA camera stream resolution

## 0.4.0

- YAMNet-based baby cry detection with live audio analysis
- Detection loop reads audio via ffmpeg from RTSP streams, PulseAudio, or ALSA
- Supports both TFLite and ONNX model backends
- Auto-starts detection on boot when stream URL is configured
- API endpoints: POST /api/detection/start, POST /api/detection/stop, GET /api/detection/status
- Cry events stored to SQLite with duration and confidence
- MQTT publish on cry detection (noxli/detection topic)
- Sleep session wake_ups populated from detected cry events
- Dev test scripts for file-based and end-to-end pipeline testing

## 0.3.0

- Add 24-hour cry event timeline to ingress UI
- SQLite event store for cry detection events
- REST endpoints: GET /api/events, POST /api/events
- Timeline auto-refreshes every 60 seconds
- Event markers with hover tooltips showing time, confidence, and duration

## 0.2.1

- Add multiple fallback methods for camera stream URL resolution
  (entity attributes, expose-camera-stream-source, go2rtc, WebSocket HLS)
- Diagnostic logging visible in addon Log tab

## 0.2.0

- Move all addon options into the ingress UI settings panel
- Add detection sensitivity slider, MQTT topic input, and log level dropdown
- Backend supports partial config updates with defaults
- Remove options/schema from HA config page

## 0.1.0

- Initial release
- Ingress UI with stream source discovery from Home Assistant
- Manual RTSP URL entry
- MQTT broker auto-discovery
- Watchdog health endpoint
