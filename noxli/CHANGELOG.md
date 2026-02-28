# Changelog

## [0.5.0](https://github.com/ssaric/noxli/compare/noxli-v0.4.1...noxli-v0.5.0) (2026-02-28)


### Features

* add 24-hour cry event timeline with SQLite event store ([f8d2268](https://github.com/ssaric/noxli/commit/f8d2268797183ed2b98943cbef615cfbb73191ce))
* add go2rtc fallback for stream URL resolution ([1f7575c](https://github.com/ssaric/noxli/commit/1f7575c68c5ad09713d0630d42c5d382875acc8b))
* add YAMNet baby cry detection with live audio pipeline ([a7af521](https://github.com/ssaric/noxli/commit/a7af521e184869d455628bf6654debe0a84329d0))


### Bug Fixes

* add CHANGELOG.md for HA addon changelog display ([2e2750d](https://github.com/ssaric/noxli/commit/2e2750df5cec48b242c7cb763069664344a862b4))
* add multiple fallback methods for stream URL resolution ([379313e](https://github.com/ssaric/noxli/commit/379313e48dd2213534f7fe5ac2814268dbbeb3e1))
* fix model download URL and add HLS stream support ([7e64699](https://github.com/ssaric/noxli/commit/7e64699bfa7772f725246e1e6dc063d4b9386dda))
* remove illegal parent path from changelog-path ([1657309](https://github.com/ssaric/noxli/commit/165730957f247bd741de133c88eccfd9a42e1a5c))

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
