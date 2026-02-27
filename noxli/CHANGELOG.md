# Changelog

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
