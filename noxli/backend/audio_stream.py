"""Detection loop — reads audio from ffmpeg pipe and runs YAMNet inference."""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import db, detector

SAMPLE_RATE = detector.SAMPLE_RATE
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SECONDS = 1
CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS  # 32000

DEFAULT_COOLDOWN = 5.0  # seconds between distinct events


@dataclass
class _CryEvent:
    """Tracks an ongoing cry event from onset to offset."""
    onset: float = 0.0
    onset_chunk: int = 0
    max_confidence: float = 0.0
    patches: int = 0


@dataclass
class DetectionStats:
    running: bool = False
    rtsp_url: str = ""
    events_detected: int = 0
    last_event_time: float | None = None
    chunks_processed: int = 0
    started_at: float | None = None
    error: str | None = None


class DetectionLoop:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None
        self._stats = DetectionStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> DetectionStats:
        with self._lock:
            return DetectionStats(**self._stats.__dict__)

    def start(self, rtsp_url: str, sensitivity: float = 0.5) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Detection loop already running")

        self._stop_event.clear()
        with self._lock:
            self._stats = DetectionStats(
                running=True,
                rtsp_url=rtsp_url,
                started_at=time.time(),
            )
        self._thread = threading.Thread(
            target=self._run,
            args=(rtsp_url, sensitivity),
            daemon=True,
        )
        self._thread.start()
        print(f"[noxli] Detection loop started: {rtsp_url}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process:
                    self._process.kill()
        if self._thread:
            self._thread.join(timeout=10)
        with self._lock:
            self._stats.running = False
        print("[noxli] Detection loop stopped")

    def _run(self, rtsp_url: str, sensitivity: float) -> None:
        try:
            self._run_loop(rtsp_url, sensitivity)
        except Exception as e:
            with self._lock:
                self._stats.error = str(e)
            print(f"[noxli] Detection loop error: {e}")
        finally:
            with self._lock:
                self._stats.running = False

    def _build_ffmpeg_cmd(self, rtsp_url: str) -> list[str]:
        out_args = ["-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]

        # Microphone input via PulseAudio or ALSA
        if rtsp_url.startswith("pulse://"):
            device = rtsp_url.removeprefix("pulse://") or "default"
            return ["ffmpeg", "-f", "pulse", "-i", device] + out_args
        if rtsp_url.startswith("alsa://"):
            device = rtsp_url.removeprefix("alsa://") or "default"
            return ["ffmpeg", "-f", "alsa", "-i", device] + out_args

        # RTSP stream
        if rtsp_url.startswith("rtsp"):
            return ["ffmpeg", "-rtsp_transport", "tcp", "-i", rtsp_url, "-vn"] + out_args

        # HTTP/HLS stream (e.g. HA supervisor HLS URL)
        if rtsp_url.startswith("http"):
            return ["ffmpeg", "-i", rtsp_url, "-vn"] + out_args

        # File path or file:// URL
        source = rtsp_url.removeprefix("file://")
        return ["ffmpeg", "-i", source, "-vn"] + out_args

    def _run_loop(self, rtsp_url: str, sensitivity: float) -> None:
        cmd = self._build_ffmpeg_cmd(rtsp_url)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        current_cry: _CryEvent | None = None
        last_event_end_chunk = -DEFAULT_COOLDOWN  # in chunk-seconds
        chunk_index = 0

        while not self._stop_event.is_set():
            data = self._process.stdout.read(CHUNK_BYTES)
            if not data:
                break
            if len(data) < CHUNK_BYTES:
                # Pad short final chunk
                data = data + b"\x00" * (CHUNK_BYTES - len(data))

            # Convert s16le bytes to float32 waveform
            waveform = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            detected, confidence = detector.is_cry(waveform, threshold=sensitivity)
            chunk_index += 1

            with self._lock:
                self._stats.chunks_processed = chunk_index

            now = time.time()
            audio_pos = chunk_index * CHUNK_SECONDS

            if detected:
                if current_cry is None:
                    # Cooldown check (in audio time)
                    if audio_pos - last_event_end_chunk < DEFAULT_COOLDOWN:
                        continue
                    current_cry = _CryEvent(
                        onset=now, onset_chunk=chunk_index,
                        max_confidence=confidence, patches=1,
                    )
                else:
                    current_cry.max_confidence = max(current_cry.max_confidence, confidence)
                    current_cry.patches += 1
            else:
                if current_cry is not None:
                    # Cry ended — compute duration from chunk count
                    duration = current_cry.patches * CHUNK_SECONDS
                    self._record_event(
                        timestamp=current_cry.onset,
                        duration=duration,
                        confidence=current_cry.max_confidence,
                    )
                    last_event_end_chunk = audio_pos
                    current_cry = None

        # If stream ends mid-cry, record it
        if current_cry is not None:
            duration = current_cry.patches * CHUNK_SECONDS
            self._record_event(
                timestamp=current_cry.onset,
                duration=duration,
                confidence=current_cry.max_confidence,
            )

        if self._process:
            self._process.wait()

    def _record_event(self, timestamp: float, duration: float, confidence: float) -> None:
        """Store event in DB and publish via MQTT if configured."""
        with db.get_db() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, duration, confidence, source) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, duration, confidence, "yamnet"),
            )
            conn.commit()

        with self._lock:
            self._stats.events_detected += 1
            self._stats.last_event_time = timestamp

        print(
            f"[noxli] Cry event: duration={duration:.1f}s, "
            f"confidence={confidence:.3f}"
        )

        self._publish_mqtt(timestamp, duration, confidence)

    def _publish_mqtt(self, timestamp: float, duration: float, confidence: float) -> None:
        host = os.environ.get("MQTT_HOST")
        if not host:
            return
        try:
            import paho.mqtt.publish as publish
            topic = "noxli/detection"
            payload = json.dumps({
                "event": "cry_detected",
                "timestamp": timestamp,
                "duration": duration,
                "confidence": confidence,
            })
            publish.single(
                topic,
                payload=payload,
                hostname=host,
                port=int(os.environ.get("MQTT_PORT", "1883")),
                auth={
                    "username": os.environ.get("MQTT_USER", ""),
                    "password": os.environ.get("MQTT_PASS", ""),
                } if os.environ.get("MQTT_USER") else None,
            )
        except Exception as e:
            print(f"[noxli] MQTT publish failed: {e}")


# Singleton instance
loop = DetectionLoop()
