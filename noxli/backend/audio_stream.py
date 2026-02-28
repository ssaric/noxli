"""Detection loop — reads audio from ffmpeg pipe and runs YAMNet inference."""

import collections
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
DEBUG_BUFFER_SIZE = 30  # keep last N chunk results for debug endpoint


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
    ffmpeg_error: str | None = None


class DetectionLoop:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None
        self._stats = DetectionStats()
        self._lock = threading.Lock()
        self._debug_buffer: collections.deque[dict] = collections.deque(maxlen=DEBUG_BUFFER_SIZE)

    @property
    def stats(self) -> DetectionStats:
        with self._lock:
            return DetectionStats(**self._stats.__dict__)

    @property
    def debug_chunks(self) -> list[dict]:
        with self._lock:
            return list(self._debug_buffer)

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
            self._debug_buffer.clear()
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
            cmd = ["ffmpeg"]
            # Supervisor HLS URLs require Bearer token auth
            if "supervisor/" in rtsp_url:
                token = os.environ.get("SUPERVISOR_TOKEN", "")
                if token:
                    cmd += ["-headers", f"Authorization: Bearer {token}\r\n"]
            cmd += ["-i", rtsp_url, "-vn"] + out_args
            return cmd

        # File path or file:// URL
        source = rtsp_url.removeprefix("file://")
        return ["ffmpeg", "-i", source, "-vn"] + out_args

    def _run_loop(self, rtsp_url: str, sensitivity: float) -> None:
        cmd = self._build_ffmpeg_cmd(rtsp_url)
        print(f"[noxli] ffmpeg cmd: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read stderr in a background thread so it doesn't block
        stderr_lines: list[str] = []

        def _drain_stderr():
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(text)
                # Keep only last 50 lines
                if len(stderr_lines) > 50:
                    stderr_lines.pop(0)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        current_cry: _CryEvent | None = None
        last_event_end_chunk = -DEFAULT_COOLDOWN  # in chunk-seconds
        chunk_index = 0

        while not self._stop_event.is_set():
            data = self._process.stdout.read(CHUNK_BYTES)
            if not data:
                # ffmpeg exited — capture why
                stderr_thread.join(timeout=2)
                stderr_tail = "\n".join(stderr_lines[-10:]) if stderr_lines else "no output"
                with self._lock:
                    self._stats.ffmpeg_error = stderr_tail
                if chunk_index == 0:
                    print(f"[noxli] ffmpeg produced no audio. stderr:\n{stderr_tail}")
                else:
                    print(f"[noxli] ffmpeg stream ended after {chunk_index} chunks")
                break
            if len(data) < CHUNK_BYTES:
                # Pad short final chunk
                data = data + b"\x00" * (CHUNK_BYTES - len(data))

            # Convert s16le bytes to float32 waveform
            waveform = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Compute audio level
            rms = float(np.sqrt(np.mean(waveform ** 2)))
            peak = float(np.max(np.abs(waveform)))

            results = detector.detect(waveform)
            chunk_index += 1

            # Get cry confidence and top class from results
            cry_conf = 0.0
            top_class = ""
            top_score = 0.0
            for r in results:
                for score in r["scores"].values():
                    cry_conf = max(cry_conf, score)
                if r["top_score"] > top_score:
                    top_score = r["top_score"]
                    top_class = r["top_class_name"]

            detected = cry_conf >= sensitivity

            # Store debug info
            debug_entry = {
                "chunk": chunk_index,
                "time": time.time(),
                "rms": round(rms, 6),
                "peak": round(peak, 4),
                "top_class": top_class,
                "top_score": round(top_score, 4),
                "cry_confidence": round(cry_conf, 4),
                "detected": detected,
                "silence": rms < 0.001,
            }
            with self._lock:
                self._stats.chunks_processed = chunk_index
                self._debug_buffer.append(debug_entry)

            now = time.time()
            audio_pos = chunk_index * CHUNK_SECONDS

            if detected:
                if current_cry is None:
                    # Cooldown check (in audio time)
                    if audio_pos - last_event_end_chunk < DEFAULT_COOLDOWN:
                        continue
                    current_cry = _CryEvent(
                        onset=now, onset_chunk=chunk_index,
                        max_confidence=cry_conf, patches=1,
                    )
                else:
                    current_cry.max_confidence = max(current_cry.max_confidence, cry_conf)
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
