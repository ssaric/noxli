#!/usr/bin/env python3
"""End-to-end stream detection test.

Creates a test WAV (silence + baby cry + silence), pipes it through the
detection loop as if it were a stream, and verifies events appear in the API.

Usage:
    # With dev server already running on port 8099:
    python test_stream.py

    # Provide a custom baby cry source:
    python test_stream.py --cry .devdata/baby-cry.wav
"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

DEVDATA = Path(__file__).parent / ".devdata"
MODELS_DIR = DEVDATA / "models"
SERVER = "http://localhost:8099"


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file, converting non-WAV via ffmpeg."""
    import soundfile as sf

    p = Path(path)
    if p.suffix.lower() in (".wav", ".flac", ".ogg"):
        data, sr = sf.read(path, dtype="float32")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_path],
            capture_output=True, check=True,
        )
        data, sr = sf.read(tmp_path, dtype="float32")
        Path(tmp_path).unlink()

    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def build_test_wav(cry_path: str | None) -> str:
    """Build a test WAV: 3s silence + cry + 3s silence."""
    import soundfile as sf

    sr = 16000
    silence_sec = 3
    silence = np.zeros(sr * silence_sec, dtype=np.float32)

    if cry_path:
        cry_audio, cry_sr = load_audio(cry_path)
        cry_audio = resample(cry_audio, cry_sr, sr)
    else:
        # Look for a default baby cry file
        for candidate in [DEVDATA / "baby-cry.wav", DEVDATA / "baby-cry.mp3"]:
            if candidate.exists():
                cry_audio, cry_sr = load_audio(str(candidate))
                cry_audio = resample(cry_audio, cry_sr, sr)
                break
        else:
            print("No baby cry audio found. Provide one with --cry <path>")
            print("or place baby-cry.wav/.mp3 in .devdata/")
            sys.exit(1)

    combined = np.concatenate([silence, cry_audio, silence])
    duration = len(combined) / sr

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, combined, sr, subtype="PCM_16")
    print(f"Test WAV: {duration:.1f}s (3s silence + {len(cry_audio)/sr:.1f}s cry + 3s silence)")
    return tmp.name


def api_get(path: str) -> dict:
    resp = urllib.request.urlopen(f"{SERVER}{path}")
    return json.loads(resp.read())


def api_post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def check_server():
    try:
        api_get("/api/health")
        return True
    except Exception:
        return False


def main():
    args = sys.argv[1:]
    cry_path = None
    if "--cry" in args:
        idx = args.index("--cry")
        cry_path = args[idx + 1]

    # Check dev server is running
    if not check_server():
        print(f"Dev server not reachable at {SERVER}")
        print("Start it first: python dev.py")
        sys.exit(1)
    print(f"Dev server OK at {SERVER}")

    # Get events before test
    before = api_get("/api/events?hours=1")
    events_before = len(before["events"])

    # Build test WAV
    test_wav = build_test_wav(cry_path)

    try:
        # Patch model dir and start detection on the file
        import backend.detector as det
        det.MODEL_DIR = MODELS_DIR

        import backend.db as db_mod
        db_mod.DB_PATH = DEVDATA / "events.db"

        from backend.audio_stream import DetectionLoop

        loop = DetectionLoop()
        print("\nStarting detection on test WAV...")
        loop.start(f"file://{test_wav}", sensitivity=0.5)

        # Wait for processing to finish
        max_wait = 30
        waited = 0
        while waited < max_wait:
            time.sleep(1)
            waited += 1
            s = loop.stats
            if not s.running:
                break
            print(f"  [{waited}s] chunks={s.chunks_processed}, events={s.events_detected}")

        loop.stop()
        stats = loop.stats
        print(f"\nDetection finished: {stats.chunks_processed} chunks, {stats.events_detected} events")

        if stats.error:
            print(f"Error: {stats.error}")

        # Check events via API
        after = api_get("/api/events?hours=1")
        events_after = len(after["events"])
        new_events = events_after - events_before

        print(f"\n{'=' * 60}")
        print(f"Events before: {events_before}")
        print(f"Events after:  {events_after}")
        print(f"New events:    {new_events}")

        if new_events > 0:
            print("\nNew events:")
            for ev in after["events"][-new_events:]:
                print(
                    f"  id={ev['id']}  confidence={ev['confidence']:.3f}  "
                    f"duration={ev['duration']:.1f}s  source={ev['source']}"
                )
            print(f"\nSUCCESS: Detected {new_events} cry event(s) from test stream")
        else:
            print("\nWARNING: No new events detected")

    finally:
        Path(test_wav).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
