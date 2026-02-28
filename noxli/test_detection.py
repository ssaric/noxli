#!/usr/bin/env python3
"""Test YAMNet baby-cry detection on an audio file.

Usage:
    python test_detection.py baby_cry.wav
    python test_detection.py baby_cry.mp3        # auto-converts via ffmpeg
    python test_detection.py                     # uses built-in synthetic test
    python test_detection.py --post baby_cry.wav  # also posts events to dev server
"""

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

# Ensure models are available
MODELS_DIR = Path(__file__).parent / ".devdata" / "models"
YAMNET_ONNX = MODELS_DIR / "yamnet.onnx"
YAMNET_TFLITE = MODELS_DIR / "yamnet.tflite"
YAMNET_CSV = MODELS_DIR / "yamnet_class_map.csv"

CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


def ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not YAMNET_CSV.exists():
        print(f"Downloading class map to {YAMNET_CSV} ...")
        urllib.request.urlretrieve(CLASS_MAP_URL, YAMNET_CSV)
    if not YAMNET_ONNX.exists() and not YAMNET_TFLITE.exists():
        print("No model found in .devdata/models/.")
        print("Please download yamnet.onnx (or yamnet.tflite) to .devdata/models/")
        sys.exit(1)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file (WAV, MP3, etc.), return (samples, sample_rate).

    For non-WAV formats, uses ffmpeg to convert to 16kHz mono WAV first.
    """
    import soundfile as sf

    p = Path(path)
    if p.suffix.lower() in (".wav", ".flac", ".ogg"):
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    # Convert to WAV via ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_path],
        capture_output=True, check=True,
    )
    data, sr = sf.read(tmp_path, dtype="float32")
    Path(tmp_path).unlink()
    return data, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def post_event(confidence: float, duration: float, server: str = "http://localhost:8099"):
    """Post a detection event to the dev server."""
    import json
    import urllib.request
    data = json.dumps({
        "confidence": confidence,
        "duration": duration,
        "source": "test_detection",
    }).encode()
    req = urllib.request.Request(
        f"{server}/api/events",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"  Posted event: id={result['id']}, confidence={confidence:.3f}")
    except Exception as e:
        print(f"  Failed to post event: {e}")


def run_detection(wav_path: str | None, do_post: bool = False):
    ensure_models()

    # Patch model dir for dev
    import backend.detector as det
    det.MODEL_DIR = MODELS_DIR

    if wav_path:
        print(f"Loading: {wav_path}")
        audio, sr = load_audio(wav_path)
        audio = resample(audio, sr, det.SAMPLE_RATE)
        duration = len(audio) / det.SAMPLE_RATE
        print(f"  Duration: {duration:.2f}s, samples: {len(audio)}")
    else:
        print("No WAV file provided — running synthetic tests\n")
        # Test 1: silence
        print("=== Test: Silence (1s) ===")
        silence = np.zeros(det.SAMPLE_RATE, dtype=np.float32)
        detected, conf = det.is_cry(silence)
        print(f"  Result: {'CRY DETECTED' if detected else 'No cry detected'}")
        print(f"  Confidence: {conf:.6f}")

        # Test 2: 440Hz tone
        print("\n=== Test: 440Hz Tone (1s) ===")
        t = np.linspace(0, 1, det.SAMPLE_RATE, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        detected, conf = det.is_cry(tone)
        print(f"  Result: {'CRY DETECTED' if detected else 'No cry detected'}")
        print(f"  Confidence: {conf:.6f}")

        # Test 3: simulated cry-like noise (high-pitched noise bursts)
        print("\n=== Test: High-frequency noise burst (simulated, not real cry) ===")
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.3, det.SAMPLE_RATE).astype(np.float32)
        # Add some high-frequency content
        t = np.linspace(0, 1, det.SAMPLE_RATE, dtype=np.float32)
        cry_sim = noise + 0.4 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
        cry_sim = np.clip(cry_sim, -1, 1).astype(np.float32)
        detected, conf = det.is_cry(cry_sim)
        print(f"  Result: {'CRY DETECTED' if detected else 'No cry detected'}")
        print(f"  Confidence: {conf:.6f}")
        return

    # Run detection on provided file
    print(f"\nRunning YAMNet detection (backend: {det._BACKEND})...\n")
    results = det.detect(audio)

    any_cry = False
    max_cry_conf = 0.0
    for r in results:
        cry_conf = max(r["scores"].values())
        max_cry_conf = max(max_cry_conf, cry_conf)
        is_cry = cry_conf >= 0.3

        status = "** CRY **" if is_cry else "         "
        print(
            f"  Patch {r['patch_index']:3d}  "
            f"top={r['top_class_name']:<30s} ({r['top_score']:.3f})  "
            f"cry={cry_conf:.3f} {status}"
        )
        if is_cry:
            any_cry = True

    print(f"\n{'=' * 60}")
    if any_cry:
        print(f"Baby cry DETECTED  (max confidence: {max_cry_conf:.3f})")
    else:
        print(f"No cry detected    (max confidence: {max_cry_conf:.6f})")

    if do_post and any_cry:
        post_event(confidence=max_cry_conf, duration=duration)


if __name__ == "__main__":
    args = sys.argv[1:]
    do_post = "--post" in args
    args = [a for a in args if a != "--post"]
    wav_path = args[0] if args else None
    run_detection(wav_path, do_post)
