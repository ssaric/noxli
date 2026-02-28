"""YAMNet-based baby cry detector — supports TFLite and ONNX backends."""

import csv
from pathlib import Path

import numpy as np

# --- Backend selection ---
# Try TFLite first (production Docker), fall back to ONNX (dev / Python 3.14+)
_BACKEND = None
_tflite = None
_ort = None

try:
    import tflite_runtime.interpreter as _tflite_mod
    _tflite = _tflite_mod
    _BACKEND = "tflite"
except ImportError:
    try:
        from ai_edge_litert import interpreter as _tflite_mod
        _tflite = _tflite_mod
        _BACKEND = "tflite"
    except ImportError:
        import onnxruntime as _ort_mod
        _ort = _ort_mod
        _BACKEND = "onnx"

# --- Constants ---

MODEL_DIR = Path("/data/models")

SAMPLE_RATE = 16000
WAVEFORM_SAMPLES = 15600  # ~0.975s per inference patch

# Mel spectrogram params (for ONNX backend)
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
STFT_WINDOW_SAMPLES = int(SAMPLE_RATE * STFT_WINDOW_SECONDS)  # 400
STFT_HOP_SAMPLES = int(SAMPLE_RATE * STFT_HOP_SECONDS)        # 160
FFT_SIZE = 512
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.001

# AudioSet class indices for cry detection
CRY_CLASSES = {
    19: "Crying, sobbing",
    20: "Baby cry, infant cry",
}

# --- Mel spectrogram helpers (ONNX backend) ---


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_bands: int, fft_size: int, sample_rate: int,
                    min_hz: float, max_hz: float) -> np.ndarray:
    """Build a mel filterbank matrix [num_fft_bins, num_bands]."""
    num_fft_bins = fft_size // 2 + 1
    mel_min = _hz_to_mel(min_hz)
    mel_max = _hz_to_mel(max_hz)
    mel_points = np.linspace(mel_min, mel_max, num_bands + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.round(hz_points * fft_size / sample_rate).astype(int)

    filterbank = np.zeros((num_fft_bins, num_bands), dtype=np.float32)
    for i in range(num_bands):
        lo, center, hi = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(lo, center):
            if center > lo:
                filterbank[j, i] = (j - lo) / (center - lo)
        for j in range(center, hi):
            if hi > center:
                filterbank[j, i] = (hi - j) / (hi - center)
    return filterbank


_MEL_FILTERBANK = None


def _get_mel_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _mel_filterbank(MEL_BANDS, FFT_SIZE, SAMPLE_RATE,
                                          MEL_MIN_HZ, MEL_MAX_HZ)
    return _MEL_FILTERBANK


def waveform_to_mel(waveform: np.ndarray) -> np.ndarray:
    """Convert raw 16kHz mono waveform to log-mel spectrogram [96, 64]."""
    window = np.hanning(STFT_WINDOW_SAMPLES).astype(np.float32)
    num_frames = 1 + (len(waveform) - STFT_WINDOW_SAMPLES) // STFT_HOP_SAMPLES
    frames = np.stack([
        waveform[i * STFT_HOP_SAMPLES : i * STFT_HOP_SAMPLES + STFT_WINDOW_SAMPLES]
        for i in range(num_frames)
    ])
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=FFT_SIZE)
    # Use amplitude (not power) to match VGGish/YAMNet preprocessing
    amplitude = np.abs(spectrum)

    fb = _get_mel_filterbank()
    mel = amplitude @ fb
    log_mel = np.log(mel + LOG_OFFSET).astype(np.float32)
    return log_mel


# --- Model loading ---

_interpreter = None
_session = None
_class_names: dict[int, str] = {}


def _load_class_map():
    global _class_names
    csv_path = MODEL_DIR / "yamnet_class_map.csv"
    if not csv_path.exists():
        print(f"[noxli] Warning: class map not found at {csv_path}")
        return
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            _class_names[int(row["index"])] = row["display_name"]


def _load_model():
    global _interpreter, _session

    _load_class_map()

    if _BACKEND == "tflite":
        model_path = MODEL_DIR / "yamnet.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        _interpreter = _tflite.Interpreter(model_path=str(model_path))
        _interpreter.allocate_tensors()
        print(f"[noxli] Loaded YAMNet TFLite model from {model_path}")
    else:
        model_path = MODEL_DIR / "yamnet.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _session = _ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        print(f"[noxli] Loaded YAMNet ONNX model from {model_path}")

    print(f"[noxli] Backend: {_BACKEND}, classes loaded: {len(_class_names)}")


def _ensure_loaded():
    if _interpreter is None and _session is None:
        _load_model()


# --- Inference ---


def detect(waveform: np.ndarray) -> list[dict]:
    """Run YAMNet inference on float32 mono 16kHz audio.

    The waveform is split into ~0.975s patches.  Returns a list of dicts,
    one per patch, with keys: patch_index, scores (dict of class_index →
    confidence for cry classes), top_class, top_class_name, top_score.
    """
    _ensure_loaded()

    waveform = waveform.astype(np.float32)
    # Split into non-overlapping patches of WAVEFORM_SAMPLES
    num_patches = len(waveform) // WAVEFORM_SAMPLES
    if num_patches == 0:
        # Pad short audio
        padded = np.zeros(WAVEFORM_SAMPLES, dtype=np.float32)
        padded[:len(waveform)] = waveform
        waveform = padded
        num_patches = 1

    results = []
    for i in range(num_patches):
        patch = waveform[i * WAVEFORM_SAMPLES : (i + 1) * WAVEFORM_SAMPLES]
        scores = _infer_patch(patch)

        cry_scores = {idx: float(scores[idx]) for idx in CRY_CLASSES}
        top_idx = max(range(len(scores)), key=lambda j: scores[j])

        results.append({
            "patch_index": i,
            "scores": cry_scores,
            "top_class": top_idx,
            "top_class_name": _class_names.get(top_idx, f"class_{top_idx}"),
            "top_score": float(scores[top_idx]),
        })

    return results


def _infer_patch(patch: np.ndarray) -> np.ndarray:
    """Run inference on a single WAVEFORM_SAMPLES-length patch. Returns [521] scores."""
    if _BACKEND == "tflite":
        return _infer_tflite(patch)
    else:
        return _infer_onnx(patch)


def _infer_tflite(patch: np.ndarray) -> np.ndarray:
    input_details = _interpreter.get_input_details()
    output_details = _interpreter.get_output_details()
    _interpreter.resize_tensor_input(input_details[0]["index"], [WAVEFORM_SAMPLES])
    _interpreter.allocate_tensors()
    _interpreter.set_tensor(input_details[0]["index"], patch)
    _interpreter.invoke()
    return _interpreter.get_tensor(output_details[0]["index"])[0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _infer_onnx(patch: np.ndarray) -> np.ndarray:
    mel = waveform_to_mel(patch)
    # Ensure exactly 96 frames
    if mel.shape[0] > 96:
        mel = mel[:96]
    elif mel.shape[0] < 96:
        pad = np.zeros((96 - mel.shape[0], MEL_BANDS), dtype=np.float32)
        mel = np.concatenate([mel, pad])
    input_tensor = mel.reshape(1, 1, 96, 64)
    outputs = _session.run(None, {"audio": input_tensor})
    # ONNX model outputs raw logits — apply sigmoid for probabilities
    return _sigmoid(outputs[0][0])


def is_cry(waveform: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
    """Convenience: returns (detected, max_confidence) across all patches."""
    results = detect(waveform)
    max_conf = 0.0
    for r in results:
        for score in r["scores"].values():
            max_conf = max(max_conf, score)
    return max_conf >= threshold, max_conf
