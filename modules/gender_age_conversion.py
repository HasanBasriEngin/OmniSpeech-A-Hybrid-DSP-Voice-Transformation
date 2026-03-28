from __future__ import annotations

import numpy as np
from scipy.signal import resample

from utils.audio_utils import match_length, pitch_shift_simple

CONVERSION_MAP = {
    "male_to_female": {"formant_ratio": 1.18, "pitch_shift": +3.5, "vtl_factor": 0.85},
    "female_to_male": {"formant_ratio": 0.85, "pitch_shift": -3.5, "vtl_factor": 1.18},
    "adult_to_child": {"formant_ratio": 1.35, "pitch_shift": +5.0, "vtl_factor": 0.74},
    "adult_to_elderly": {"formant_ratio": 0.92, "pitch_shift": -1.5, "vtl_factor": 1.05},
    "child_to_adult": {"formant_ratio": 0.74, "pitch_shift": -5.0, "vtl_factor": 1.35},
}


def warp_spectral_envelope(envelope: np.ndarray, warp: float) -> np.ndarray:
    envelope = np.asarray(envelope, dtype=np.float32)
    idx = np.arange(len(envelope), dtype=np.float32)
    src = np.clip(idx / max(warp, 1e-6), 0, len(envelope) - 1)
    return np.interp(idx, src, envelope).astype(np.float32)


def apply_vtln(audio: np.ndarray, sr: int, warp_factor: float) -> np.ndarray:
    _ = sr
    x = np.asarray(audio, dtype=np.float32)
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    phase = np.angle(spec)
    warped_mag = warp_spectral_envelope(mag, warp_factor)
    out = np.fft.irfft(warped_mag * np.exp(1j * phase), n=len(x))
    return out.astype(np.float32)


def shift_formants(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    _ = sr
    x = np.asarray(audio, dtype=np.float32)
    warped_len = max(1, int(len(x) / ratio))
    warped = resample(x, warped_len)
    return resample(warped, len(x)).astype(np.float32)


def convert_gender_age(audio: np.ndarray, sr: int, conversion_type: str) -> np.ndarray:
    params = CONVERSION_MAP.get(conversion_type)
    if params is None:
        raise ValueError(f"Unknown conversion type: {conversion_type}")
    original_len = len(audio)
    out = pitch_shift_simple(audio, params["pitch_shift"])
    out = shift_formants(out, sr, params["formant_ratio"])
    out = apply_vtln(out, sr, params["vtl_factor"])
    out = match_length(out, original_len)
    return np.clip(out, -1.0, 1.0).astype(np.float32)

