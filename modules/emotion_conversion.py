from __future__ import annotations

import numpy as np

from core.preprocessing import detect_f0
from utils.audio_utils import match_length, pitch_shift_simple, time_stretch_simple

EMOTION_PROFILES = {
    "sad": {"pitch_shift": -2.0, "rate": 0.85, "energy_scale": 0.70, "spectral_tilt": -2.0},
    "angry": {"pitch_shift": +1.5, "rate": 1.15, "energy_scale": 1.40, "spectral_tilt": +1.5},
    "excited": {"pitch_shift": +3.0, "rate": 1.25, "energy_scale": 1.30, "spectral_tilt": +2.0},
    "whispered": {
        "pitch_shift": 0.0,
        "rate": 0.90,
        "energy_scale": 0.40,
        "spectral_tilt": -3.0,
    },
    "calm": {"pitch_shift": -0.5, "rate": 0.95, "energy_scale": 0.85, "spectral_tilt": -0.5},
}


def shift_f0_contour(f0: np.ndarray, shift_st: float, modulate: bool = True) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float32)
    ratio = 2 ** (shift_st / 12.0)
    shifted = f0 * ratio
    if modulate and shifted.size > 0:
        idx = np.linspace(0, np.pi * 2, shifted.size, dtype=np.float32)
        shifted = shifted * (1.0 + 0.03 * np.sin(idx))
    return shifted.astype(np.float32)


def adjust_speech_rate(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    _ = sr
    return time_stretch_simple(audio, rate)


def modify_energy_envelope(audio: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(audio, dtype=np.float32) * float(scale)).astype(np.float32)


def apply_spectral_tilt(audio: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    _ = sr
    x = np.asarray(audio, dtype=np.float32)
    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    tilt = 10 ** ((tilt_db * (freqs - 0.5)) / 20.0)
    out = np.fft.irfft(spec * tilt, n=len(x))
    return out.astype(np.float32)


def convert_emotion(audio: np.ndarray, sr: int, target_emotion: str) -> np.ndarray:
    profile = EMOTION_PROFILES.get(target_emotion)
    if profile is None:
        raise ValueError(f"Unknown emotion: {target_emotion}")

    original_len = len(audio)
    out = pitch_shift_simple(audio, profile["pitch_shift"])
    out = adjust_speech_rate(out, sr, profile["rate"])
    out = modify_energy_envelope(out, profile["energy_scale"])
    out = apply_spectral_tilt(out, sr, profile["spectral_tilt"])
    out = match_length(out, original_len)

    # Keep contour call for pipeline compatibility.
    _ = shift_f0_contour(detect_f0(np.asarray(audio, dtype=np.float32), sr), profile["pitch_shift"])
    return np.clip(out, -1.0, 1.0).astype(np.float32)

