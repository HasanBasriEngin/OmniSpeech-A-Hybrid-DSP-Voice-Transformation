from __future__ import annotations

import numpy as np
from scipy import signal


def _safe_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _band_rms(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    x = _safe_mono_float(audio)
    if x.size < 64 or sample_rate <= 0:
        return 0.0

    nyquist = sample_rate / 2.0
    high = min(high_hz, nyquist * 0.98)
    low = max(low_hz, 20.0)
    if low >= high:
        return 0.0

    try:
        sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        band = signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return 0.0

    return float(np.sqrt(np.mean(band**2))) if band.size else 0.0


def measure_audio_health(audio: np.ndarray, sample_rate: int, *, prefix: str = "") -> dict[str, float]:
    x = _safe_mono_float(audio)
    if x.size == 0:
        return {
            f"{prefix}peak": 0.0,
            f"{prefix}rms": 0.0,
            f"{prefix}clipping_ratio": 0.0,
            f"{prefix}finite_ratio": 1.0,
            f"{prefix}dc_offset": 0.0,
            f"{prefix}sibilance_ratio": 0.0,
        }

    finite = np.isfinite(np.asarray(audio, dtype=np.float32))
    finite_ratio = float(np.mean(finite)) if finite.size else 1.0
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    clipping_ratio = float(np.mean(np.abs(x) >= 0.98))
    dc_offset = float(np.mean(x))

    full_rms = rms + 1e-7
    sib_rms = _band_rms(x, sample_rate, 5_500.0, 9_500.0)
    sibilance_ratio = float(np.clip(sib_rms / full_rms, 0.0, 10.0))

    return {
        f"{prefix}peak": round(peak, 6),
        f"{prefix}rms": round(rms, 6),
        f"{prefix}clipping_ratio": round(clipping_ratio, 6),
        f"{prefix}finite_ratio": round(finite_ratio, 6),
        f"{prefix}dc_offset": round(dc_offset, 6),
        f"{prefix}sibilance_ratio": round(sibilance_ratio, 6),
    }
