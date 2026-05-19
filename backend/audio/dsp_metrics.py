from __future__ import annotations

import numpy as np
from scipy import signal

from backend.audio.features import extract_pitch_contour


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


def _noise_floor_ratio(audio: np.ndarray, sample_rate: int, rms: float) -> float:
    x = _safe_mono_float(audio)
    if x.size < 16 or rms <= 1e-8:
        return 0.0
    win = max(8, int(sample_rate * 0.02))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    local = np.sqrt(np.convolve(x * x, kernel, mode="same").clip(min=0.0))
    return float(np.clip(np.percentile(local, 12) / (rms + 1e-8), 0.0, 5.0))


def _spectral_flatness(audio: np.ndarray) -> float:
    x = _safe_mono_float(audio)
    if x.size < 16:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size).astype(np.float32))) + 1e-8
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum)) + 1e-8
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))


def _pitch_stability(audio: np.ndarray, sample_rate: int) -> float:
    contour = extract_pitch_contour(audio, sample_rate)
    voiced = contour[contour > 0]
    if voiced.size < 2:
        return 1.0
    median = float(np.median(voiced))
    if median <= 1e-6:
        return 1.0
    variation = float(np.std(voiced) / median)
    return float(np.clip(1.0 - variation, 0.0, 1.0))


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
            f"{prefix}noise_floor_ratio": 0.0,
            f"{prefix}spectral_flatness": 0.0,
            f"{prefix}harshness_ratio": 0.0,
            f"{prefix}pitch_stability": 1.0,
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
    harsh_rms = _band_rms(x, sample_rate, 2_700.0, 5_700.0)
    harshness_ratio = float(np.clip(harsh_rms / full_rms, 0.0, 10.0))
    noise_floor_ratio = _noise_floor_ratio(x, sample_rate, rms)
    spectral_flatness = _spectral_flatness(x)
    pitch_stability = _pitch_stability(x, sample_rate)

    return {
        f"{prefix}peak": round(peak, 6),
        f"{prefix}rms": round(rms, 6),
        f"{prefix}clipping_ratio": round(clipping_ratio, 6),
        f"{prefix}finite_ratio": round(finite_ratio, 6),
        f"{prefix}dc_offset": round(dc_offset, 6),
        f"{prefix}sibilance_ratio": round(sibilance_ratio, 6),
        f"{prefix}noise_floor_ratio": round(noise_floor_ratio, 6),
        f"{prefix}spectral_flatness": round(spectral_flatness, 6),
        f"{prefix}harshness_ratio": round(harshness_ratio, 6),
        f"{prefix}pitch_stability": round(pitch_stability, 6),
    }
