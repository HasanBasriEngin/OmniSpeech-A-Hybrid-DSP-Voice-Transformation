from __future__ import annotations

import numpy as np
from scipy import signal


_BANDS: tuple[tuple[str, float, float], ...] = (
    ("rumble", 20.0, 120.0),
    ("body", 120.0, 400.0),
    ("vocal", 400.0, 1_200.0),
    ("presence", 1_200.0, 3_000.0),
    ("harsh", 3_000.0, 6_000.0),
    ("sibilance", 6_000.0, 10_000.0),
    ("air", 10_000.0, 20_000.0),
)


def _safe_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _empty_metrics(prefix: str) -> dict[str, float]:
    base = {
        "spectrogram_rumble_ratio": 0.0,
        "spectrogram_body_ratio": 0.0,
        "spectrogram_vocal_ratio": 0.0,
        "spectrogram_presence_ratio": 0.0,
        "spectrogram_harsh_ratio": 0.0,
        "spectrogram_sibilance_ratio": 0.0,
        "spectrogram_air_ratio": 0.0,
        "spectrogram_high_balance": 0.0,
        "spectrogram_sibilance_spike_score": 0.0,
        "spectrogram_harsh_spike_score": 0.0,
        "spectrogram_high_noise_score": 0.0,
        "spectrogram_metallic_score": 0.0,
        "spectrogram_muffled_score": 0.0,
        "spectrogram_rumble_score": 0.0,
        "spectrogram_balance_score": 1.0,
        "spectrogram_artifact_score": 0.0,
    }
    return {f"{prefix}{key}": value for key, value in base.items()}


def _band_mask(freqs: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    low = max(0.0, min(float(low_hz), nyquist * 0.98))
    high = max(low + 1.0, min(float(high_hz), nyquist * 0.98))
    return (freqs >= low) & (freqs < high)


def _band_power_ratio(power: np.ndarray, freqs: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mask = _band_mask(freqs, sample_rate, low_hz, high_hz)
    total = float(np.sum(power)) + 1e-10
    if not np.any(mask):
        return 0.0
    return float(np.clip(np.sum(power[mask, :]) / total, 0.0, 1.0))


def _frame_band_ratio(power: np.ndarray, freqs: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    mask = _band_mask(freqs, sample_rate, low_hz, high_hz)
    if not np.any(mask) or power.size == 0:
        return np.zeros(power.shape[1] if power.ndim == 2 else 0, dtype=np.float32)
    frame_total = np.sum(power, axis=0) + 1e-10
    return np.clip(np.sum(power[mask, :], axis=0) / frame_total, 0.0, 1.0).astype(np.float32)


def _stationary_tone_score(power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> float:
    mask = _band_mask(freqs, sample_rate, 900.0, 9_500.0)
    if not np.any(mask):
        return 0.0
    mean_by_freq = np.mean(power[mask, :], axis=1)
    total = float(np.sum(mean_by_freq)) + 1e-10
    if total <= 1e-9:
        return 0.0
    concentration = float(np.max(mean_by_freq) / total)
    return float(np.clip((concentration - 0.22) / 0.30, 0.0, 1.0))


def analyze_spectrogram_quality(audio: np.ndarray, sample_rate: int, *, prefix: str = "") -> dict[str, float]:
    x = _safe_mono_float(audio)
    if x.size < 64 or sample_rate <= 0:
        return _empty_metrics(prefix)

    nperseg = min(max(512, sample_rate // 32), 2048, x.size)
    if nperseg < 64:
        return _empty_metrics(prefix)
    noverlap = max(0, min(nperseg - 1, nperseg // 2))

    try:
        freqs, _, stft = signal.stft(
            x,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
    except Exception:
        return _empty_metrics(prefix)

    if stft.size == 0 or freqs.size == 0:
        return _empty_metrics(prefix)

    power = (np.abs(stft).astype(np.float32) ** 2) + 1e-12
    ratios = {
        name: _band_power_ratio(power, freqs, sample_rate, low, high)
        for name, low, high in _BANDS
    }

    sibilance_frames = _frame_band_ratio(power, freqs, sample_rate, 6_000.0, 10_000.0)
    harsh_frames = _frame_band_ratio(power, freqs, sample_rate, 3_000.0, 6_000.0)
    high_frames = _frame_band_ratio(power, freqs, sample_rate, 6_000.0, 20_000.0)
    frame_total = np.sum(power, axis=0)

    quiet_limit = float(np.percentile(frame_total, 25)) if frame_total.size else 0.0
    quiet = frame_total <= quiet_limit
    quiet_high_ratio = float(np.mean(high_frames[quiet])) if np.any(quiet) else 0.0

    high_balance = float(
        np.clip(
            ratios["presence"] + ratios["harsh"] + ratios["sibilance"] + ratios["air"],
            0.0,
            1.0,
        )
    )
    sibilance_spike = float(np.percentile(sibilance_frames, 95)) if sibilance_frames.size else 0.0
    harsh_spike = float(np.percentile(harsh_frames, 95)) if harsh_frames.size else 0.0

    sibilance_spike_score = float(np.clip((sibilance_spike - 0.34) / 0.38, 0.0, 1.0))
    harsh_spike_score = float(np.clip((harsh_spike - 0.42) / 0.36, 0.0, 1.0))
    high_noise_score = float(np.clip((quiet_high_ratio - 0.28) / 0.42, 0.0, 1.0))
    metallic_score = _stationary_tone_score(power, freqs, sample_rate)
    muffled_score = float(np.clip((0.16 - high_balance) / 0.16, 0.0, 1.0))
    rumble_score = float(np.clip((ratios["rumble"] - 0.16) / 0.24, 0.0, 1.0))

    balance_penalty = max(
        sibilance_spike_score,
        harsh_spike_score,
        high_noise_score,
        metallic_score,
        muffled_score * 0.75,
        rumble_score,
    )
    balance_score = float(np.clip(1.0 - balance_penalty, 0.0, 1.0))
    artifact_score = float(
        np.clip(
            0.22 * sibilance_spike_score
            + 0.20 * harsh_spike_score
            + 0.18 * high_noise_score
            + 0.18 * metallic_score
            + 0.12 * muffled_score
            + 0.10 * rumble_score,
            0.0,
            1.0,
        )
    )

    raw = {
        "spectrogram_rumble_ratio": ratios["rumble"],
        "spectrogram_body_ratio": ratios["body"],
        "spectrogram_vocal_ratio": ratios["vocal"],
        "spectrogram_presence_ratio": ratios["presence"],
        "spectrogram_harsh_ratio": ratios["harsh"],
        "spectrogram_sibilance_ratio": ratios["sibilance"],
        "spectrogram_air_ratio": ratios["air"],
        "spectrogram_high_balance": high_balance,
        "spectrogram_sibilance_spike_score": sibilance_spike_score,
        "spectrogram_harsh_spike_score": harsh_spike_score,
        "spectrogram_high_noise_score": high_noise_score,
        "spectrogram_metallic_score": metallic_score,
        "spectrogram_muffled_score": muffled_score,
        "spectrogram_rumble_score": rumble_score,
        "spectrogram_balance_score": balance_score,
        "spectrogram_artifact_score": artifact_score,
    }
    return {f"{prefix}{key}": round(float(value), 6) for key, value in raw.items()}
