from __future__ import annotations

import numpy as np

from backend.audio.features import pitch_shift_audio, stretch_to_length


CELEBRITY_PROFILES: dict[str, dict[str, float]] = {
    "michael_jackson": {
        "pitch_shift": 4.2,
        "vibrato_rate": 5.8,
        "vibrato_depth": 0.6,
        "breathiness": 0.15,
        "formant_shift": 1.12,
        "brightness": 0.4,
    },
    "morgan_freeman": {
        "pitch_shift": -3.5,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
        "breathiness": 0.08,
        "formant_shift": 0.88,
        "brightness": -0.2,
    },
    "adele": {
        "pitch_shift": 1.7,
        "vibrato_rate": 5.0,
        "vibrato_depth": 0.25,
        "breathiness": 0.05,
        "formant_shift": 1.06,
        "brightness": 0.15,
    },
    "james_earl_jones": {
        "pitch_shift": -4.8,
        "vibrato_rate": 1.3,
        "vibrato_depth": 0.12,
        "breathiness": 0.04,
        "formant_shift": 0.82,
        "brightness": -0.28,
    },
    "taylor_swift": {
        "pitch_shift": 2.8,
        "vibrato_rate": 5.4,
        "vibrato_depth": 0.22,
        "breathiness": 0.06,
        "formant_shift": 1.08,
        "brightness": 0.24,
    },
}


def _apply_formant_warp(audio: np.ndarray, factor: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 16:
        return x

    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    idx = np.arange(magnitude.size, dtype=np.float32)
    src_idx = np.clip(idx / max(factor, 1e-4), 0.0, magnitude.size - 1.0)
    warped_mag = np.interp(src_idx, idx, magnitude).astype(np.float32)

    warped_spec = warped_mag * np.exp(1j * phase)
    return np.fft.irfft(warped_spec, n=x.size).astype(np.float32)


def _apply_vibrato(audio: np.ndarray, sample_rate: int, rate_hz: float, depth_semitones: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 1024 or rate_hz <= 0.0 or depth_semitones <= 0.0:
        return x

    t = np.arange(x.size, dtype=np.float32) / max(sample_rate, 1)
    lfo = np.sin(2 * np.pi * rate_hz * t).astype(np.float32)
    max_delay = max(1, int(sample_rate * 0.0035))
    delay = (0.5 * max_delay * (1.0 + lfo * (depth_semitones / 2.0))).astype(np.float32)
    read_pos = np.arange(x.size, dtype=np.float32) - delay
    read_pos = np.clip(read_pos, 0.0, x.size - 1.0)
    return np.interp(read_pos, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


def _apply_timbre(audio: np.ndarray, brightness: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    shelf = 1.0 + brightness * np.clip((freqs - 0.2) / 0.8, 0.0, 1.0)
    tilt = 1.0 + brightness * 0.35 * (freqs - 0.4)
    curve = np.clip(shelf * tilt, 0.25, 2.5)
    return np.fft.irfft(spec * curve, n=x.size).astype(np.float32)


def _apply_breathiness(audio: np.ndarray, amount: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if amount <= 0.0:
        return x

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, amount * 0.08, size=x.size).astype(np.float32)
    derivative = np.concatenate([[0.0], np.diff(x)]).astype(np.float32)
    airy = 0.65 * derivative + 0.35 * noise
    return (0.88 * x + 0.12 * airy).astype(np.float32)


def _peak_normalize(audio: np.ndarray, peak_target: float = 0.97) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-8:
        return x
    return (x * (peak_target / peak)).astype(np.float32)


def convert_celebrity(audio: np.ndarray, sample_rate: int, celebrity: str) -> np.ndarray:
    profile = CELEBRITY_PROFILES.get(celebrity)
    if profile is None:
        allowed = ", ".join(sorted(CELEBRITY_PROFILES))
        raise ValueError(f"Unsupported celebrity '{celebrity}'. Allowed: {allowed}")

    source = np.asarray(audio, dtype=np.float32)
    if source.size == 0:
        return source

    pitched = pitch_shift_audio(source, sample_rate, profile["pitch_shift"])
    pitched = stretch_to_length(np.asarray(pitched, dtype=np.float32), source.size)

    warped = _apply_formant_warp(pitched, profile["formant_shift"])
    vibrato = _apply_vibrato(warped, sample_rate, profile["vibrato_rate"], profile["vibrato_depth"])
    timbre = _apply_timbre(vibrato, profile["brightness"])
    airy = _apply_breathiness(timbre, profile["breathiness"])
    output = _peak_normalize(np.tanh(airy * 1.08))
    return np.clip(output, -1.0, 1.0).astype(np.float32)
