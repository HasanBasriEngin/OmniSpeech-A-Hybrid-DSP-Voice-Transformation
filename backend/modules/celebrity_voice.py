from __future__ import annotations

import numpy as np

from backend.audio.features import pitch_shift_audio, stretch_to_length


CELEBRITY_PROFILES: dict[str, dict[str, float]] = {
    "michael_jackson": {
        "pitch_shift": 2.4,
        "vibrato_rate": 5.4,
        "vibrato_depth": 0.12,
        "breathiness": 0.025,
        "formant_shift": 1.06,
        "brightness": 0.16,
    },
    "morgan_freeman": {
        "pitch_shift": -2.4,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
        "breathiness": 0.014,
        "formant_shift": 0.92,
        "brightness": -0.12,
    },
    "adele": {
        "pitch_shift": 1.1,
        "vibrato_rate": 5.0,
        "vibrato_depth": 0.10,
        "breathiness": 0.012,
        "formant_shift": 1.03,
        "brightness": 0.08,
    },
    "james_earl_jones": {
        "pitch_shift": -3.2,
        "vibrato_rate": 1.3,
        "vibrato_depth": 0.04,
        "breathiness": 0.010,
        "formant_shift": 0.90,
        "brightness": -0.16,
    },
    "taylor_swift": {
        "pitch_shift": 1.6,
        "vibrato_rate": 5.2,
        "vibrato_depth": 0.09,
        "breathiness": 0.012,
        "formant_shift": 1.04,
        "brightness": 0.10,
    },
}


def _apply_formant_warp(audio: np.ndarray, factor: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 16:
        return x

    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum).astype(np.float32) + 1e-7
    phase = np.angle(spectrum)
    log_mag = np.log(magnitude)
    cepstrum = np.fft.irfft(log_mag, n=(magnitude.size - 1) * 2).astype(np.float32)
    lifter = max(10, min(44, cepstrum.size // 20))
    smooth = np.zeros_like(cepstrum)
    smooth[:lifter] = cepstrum[:lifter]
    smooth[-lifter + 1 :] = cepstrum[-lifter + 1 :]
    envelope = np.exp(np.fft.rfft(smooth).real).astype(np.float32)
    detail = magnitude / np.maximum(envelope, 1e-7)

    idx = np.arange(envelope.size, dtype=np.float32)
    src_idx = np.clip(idx / max(factor, 1e-4), 0.0, envelope.size - 1.0)
    warped_env = np.interp(src_idx, idx, envelope).astype(np.float32)
    blended_env = 0.72 * envelope + 0.28 * warped_env
    rebuilt = np.clip(blended_env * detail, 1e-7, np.max(magnitude) * 2.4)
    return np.fft.irfft(rebuilt * np.exp(1j * phase), n=x.size).astype(np.float32)


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
    curve = np.clip(shelf * tilt, 0.70, 1.35)
    return np.fft.irfft(spec * curve, n=x.size).astype(np.float32)


def _apply_breathiness(audio: np.ndarray, amount: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if amount <= 0.0:
        return x

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, amount * 0.020, size=x.size).astype(np.float32)
    derivative = np.concatenate([[0.0], np.diff(x)]).astype(np.float32)
    airy = 0.72 * derivative + 0.28 * noise
    return ((1.0 - min(amount, 0.08)) * x + min(amount, 0.08) * airy).astype(np.float32)


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
    output = _peak_normalize(np.tanh(airy * 1.01), peak_target=0.93)
    return np.clip(output, -1.0, 1.0).astype(np.float32)
