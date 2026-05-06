from __future__ import annotations

import numpy as np

from backend.audio.features import pitch_shift_audio, stretch_to_length, time_stretch_audio


EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "sad": {
        "pitch_shift": -3.6,
        "rate": 0.86,
        "spectral_tilt": -2.8,
        "prosody_depth": 0.14,
        "drive": 0.92,
        "target_rms": 0.085,
    },
    "angry": {
        "pitch_shift": 4.8,
        "rate": 1.08,
        "spectral_tilt": 3.8,
        "prosody_depth": 0.32,
        "drive": 1.75,
        "target_rms": 0.19,
    },
    "excited": {
        "pitch_shift": 5.4,
        "rate": 1.14,
        "spectral_tilt": 2.9,
        "prosody_depth": 0.24,
        "drive": 1.45,
        "target_rms": 0.155,
    },
    "whisper": {
        "pitch_shift": -0.8,
        "rate": 0.95,
        "spectral_tilt": 1.2,
        "prosody_depth": 0.03,
        "drive": 0.7,
        "target_rms": 0.05,
    },
    "calm": {
        "pitch_shift": -1.2,
        "rate": 0.95,
        "spectral_tilt": -1.0,
        "prosody_depth": 0.07,
        "drive": 0.88,
        "target_rms": 0.095,
    },
}


def _apply_spectral_tilt(audio: np.ndarray, tilt_strength: float, presence_boost: float) -> np.ndarray:
    spec = np.fft.rfft(np.asarray(audio, dtype=np.float32))
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    presence = 1.0 + presence_boost * np.exp(-((freqs - 0.42) ** 2) / (2 * 0.07**2))
    curve = np.clip((1.0 + (tilt_strength * 0.45) * (freqs - 0.32)) * presence, 0.35, 2.3)
    return np.fft.irfft(spec * curve, n=audio.size).astype(np.float32)


def _apply_prosody(audio: np.ndarray, sample_rate: int, depth: float, emotion: str) -> np.ndarray:
    if audio.size == 0 or depth <= 0:
        return np.asarray(audio, dtype=np.float32)

    t = np.arange(audio.size, dtype=np.float32) / max(sample_rate, 1)
    mod_hz = 1.4 if emotion == "sad" else 2.8 if emotion == "calm" else 4.6
    modulation = 1.0 + depth * np.sin(2 * np.pi * mod_hz * t)
    return (np.asarray(audio, dtype=np.float32) * modulation).astype(np.float32)


def _match_target_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)) + 1e-8)
    scaled = np.asarray(audio, dtype=np.float32) * (target_rms / current_rms)
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 0.98:
        scaled = scaled * (0.98 / peak)
    return np.asarray(scaled, dtype=np.float32)


def convert_emotion(
    audio: np.ndarray,
    sample_rate: int,
    emotion: str,
    pitch_override: float | None = None,
    rate_override: float | None = None,
    energy_override: float | None = None,
) -> np.ndarray:
    profile = EMOTION_PROFILES.get(emotion)
    if profile is None:
        allowed = ", ".join(sorted(EMOTION_PROFILES))
        raise ValueError(f"Unsupported emotion '{emotion}'. Allowed: {allowed}")

    source = np.asarray(audio, dtype=np.float32)
    if source.size == 0:
        return source

    pitch_shift = float(profile["pitch_shift"] if pitch_override is None else np.clip(pitch_override, -8.0, 8.0))
    rate = float(profile["rate"] if rate_override is None else np.clip(rate_override, 0.6, 1.5))
    energy = float(1.0 if energy_override is None else np.clip(energy_override, 0.2, 2.0))

    pitched = pitch_shift_audio(source, sample_rate, pitch_shift)

    if source.size >= 2048:
        stretched = time_stretch_audio(np.asarray(pitched, dtype=np.float32), rate=rate)
    else:
        stretched = np.asarray(pitched, dtype=np.float32)

    timed = stretch_to_length(np.asarray(stretched, dtype=np.float32), source.size)
    expressive = _apply_prosody(timed, sample_rate, profile["prosody_depth"], emotion)
    driven = np.tanh(np.asarray(expressive * profile["drive"], dtype=np.float32))
    presence_boost = 0.18 if emotion == "sad" else 0.52 if emotion == "angry" else 0.4 if emotion == "excited" else 0.12
    tilted = _apply_spectral_tilt(driven, profile["spectral_tilt"], presence_boost)
    scaled = _match_target_rms(tilted, profile["target_rms"] * energy)

    if emotion == "whisper":
        breath = np.concatenate([[0.0], np.diff(scaled)]).astype(np.float32)
        scaled = 0.82 * scaled + 0.18 * breath

    return np.clip(scaled, -1.0, 1.0).astype(np.float32)
