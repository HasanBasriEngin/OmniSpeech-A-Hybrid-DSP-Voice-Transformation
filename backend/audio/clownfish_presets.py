from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from backend.audio.features import pitch_shift_audio, stretch_to_length, time_stretch_audio
from backend.audio.pure_dsp import analyze_speech_regions


@dataclass(frozen=True)
class ClownfishStylePreset:
    key: str
    label: str
    effect_id: int
    pitch_semitones: float = 0.0
    formant_ratio: float = 1.0
    rate: float = 1.0
    brightness: float = 0.0
    warmth: float = 0.0


PRESETS: dict[str, ClownfishStylePreset] = {
    "none": ClownfishStylePreset("none", "None", 0),
    "male_pitch": ClownfishStylePreset("male_pitch", "Male pitch", 7, -5.0, 0.82, 1.0, -0.10, 0.25),
    "female_pitch": ClownfishStylePreset("female_pitch", "Female pitch", 9, 4.5, 1.18, 1.0, 0.18, -0.04),
    "helium_pitch": ClownfishStylePreset("helium_pitch", "Helium pitch", 10, 8.0, 1.26, 1.0, 0.30, -0.18),
    "baby_pitch": ClownfishStylePreset("baby_pitch", "Baby pitch", 11, 7.0, 1.34, 1.04, 0.28, -0.12),
    "radio": ClownfishStylePreset("radio", "Radio", 12, 0.0, 1.0, 1.0, 0.24, -0.20),
    "robot": ClownfishStylePreset("robot", "Robot", 14, 0.0, 1.0, 1.0, 0.10, -0.05),
    "clone": ClownfishStylePreset("clone", "Clone", 3, 0.0, 1.0, 1.0, 0.04, 0.04),
    "custom_pitch": ClownfishStylePreset("custom_pitch", "Custom pitch", 13, 0.0, 1.0, 1.0, 0.0, 0.0),
}


ALIASES = {
    "male": "male_pitch",
    "female": "female_pitch",
    "helium": "helium_pitch",
    "baby": "baby_pitch",
    "custom": "custom_pitch",
}


def normalize_clownfish_preset_key(value: str | None) -> str:
    if value is None:
        return "none"
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in PRESETS:
        allowed = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unsupported Clownfish preset '{value}'. Allowed: {allowed}")
    return key


def get_clownfish_preset(value: str | None) -> ClownfishStylePreset:
    return PRESETS[normalize_clownfish_preset_key(value)]


def effective_pitch_semitones(preset_key: str | None, custom_pitch: float | None = None) -> float:
    preset = get_clownfish_preset(preset_key)
    if preset.key == "custom_pitch" and custom_pitch is not None:
        return float(np.clip(custom_pitch, -15.0, 15.0))
    return preset.pitch_semitones


def list_clownfish_presets() -> list[dict[str, object]]:
    return [
        {
            "key": preset.key,
            "label": preset.label,
            "effect_id": preset.effect_id,
            "pitch_semitones": preset.pitch_semitones,
        }
        for preset in PRESETS.values()
    ]


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _tone_shape(audio: np.ndarray, brightness: float, warmth: float) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size < 16 or (abs(brightness) < 1e-4 and abs(warmth) < 1e-4):
        return x

    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    high = 1.0 + brightness * np.clip((freqs - 0.18) / 0.68, 0.0, 1.0)
    low = 1.0 + warmth * np.exp(-((freqs - 0.08) ** 2) / (2 * 0.085**2))
    presence = 1.0 + max(brightness, 0.0) * 0.24 * np.exp(-((freqs - 0.38) ** 2) / (2 * 0.07**2))
    curve = np.clip(high * low * presence, 0.22, 2.2)
    return np.fft.irfft(spec * curve, n=x.size).astype(np.float32)


def _formant_envelope_warp(audio: np.ndarray, ratio: float, amount: float = 0.35) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size < 256 or abs(ratio - 1.0) < 1e-4:
        return x

    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum).astype(np.float32) + 1e-7
    phase = np.angle(spectrum)
    log_mag = np.log(magnitude)
    cepstrum = np.fft.irfft(log_mag, n=(magnitude.size - 1) * 2).astype(np.float32)

    lifter = max(10, min(48, cepstrum.size // 20))
    smooth = np.zeros_like(cepstrum)
    smooth[:lifter] = cepstrum[:lifter]
    smooth[-lifter + 1 :] = cepstrum[-lifter + 1 :]
    envelope = np.exp(np.fft.rfft(smooth).real).astype(np.float32)
    detail = magnitude / np.maximum(envelope, 1e-7)

    idx = np.arange(envelope.size, dtype=np.float32)
    source_idx = np.clip(idx / max(float(ratio), 1e-4), 0.0, envelope.size - 1.0)
    warped = np.interp(source_idx, idx, envelope).astype(np.float32)
    blended = envelope * (1.0 - amount) + warped * amount
    rebuilt = np.clip(blended * detail, 1e-7, np.max(magnitude) * 2.5)
    return np.fft.irfft(rebuilt * np.exp(1j * phase), n=x.size).astype(np.float32)


def _radio_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size < 64 or sample_rate <= 0:
        return x

    nyquist = sample_rate / 2.0
    high = min(3400.0, nyquist * 0.96)
    low = min(280.0, high * 0.5)
    try:
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        x = signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        pass
    return np.tanh(x * 1.65).astype(np.float32)


def _robot_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size == 0 or sample_rate <= 0:
        return x
    t = np.arange(x.size, dtype=np.float32) / float(sample_rate)
    carrier = 0.72 + 0.28 * np.sign(np.sin(2 * np.pi * 52.0 * t)).astype(np.float32)
    return np.tanh(x * carrier * 1.35).astype(np.float32)


def _clone_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size == 0:
        return x
    low = pitch_shift_audio(x, sample_rate, -2.4)
    high = pitch_shift_audio(x, sample_rate, 2.1)
    delay = max(1, int(round(sample_rate * 0.012)))
    delayed = np.pad(high, (delay, 0), mode="constant")[: x.size]
    return (0.58 * x + 0.24 * low + 0.18 * delayed).astype(np.float32)


def _blend_voiced_regions(source: np.ndarray, transformed: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    x = _as_mono_float(source)
    y = stretch_to_length(_as_mono_float(transformed), x.size)
    if x.size == 0:
        return y

    try:
        masks = analyze_speech_regions(x, sample_rate)
        blend = np.clip(masks.blend * float(strength), 0.0, 1.0)
    except Exception:
        blend = np.full(x.size, float(np.clip(strength, 0.0, 1.0)), dtype=np.float32)
    return ((1.0 - blend) * x + blend * y).astype(np.float32)


def _peak_guard(audio: np.ndarray, peak_target: float = 0.93) -> np.ndarray:
    x = _as_mono_float(audio)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= peak_target or peak <= 1e-7:
        return x
    return (x * (peak_target / peak)).astype(np.float32)


def apply_clownfish_preset(
    audio: np.ndarray,
    sample_rate: int,
    preset_key: str | None,
    *,
    custom_pitch: float | None = None,
) -> np.ndarray:
    preset = get_clownfish_preset(preset_key)
    source = _as_mono_float(audio)
    if source.size == 0 or preset.key == "none":
        return source

    if preset.key == "radio":
        transformed = _radio_effect(source, sample_rate)
        return _peak_guard(_blend_voiced_regions(source, transformed, sample_rate, 0.92))

    if preset.key == "robot":
        transformed = _robot_effect(source, sample_rate)
        return _peak_guard(_blend_voiced_regions(source, transformed, sample_rate, 0.82))

    if preset.key == "clone":
        transformed = _clone_effect(source, sample_rate)
        return _peak_guard(_blend_voiced_regions(source, transformed, sample_rate, 0.86))

    pitch = effective_pitch_semitones(preset.key, custom_pitch)
    transformed = pitch_shift_audio(source, sample_rate, pitch)
    if abs(preset.rate - 1.0) > 1e-4:
        transformed = stretch_to_length(time_stretch_audio(transformed, preset.rate), source.size)
    transformed = _formant_envelope_warp(transformed, preset.formant_ratio)
    transformed = _tone_shape(transformed, preset.brightness, preset.warmth)

    strength = 0.94 if preset.key == "custom_pitch" else 0.90
    return _peak_guard(_blend_voiced_regions(source, transformed, sample_rate, strength))
