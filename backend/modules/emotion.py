from __future__ import annotations

import logging

import numpy as np

from backend.audio.features import pitch_shift_audio, stretch_to_length, time_stretch_audio

logger = logging.getLogger(__name__)


EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "sad": {
        "pitch_shift": -5.0,
        "rate": 0.78,
        "spectral_tilt": -3.4,
        "prosody_depth": 0.22,
        "drive": 0.82,
        "target_rms": 0.075,
        "attack": 0.55,
        "breath": 0.02,
        "jitter": 0.012,
        "vibrato_rate": 4.1,
        "vibrato_depth": 0.16,
    },
    "angry": {
        "pitch_shift": 5.2,
        "rate": 1.12,
        "spectral_tilt": 4.4,
        "prosody_depth": 0.40,
        "drive": 2.05,
        "target_rms": 0.205,
        "attack": 1.35,
        "breath": 0.01,
        "jitter": 0.018,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
    },
    "excited": {
        "pitch_shift": 5.8,
        "rate": 1.18,
        "spectral_tilt": 3.2,
        "prosody_depth": 0.34,
        "drive": 1.55,
        "target_rms": 0.16,
        "attack": 0.90,
        "breath": 0.012,
        "jitter": 0.035,
        "vibrato_rate": 5.0,
        "vibrato_depth": 0.12,
    },
    "whisper": {
        "pitch_shift": -1.0,
        "rate": 0.92,
        "spectral_tilt": 2.1,
        "prosody_depth": 0.08,
        "drive": 0.58,
        "target_rms": 0.045,
        "attack": 0.25,
        "breath": 0.55,
        "jitter": 0.006,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
    },
    "calm": {
        "pitch_shift": -1.4,
        "rate": 0.94,
        "spectral_tilt": -1.4,
        "prosody_depth": 0.04,
        "drive": 0.82,
        "target_rms": 0.09,
        "attack": 0.20,
        "breath": 0.0,
        "jitter": 0.0,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
    },
}

FORMANT_FACTORS: dict[str, tuple[float, float, float]] = {
    "sad": (0.92, 0.96, 0.98),
    "angry": (1.08, 1.07, 1.04),
    "excited": (1.06, 1.05, 1.03),
    "whisper": (1.02, 1.10, 1.16),
    "calm": (0.98, 0.99, 1.00),
}

EMOTION_SEEDS = {
    "sad": 17,
    "angry": 31,
    "excited": 47,
    "whisper": 61,
    "calm": 73,
}


def _estimate_pitch_period(audio: np.ndarray, sample_rate: int) -> int | None:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 256 or sample_rate <= 0:
        return None

    frame_size = min(x.size, max(1024, sample_rate // 5))
    start = max(0, (x.size - frame_size) // 2)
    frame = x[start : start + frame_size]
    frame = frame - float(np.mean(frame))
    if float(np.sqrt(np.mean(frame**2))) < 1e-5:
        return None

    corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
    min_lag = max(1, int(sample_rate / 480.0))
    max_lag = min(frame_size - 1, int(sample_rate / 70.0))
    if min_lag >= max_lag:
        return None

    search = corr[min_lag : max_lag + 1]
    lag = int(np.argmax(search)) + min_lag
    if float(corr[lag]) < 0.08 * float(corr[0] + 1e-8):
        return None
    return lag


def _psola_pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if source.size < 512 or abs(semitones) < 1e-4:
        return source

    period = _estimate_pitch_period(source, sample_rate)
    if period is None:
        return pitch_shift_audio(source, sample_rate, semitones)

    ratio = float(2.0 ** (semitones / 12.0))
    target_period = max(8, int(round(period / max(ratio, 1e-4))))
    radius = min(max(period, 32), max(32, source.size // 4))
    if radius * 2 + 1 >= source.size:
        return pitch_shift_audio(source, sample_rate, semitones)

    analysis_marks = np.arange(radius, source.size - radius, period, dtype=np.int32)
    synthesis_marks = np.arange(radius, source.size - radius, target_period, dtype=np.int32)
    if analysis_marks.size == 0 or synthesis_marks.size == 0:
        return pitch_shift_audio(source, sample_rate, semitones)

    window = np.hanning(radius * 2 + 1).astype(np.float32)
    output = np.zeros_like(source)
    weights = np.zeros_like(source)

    for synth_index, synth_mark in enumerate(synthesis_marks):
        analysis_index = min(int(round(synth_index / max(ratio, 1e-4))), analysis_marks.size - 1)
        analysis_mark = int(analysis_marks[analysis_index])
        source_start = analysis_mark - radius
        source_end = analysis_mark + radius + 1
        dest_start = int(synth_mark) - radius
        dest_end = int(synth_mark) + radius + 1

        if dest_start < 0 or dest_end > output.size:
            continue

        frame = source[source_start:source_end] * window
        output[dest_start:dest_end] += frame
        weights[dest_start:dest_end] += window

    valid = weights > 1e-6
    if not np.any(valid):
        return pitch_shift_audio(source, sample_rate, semitones)
    output[valid] /= weights[valid]
    output[~valid] = source[~valid]
    return output.astype(np.float32)


def _apply_vibrato(audio: np.ndarray, sample_rate: int, rate_hz: float, depth_semitones: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 512 or rate_hz <= 0.0 or depth_semitones <= 0.0:
        return x

    t = np.arange(x.size, dtype=np.float32) / max(sample_rate, 1)
    lfo = np.sin(2 * np.pi * rate_hz * t).astype(np.float32)
    max_delay = max(1, int(sample_rate * 0.003))
    delay = lfo * max_delay * (depth_semitones / 2.0)
    read_pos = np.clip(np.arange(x.size, dtype=np.float32) - delay, 0.0, x.size - 1.0)
    return np.interp(read_pos, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


def _apply_pitch_jitter(audio: np.ndarray, sample_rate: int, depth: float, emotion: str) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 512 or depth <= 0.0:
        return x

    rng = np.random.default_rng(EMOTION_SEEDS[emotion])
    control_count = max(4, int(round(x.size / max(sample_rate, 1) * 12)))
    control = rng.normal(0.0, depth, control_count).astype(np.float32)
    modulation = np.interp(
        np.linspace(0.0, control_count - 1, x.size, dtype=np.float32),
        np.arange(control_count, dtype=np.float32),
        control,
    ).astype(np.float32)
    rate = np.clip(1.0 + modulation, 0.92, 1.08)
    read_pos = np.cumsum(rate).astype(np.float32)
    read_pos = read_pos * ((x.size - 1) / max(float(read_pos[-1]), 1.0))
    return np.interp(read_pos, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


def _apply_spectral_tilt(audio: np.ndarray, tilt_strength: float, presence_boost: float) -> np.ndarray:
    spec = np.fft.rfft(np.asarray(audio, dtype=np.float32))
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    presence = 1.0 + presence_boost * np.exp(-((freqs - 0.42) ** 2) / (2 * 0.07**2))
    curve = np.clip((1.0 + (tilt_strength * 0.45) * (freqs - 0.32)) * presence, 0.35, 2.3)
    return np.fft.irfft(spec * curve, n=audio.size).astype(np.float32)


def _apply_formant_shift(audio: np.ndarray, sample_rate: int, factors: tuple[float, float, float]) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 32:
        return x

    spec = np.fft.rfft(x)
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    freqs = np.linspace(0.0, sample_rate / 2.0, magnitude.size, dtype=np.float32)
    source_freqs = freqs.copy()

    for center, width, factor in zip((700.0, 1250.0, 2600.0), (420.0, 620.0, 1100.0), factors):
        blend = np.exp(-((freqs - center) ** 2) / (2 * width**2)).astype(np.float32)
        source_freqs = source_freqs * (1.0 - blend) + (freqs / max(factor, 1e-4)) * blend

    warped = np.interp(np.clip(source_freqs, 0.0, sample_rate / 2.0), freqs, magnitude).astype(np.float32)
    return np.fft.irfft(warped * np.exp(1j * phase), n=x.size).astype(np.float32)


def _apply_prosody(audio: np.ndarray, sample_rate: int, depth: float, emotion: str) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0 or depth <= 0:
        return x

    frame_size = max(128, sample_rate // 40)
    hop = frame_size
    rms_values: list[float] = []
    for start in range(0, x.size, hop):
        frame = x[start : start + frame_size]
        rms_values.append(float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0)

    rms = np.asarray(rms_values, dtype=np.float32)
    if rms.size == 0 or float(np.max(rms)) <= 1e-7:
        base_envelope = np.ones(x.size, dtype=np.float32)
    else:
        rms = rms / (float(np.mean(rms)) + 1e-7)
        base_envelope = np.interp(
            np.linspace(0.0, max(rms.size - 1, 0), x.size, dtype=np.float32),
            np.arange(rms.size, dtype=np.float32),
            rms,
        ).astype(np.float32)

    rng = np.random.default_rng(EMOTION_SEEDS[emotion])
    control_count = max(4, int(round(x.size / max(sample_rate, 1) * 5)))
    control = rng.uniform(-1.0, 1.0, control_count).astype(np.float32)
    if emotion == "sad":
        control = 0.45 * control + np.linspace(0.15, -0.35, control_count, dtype=np.float32)
    elif emotion == "angry":
        control = np.maximum(control, 0.0) * 1.15
    elif emotion == "excited":
        control = control * 0.9 + np.abs(np.roll(control, 1)) * 0.35
    elif emotion == "calm":
        control = control * 0.18
    else:
        control = control * 0.35

    dynamic = np.interp(
        np.linspace(0.0, control_count - 1, x.size, dtype=np.float32),
        np.arange(control_count, dtype=np.float32),
        control,
    ).astype(np.float32)
    envelope = 1.0 + depth * (0.55 * (base_envelope - 1.0) + dynamic)
    return (x * np.clip(envelope, 0.45, 1.85)).astype(np.float32)


def _apply_attack_emphasis(audio: np.ndarray, amount: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 2 or amount <= 0.0:
        return x
    transient = np.concatenate([[0.0], np.diff(x)]).astype(np.float32)
    return (x + transient * amount * 0.12).astype(np.float32)


def _apply_breathiness(audio: np.ndarray, amount: float, emotion: str) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0 or amount <= 0.0:
        return x

    rng = np.random.default_rng(EMOTION_SEEDS[emotion] + 101)
    noise = rng.normal(0.0, 0.08, x.size).astype(np.float32)
    if noise.size > 8:
        smooth = np.convolve(noise, np.ones(9, dtype=np.float32) / 9.0, mode="same")
        noise = noise - smooth.astype(np.float32)

    derivative = np.concatenate([[0.0], np.diff(x)]).astype(np.float32)
    airy = 0.62 * derivative + 0.38 * noise
    return ((1.0 - amount) * x + amount * airy).astype(np.float32)


def _match_target_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)) + 1e-8)
    scaled = np.asarray(audio, dtype=np.float32) * (target_rms / current_rms)
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 0.98:
        scaled = scaled * (0.98 / peak)
    return np.asarray(scaled, dtype=np.float32)


def _pitch_shift_with_parselmouth(audio: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray | None:
    source = np.asarray(audio, dtype=np.float32)
    if source.size == 0:
        return source
    if abs(n_steps) < 1e-5:
        return source.copy()

    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return None

    try:
        factor = float(2.0 ** (n_steps / 12.0))
        sound = parselmouth.Sound(source.astype(np.float64), sampling_frequency=sample_rate)
        manipulation = call(sound, "To Manipulation", 0.01, 75.0, 600.0)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
        call([pitch_tier, manipulation], "Replace pitch tier")
        shifted = call(manipulation, "Get resynthesis (overlap-add)")
        values = np.asarray(shifted.values, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - depends on optional Praat bindings
        logger.warning("Parselmouth pitch shift failed; falling back to Librosa: %s", exc)
        return None

    if values.ndim > 1:
        values = values[0]
    return stretch_to_length(np.asarray(values, dtype=np.float32), source.size)


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

    pitched = _pitch_shift_with_parselmouth(source, sample_rate, pitch_shift)
    if pitched is None:
        pitched = _psola_pitch_shift(source, sample_rate, pitch_shift)
    pitched = _apply_pitch_jitter(pitched, sample_rate, profile["jitter"], emotion)
    pitched = _apply_vibrato(pitched, sample_rate, profile["vibrato_rate"], profile["vibrato_depth"])

    if source.size >= 2048:
        stretched = time_stretch_audio(np.asarray(pitched, dtype=np.float32), rate=rate)
    else:
        stretched = np.asarray(pitched, dtype=np.float32)

    timed = stretch_to_length(np.asarray(stretched, dtype=np.float32), source.size)
    expressive = _apply_prosody(timed, sample_rate, profile["prosody_depth"], emotion)
    formed = _apply_formant_shift(expressive, sample_rate, FORMANT_FACTORS[emotion])
    attacked = _apply_attack_emphasis(formed, profile["attack"])
    driven = np.tanh(np.asarray(attacked * profile["drive"], dtype=np.float32))
    presence_boost = 0.18 if emotion == "sad" else 0.52 if emotion == "angry" else 0.4 if emotion == "excited" else 0.12
    tilted = _apply_spectral_tilt(driven, profile["spectral_tilt"], presence_boost)
    scaled = _match_target_rms(tilted, profile["target_rms"] * energy)

    if profile["breath"] > 0:
        scaled = _apply_breathiness(scaled, profile["breath"], emotion)

    return np.clip(scaled, -1.0, 1.0).astype(np.float32)
