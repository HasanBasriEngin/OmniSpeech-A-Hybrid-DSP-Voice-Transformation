from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from backend.audio.features import stretch_to_length
from backend.audio.filtering import apply_post_filter
from backend.audio.voice_analysis import analyze_pitch_confidence, confidence_to_sample_mask


@dataclass(frozen=True)
class SpeechRegionMasks:
    voiced: np.ndarray
    unvoiced: np.ndarray
    silence: np.ndarray
    transient: np.ndarray
    blend: np.ndarray


@dataclass(frozen=True)
class PureDSPCleanResult:
    audio: np.ndarray
    masks: SpeechRegionMasks
    metrics: dict[str, float]


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _local_rms(audio: np.ndarray, sample_rate: int, milliseconds: float) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size == 0:
        return x
    window = max(3, int(sample_rate * milliseconds / 1000.0))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.sqrt(np.convolve(x * x, kernel, mode="same").clip(min=0.0)).astype(np.float32)


def _band_envelope(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size < 64 or sample_rate <= 0:
        return np.zeros_like(x)
    nyquist = sample_rate / 2.0
    low = max(20.0, min(low_hz, nyquist * 0.92))
    high = max(low + 1.0, min(high_hz, nyquist * 0.98))
    if low >= high:
        return np.zeros_like(x)
    try:
        sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        band = signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return np.zeros_like(x)
    return _local_rms(band, sample_rate, 18.0)


def _smooth_mask(mask: np.ndarray, sample_rate: int, milliseconds: float = 18.0) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if values.size == 0:
        return values
    window = max(3, int(sample_rate * milliseconds / 1000.0))
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window).astype(np.float32)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 1e-8:
        return values
    kernel /= kernel_sum
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def phase_align_to_guide(
    guide: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    max_lag_ms: float = 3.0,
    min_correlation: float = 0.12,
) -> np.ndarray:
    guide_x = _as_mono_float(guide)
    target_x = stretch_to_length(_as_mono_float(target), guide_x.size)
    if guide_x.size < 128 or sample_rate <= 0:
        return target_x

    max_lag = min(max(1, int(round(sample_rate * max_lag_ms / 1000.0))), guide_x.size // 8)
    if max_lag <= 0:
        return target_x

    g = guide_x - float(np.mean(guide_x))
    t = target_x - float(np.mean(target_x))
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = g[:lag]
            right = t[-lag:]
        elif lag > 0:
            left = g[lag:]
            right = t[:-lag]
        else:
            left = g
            right = t
        if left.size < 64 or right.size < 64:
            continue
        denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-8)
        score = float(np.dot(left, right) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag == 0 or best_score < min_correlation:
        return target_x
    if best_lag > 0:
        return np.concatenate([np.zeros(best_lag, dtype=np.float32), target_x[:-best_lag]]).astype(np.float32)
    shift = -best_lag
    return np.concatenate([target_x[shift:], np.zeros(shift, dtype=np.float32)]).astype(np.float32)


def analyze_speech_regions(audio: np.ndarray, sample_rate: int) -> SpeechRegionMasks:
    x = _as_mono_float(audio)
    if x.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return SpeechRegionMasks(empty, empty, empty, empty, empty)

    rms = _local_rms(x, sample_rate, 22.0)
    global_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    noise_floor = float(np.percentile(rms, 18)) if rms.size else 0.0
    if global_rms > 1e-7 and noise_floor > global_rms * 0.45:
        active_threshold = max(global_rms * 0.12, 1e-4)
    else:
        active_threshold = max(noise_floor * 2.15, global_rms * 0.16, 1e-4)
    active = rms > active_threshold

    pitch_track = analyze_pitch_confidence(x, sample_rate)
    f0_voiced = confidence_to_sample_mask(pitch_track, x.size, threshold=0.34) > 0.35

    low_mid = _band_envelope(x, sample_rate, 85.0, 3_400.0)
    fricative = _band_envelope(x, sample_rate, 3_800.0, 9_500.0)
    tonal_voiced = (low_mid > fricative * 0.72) & (low_mid > global_rms * 0.08)
    voiced = active & (tonal_voiced | f0_voiced)
    unvoiced = active & ~voiced
    silence = ~active

    diff = np.concatenate([[0.0], np.abs(np.diff(x))]).astype(np.float32)
    transient_threshold = max(float(np.percentile(diff, 96)), global_rms * 0.55, 0.015)
    transient = (diff > transient_threshold) & active

    blend = np.zeros_like(x, dtype=np.float32)
    blend[voiced] = 1.0
    blend[unvoiced] = 0.34
    blend[silence] = 0.08
    blend[transient] = np.minimum(blend[transient], 0.48)
    blend = np.clip(_smooth_mask(blend, sample_rate, milliseconds=22.0), 0.0, 1.0)

    return SpeechRegionMasks(
        voiced=voiced.astype(np.float32),
        unvoiced=unvoiced.astype(np.float32),
        silence=silence.astype(np.float32),
        transient=transient.astype(np.float32),
        blend=blend.astype(np.float32),
    )


def _rms_normalize(
    audio: np.ndarray,
    target_rms: float = 0.075,
    peak_ceiling: float = 0.92,
    sample_rate: int | None = None,
) -> np.ndarray:
    x = _as_mono_float(audio)
    if x.size == 0:
        return x
    level_source = x
    if sample_rate is not None and sample_rate > 0 and x.size >= 64:
        envelope = _local_rms(x, sample_rate, 24.0)
        global_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        threshold = max(float(np.percentile(envelope, 55)) * 0.55, global_rms * 0.18, 1e-5)
        active = envelope > threshold
        if int(np.count_nonzero(active)) >= max(16, x.size // 20):
            level_source = x[active]

    rms = float(np.sqrt(np.mean(level_source * level_source))) + 1e-8
    scaled = x * float(target_rms / rms)
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > peak_ceiling:
        scaled = scaled * float(peak_ceiling / peak)
    return np.asarray(scaled, dtype=np.float32)


def prepare_clean_speech(audio: np.ndarray, sample_rate: int) -> PureDSPCleanResult:
    x = _as_mono_float(audio)
    if x.size == 0:
        masks = analyze_speech_regions(x, sample_rate)
        return PureDSPCleanResult(audio=x, masks=masks, metrics={"pure_dsp_input_cleaned": 1.0})

    before_peak = float(np.max(np.abs(x)))
    before_rms = float(np.sqrt(np.mean(x * x)))

    x = x - float(np.mean(x))
    x = apply_post_filter(
        x,
        sample_rate,
        speech_band=True,
        declick=True,
        soft_limit=False,
        deess=True,
        use_noisereduce=False,
        use_pedalboard=False,
        deess_reduction_db=3.0,
    )

    masks = analyze_speech_regions(x, sample_rate)

    x = _rms_normalize(x, sample_rate=sample_rate)
    x = apply_post_filter(
        x,
        sample_rate,
        speech_band=False,
        declick=True,
        soft_limit=True,
        deess=True,
        use_noisereduce=False,
        use_pedalboard=False,
        deess_reduction_db=2.0,
        ceiling=0.94,
        knee_db=5.0,
    )

    masks = analyze_speech_regions(x, sample_rate)
    after_peak = float(np.max(np.abs(x))) if x.size else 0.0
    after_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    metrics = {
        "pure_dsp_input_cleaned": 1.0,
        "pure_dsp_silence_preserved": 1.0,
        "pure_dsp_pre_peak": round(before_peak, 6),
        "pure_dsp_pre_rms": round(before_rms, 6),
        "pure_dsp_clean_peak": round(after_peak, 6),
        "pure_dsp_clean_rms": round(after_rms, 6),
        "pure_dsp_voiced_ratio": round(float(np.mean(masks.voiced)) if masks.voiced.size else 0.0, 6),
        "pure_dsp_unvoiced_ratio": round(float(np.mean(masks.unvoiced)) if masks.unvoiced.size else 0.0, 6),
        "pure_dsp_silence_ratio": round(float(np.mean(masks.silence)) if masks.silence.size else 0.0, 6),
        "pure_dsp_transient_ratio": round(float(np.mean(masks.transient)) if masks.transient.size else 0.0, 6),
    }
    return PureDSPCleanResult(audio=np.asarray(x, dtype=np.float32), masks=masks, metrics=metrics)


def merge_preserving_noise_regions(
    clean_source: np.ndarray,
    converted: np.ndarray,
    sample_rate: int,
    *,
    intensity: float = 1.0,
    smoothing: float = 1.0,
    preserve_length: bool = True,
) -> np.ndarray:
    source = _as_mono_float(clean_source)
    converted_target = _as_mono_float(converted)
    target_length = source.size if preserve_length else converted_target.size
    target = stretch_to_length(converted_target, target_length)
    guide = stretch_to_length(source, target_length)
    if guide.size == 0:
        return target

    target = phase_align_to_guide(guide, target, sample_rate, max_lag_ms=3.0)
    masks = analyze_speech_regions(guide, sample_rate)
    blend = masks.blend
    smooth_ms = float(np.clip(16.0 * max(smoothing, 0.35), 8.0, 42.0))
    blend = _smooth_mask(blend, sample_rate, milliseconds=smooth_ms)
    blend = np.clip(blend * float(np.clip(intensity, 0.35, 1.18)), 0.0, 1.0)
    blend = np.clip(blend * blend * (3.0 - 2.0 * blend), 0.0, 1.0)
    return ((1.0 - blend) * guide + blend * target).astype(np.float32)
