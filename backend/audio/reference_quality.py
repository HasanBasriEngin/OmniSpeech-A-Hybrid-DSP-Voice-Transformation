from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceQualityReport:
    quality_score: float
    duration_seconds: float
    silence_ratio: float
    clipping_ratio: float
    rms_db: float
    noise_score: float
    too_short: bool
    clipping_warning: bool
    low_quality_warning: bool


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return np.zeros(1, dtype=np.float32)

    frame_length = max(256, min(frame_length, x.size))
    hop_length = max(64, min(hop_length, frame_length))
    frames: list[float] = []
    for start in range(0, max(1, x.size - frame_length + 1), hop_length):
        frame = x[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        frames.append(float(np.sqrt(np.mean(frame**2) + 1e-12)))
    return np.asarray(frames, dtype=np.float32)


def _rms_to_db(rms: float) -> float:
    return float(20.0 * np.log10(max(rms, 1e-8)))


def _estimate_noise_score(audio: np.ndarray, sample_rate: int) -> float:
    """0 = temiz, 1 = gürültülü tahmin."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 256 or sample_rate <= 0:
        return 0.5

    frame_length = min(2048, x.size)
    hop = max(128, frame_length // 4)
    rms_frames = _frame_rms(x, frame_length, hop)
    if rms_frames.size == 0:
        return 0.5

    floor = float(np.percentile(rms_frames, 20))
    ceiling = float(np.percentile(rms_frames, 95) + 1e-8)
    dynamic_range = (ceiling - floor) / ceiling

    # Dusuk dinamik aralik ve yuksek zemin genelde gurultu/oda hum gostergesi.
    noise_from_floor = float(np.clip(floor / ceiling, 0.0, 1.0))

    diff = np.diff(x, prepend=x[:1])
    hf_energy = float(np.mean(diff**2))
    lf_energy = float(np.mean(x**2) + 1e-8)
    hf_ratio = float(np.clip(hf_energy / lf_energy, 0.0, 2.0))

    return float(np.clip(0.55 * noise_from_floor + 0.25 * (1.0 - dynamic_range) + 0.20 * min(hf_ratio, 1.0), 0.0, 1.0))


def analyze_reference_quality(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_recommended_duration: float = 10.0,
    min_usable_duration: float = 1.0,
    clipping_threshold: float = 0.98,
) -> ReferenceQualityReport:
    x = np.asarray(audio, dtype=np.float32)
    duration = float(x.size / max(sample_rate, 1))
    if x.size == 0 or sample_rate <= 0:
        return ReferenceQualityReport(
            quality_score=0.0,
            duration_seconds=0.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            rms_db=-80.0,
            noise_score=1.0,
            too_short=True,
            clipping_warning=False,
            low_quality_warning=True,
        )

    frame_length = min(2048, max(256, x.size // 8))
    hop = max(128, frame_length // 4)
    frame_rms = _frame_rms(x, frame_length, hop)
    peak_rms = float(np.max(frame_rms)) if frame_rms.size else 0.0
    silence_cutoff = max(peak_rms * 0.08, 1e-5)
    silence_ratio = float(np.mean(frame_rms < silence_cutoff)) if frame_rms.size else 1.0

    clipping_ratio = float(np.mean(np.abs(x) >= clipping_threshold))
    overall_rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    rms_db = _rms_to_db(overall_rms)
    noise_score = _estimate_noise_score(x, sample_rate)

    too_short = duration < min_usable_duration
    clipping_warning = clipping_ratio >= 0.01 or float(np.max(np.abs(x))) >= 0.999

    score = 1.0
    if duration < min_recommended_duration:
        short_penalty = (min_recommended_duration - duration) / min_recommended_duration
        score -= 0.45 * float(np.clip(short_penalty, 0.0, 1.0))
    if duration < 3.0:
        score -= 0.15

    score -= 0.30 * float(np.clip(silence_ratio, 0.0, 1.0))
    score -= 0.35 * float(np.clip(clipping_ratio * 25.0, 0.0, 1.0))
    if clipping_warning:
        score -= 0.10
    score -= 0.25 * float(np.clip(noise_score, 0.0, 1.0))

    if rms_db < -42.0:
        score -= 0.12
    elif rms_db > -6.0:
        score -= 0.08

    quality_score = float(np.clip(score, 0.0, 1.0))
    low_quality_warning = quality_score < 0.45 or too_short

    return ReferenceQualityReport(
        quality_score=quality_score,
        duration_seconds=duration,
        silence_ratio=float(round(silence_ratio, 4)),
        clipping_ratio=float(round(clipping_ratio, 6)),
        rms_db=float(round(rms_db, 3)),
        noise_score=float(round(noise_score, 4)),
        too_short=too_short,
        clipping_warning=clipping_warning,
        low_quality_warning=low_quality_warning,
    )


def quality_report_to_metrics(report: ReferenceQualityReport, index: int) -> dict[str, float]:
    return {
        "reference_quality_score": report.quality_score,
        "reference_duration_seconds": report.duration_seconds,
        "reference_silence_ratio": report.silence_ratio,
        "reference_clipping_ratio": report.clipping_ratio,
        "reference_rms_db": report.rms_db,
        "reference_noise_score": report.noise_score,
        "selected_reference_index": float(index),
        "reference_clipping_warning": 1.0 if report.clipping_warning else 0.0,
        "reference_low_quality_warning": 1.0 if report.low_quality_warning else 0.0,
    }
