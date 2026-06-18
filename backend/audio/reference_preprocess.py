from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import signal
import soundfile as sf

from backend.audio.io import normalize_audio
from backend.audio.reference_quality import ReferenceQualityReport, analyze_reference_quality, quality_report_to_metrics


@dataclass(frozen=True)
class PreparedReference:
    audio: np.ndarray
    sample_rate: int
    source_index: int
    quality: ReferenceQualityReport
    metrics: dict[str, float]


def _trim_silence(audio: np.ndarray, sample_rate: int, *, frame_ms: float = 25.0, threshold_db: float = -38.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < sample_rate // 10:
        return x

    frame_length = max(256, int(sample_rate * frame_ms / 1000.0))
    hop = max(64, frame_length // 2)
    peak = float(np.max(np.abs(x)) + 1e-8)
    linear_threshold = peak * (10.0 ** (threshold_db / 20.0))

    voiced: list[np.ndarray] = []
    for start in range(0, max(1, x.size - frame_length + 1), hop):
        frame = x[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        if float(np.max(np.abs(frame))) >= linear_threshold:
            voiced.append(x[start : start + frame_length])

    if not voiced:
        return x

    start_idx = 0
    for start in range(0, max(1, x.size - frame_length + 1), hop):
        frame = x[start : start + frame_length]
        if float(np.max(np.abs(frame))) >= linear_threshold:
            start_idx = start
            break

    end_idx = x.size
    for start in range(max(0, x.size - frame_length), 0, -hop):
        frame = x[start : start + frame_length]
        if float(np.max(np.abs(frame))) >= linear_threshold:
            end_idx = min(x.size, start + frame_length)
            break

    margin = int(sample_rate * 0.05)
    start_idx = max(0, start_idx - margin)
    end_idx = min(x.size, end_idx + margin)
    if end_idx <= start_idx + sample_rate // 20:
        return x
    return x[start_idx:end_idx].astype(np.float32)


def _light_noise_reduction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 64 or sample_rate <= 0:
        return x

    nyquist = sample_rate / 2.0
    low_cut = min(90.0, nyquist * 0.95)
    b, a = signal.butter(2, low_cut / nyquist, btype="high")
    try:
        highpassed = signal.filtfilt(b, a, x)
    except ValueError:
        return x

    blend = 0.82 * x + 0.18 * highpassed
    return np.clip(blend, -1.0, 1.0).astype(np.float32)


def preprocess_reference_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    trimmed = _trim_silence(x, sample_rate)
    denoised = _light_noise_reduction(trimmed, sample_rate)
    normalized = normalize_audio(denoised, mode="rms", target_rms=0.12)
    return np.asarray(normalized, dtype=np.float32)


def select_best_reference_index(reports: list[ReferenceQualityReport]) -> int:
    if not reports:
        return 0
    return int(max(range(len(reports)), key=lambda idx: reports[idx].quality_score))


def prepare_reference_from_paths(
    reference_paths: list[str],
    sample_rate: int,
    load_fn,
) -> PreparedReference | None:
    if not reference_paths:
        return None

    loaded: list[np.ndarray] = []
    reports: list[ReferenceQualityReport] = []
    for path in reference_paths:
        audio = np.asarray(load_fn(path, sample_rate), dtype=np.float32)
        loaded.append(audio)
        reports.append(analyze_reference_quality(audio, sample_rate))

    best_index = select_best_reference_index(reports)
    processed = preprocess_reference_audio(loaded[best_index], sample_rate)
    quality = analyze_reference_quality(processed, sample_rate)

    metrics = quality_report_to_metrics(quality, best_index)
    metrics["reference_preprocess_applied"] = 1.0
    metrics["reference_candidate_count"] = float(len(reference_paths))

    return PreparedReference(
        audio=processed,
        sample_rate=sample_rate,
        source_index=best_index,
        quality=quality,
        metrics=metrics,
    )


def write_reference_wav(audio: np.ndarray, sample_rate: int, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"freevc_ref_{uuid4().hex}.wav"
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate)
    return path
