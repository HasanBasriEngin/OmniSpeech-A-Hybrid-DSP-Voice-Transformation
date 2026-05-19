from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchConfidenceTrack:
    f0_hz: np.ndarray
    confidence: np.ndarray
    frame_centers: np.ndarray
    hop_length: int
    frame_length: int


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _praat_pitch_track(audio: np.ndarray, sample_rate: int, frame_length: int, hop_length: int) -> PitchConfidenceTrack | None:
    try:
        import parselmouth
    except Exception:
        return None

    x = _as_mono_float(audio)
    if x.size < frame_length or sample_rate <= 0:
        return None

    try:
        sound = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sample_rate)
        time_step = hop_length / float(sample_rate)
        pitch = sound.to_pitch_ac(time_step=time_step, pitch_floor=65.0, pitch_ceiling=1_050.0)
        values = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
        strengths = np.asarray(pitch.selected_array.get("strength", np.zeros_like(values)), dtype=np.float32)
    except Exception:
        return None

    if values.size == 0:
        return None
    confidence = np.clip(strengths, 0.0, 1.0).astype(np.float32)
    values = np.where(confidence >= 0.35, values, 0.0).astype(np.float32)
    centers = np.linspace(frame_length // 2, max(frame_length // 2, x.size - frame_length // 2), values.size, dtype=np.float32)
    return PitchConfidenceTrack(values, confidence, centers, hop_length, frame_length)


def _autocorr_pitch_track(audio: np.ndarray, sample_rate: int, frame_length: int, hop_length: int) -> PitchConfidenceTrack:
    x = _as_mono_float(audio)
    if x.size < 64 or sample_rate <= 0:
        empty = np.zeros(0 if x.size == 0 else 1, dtype=np.float32)
        centers = np.zeros_like(empty)
        return PitchConfidenceTrack(empty, empty, centers, hop_length, frame_length)

    frame_length = min(max(256, frame_length), x.size)
    hop_length = max(64, min(hop_length, frame_length // 2))
    fmin = 65.0
    fmax = 1_050.0
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_length - 1, int(sample_rate / fmin))
    if min_lag >= max_lag:
        empty = np.zeros(1, dtype=np.float32)
        return PitchConfidenceTrack(empty, empty, empty, hop_length, frame_length)

    window = np.hanning(frame_length).astype(np.float32)
    f0_values: list[float] = []
    confidence_values: list[float] = []
    centers: list[float] = []
    for start in range(0, max(1, x.size - frame_length + 1), hop_length):
        frame = x[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        frame = (frame - float(np.mean(frame))) * window
        energy = float(np.sqrt(np.mean(frame * frame)))
        centers.append(float(start + frame_length / 2))
        if energy < 1e-4:
            f0_values.append(0.0)
            confidence_values.append(0.0)
            continue

        autocorr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        total = float(autocorr[0]) + 1e-8
        search = autocorr[min_lag : max_lag + 1]
        if search.size == 0:
            f0_values.append(0.0)
            confidence_values.append(0.0)
            continue

        peak_offset = int(np.argmax(search))
        peak = float(search[peak_offset])
        confidence = float(np.clip(peak / total, 0.0, 1.0))
        if confidence < 0.22:
            f0_values.append(0.0)
            confidence_values.append(confidence)
            continue

        lag = min_lag + peak_offset
        f0_values.append(float(sample_rate / max(lag, 1)))
        confidence_values.append(confidence)

    return PitchConfidenceTrack(
        f0_hz=np.asarray(f0_values, dtype=np.float32),
        confidence=np.asarray(confidence_values, dtype=np.float32),
        frame_centers=np.asarray(centers, dtype=np.float32),
        hop_length=hop_length,
        frame_length=frame_length,
    )


def analyze_pitch_confidence(audio: np.ndarray, sample_rate: int, frame_length: int | None = None) -> PitchConfidenceTrack:
    frame_size = int(frame_length) if frame_length is not None else max(512, sample_rate // 25)
    hop = max(128, frame_size // 4)
    praat = _praat_pitch_track(audio, sample_rate, frame_size, hop)
    if praat is not None:
        return praat
    return _autocorr_pitch_track(audio, sample_rate, frame_size, hop)


def confidence_to_sample_mask(track: PitchConfidenceTrack, sample_count: int, *, threshold: float = 0.35) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)
    if track.confidence.size == 0 or track.frame_centers.size == 0:
        return np.zeros(sample_count, dtype=np.float32)
    voiced = ((track.f0_hz > 0.0) & (track.confidence >= threshold)).astype(np.float32)
    positions = np.arange(sample_count, dtype=np.float32)
    return np.interp(positions, track.frame_centers, voiced, left=float(voiced[0]), right=float(voiced[-1])).astype(np.float32)
