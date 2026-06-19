from __future__ import annotations

import numpy as np
from backend.audio.features import extract_mfcc, extract_pitch_contour, pitch_shift_audio, stretch_to_length


def _smooth_spectrum(magnitude: np.ndarray, taps: int = 23) -> np.ndarray:
    values = np.asarray(magnitude, dtype=np.float32)
    if values.size < 4:
        return values
    taps = min(max(5, taps), values.size if values.size % 2 == 1 else values.size - 1)
    kernel = np.hanning(taps).astype(np.float32)
    kernel /= float(np.sum(kernel)) + 1e-8
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _median_f0(audio: np.ndarray, sample_rate: int) -> float:
    contour = extract_pitch_contour(audio, sample_rate)
    voiced = contour[contour > 0]
    return float(np.median(voiced)) if voiced.size else 0.0


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 32 or sample_rate <= 0:
        return 0.0
    mag = np.abs(np.fft.rfft(x)).astype(np.float32)
    if float(np.sum(mag)) <= 1e-8:
        return 0.0
    freqs = np.linspace(0.0, sample_rate / 2.0, mag.size, dtype=np.float32)
    return float(np.sum(freqs * mag) / (np.sum(mag) + 1e-8))


def _match_reference_timbre(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    del sample_rate
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 32 or not references:
        return x

    source_spec = np.fft.rfft(x)
    source_env = _smooth_spectrum(np.log(np.abs(source_spec) + 1e-6))
    ref_envs: list[np.ndarray] = []
    for ref in references:
        ref_x = stretch_to_length(np.asarray(ref, dtype=np.float32), x.size)
        ref_envs.append(_smooth_spectrum(np.log(np.abs(np.fft.rfft(ref_x)) + 1e-6)))
    if not ref_envs:
        return x

    ref_env = np.mean(np.stack(ref_envs, axis=0), axis=0).astype(np.float32)
    ratio = np.exp(np.clip(ref_env - source_env, -0.32, 0.32)).astype(np.float32)
    ratio = np.clip(0.35 * ratio + 0.65, 0.72, 1.38)
    matched = np.fft.irfft(source_spec * ratio, n=x.size).astype(np.float32)
    return (0.82 * x + 0.18 * matched).astype(np.float32)


def _match_reference_pitch(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if not references:
        return x
    source_hz = _median_f0(x, sample_rate)
    ref_hz_values = [_median_f0(ref, sample_rate) for ref in references]
    ref_hz_values = [value for value in ref_hz_values if value > 0]
    if source_hz <= 0 or not ref_hz_values:
        return x
    ref_hz = float(np.median(np.asarray(ref_hz_values, dtype=np.float32)))
    semitones = float(np.clip(12.0 * np.log2(ref_hz / max(source_hz, 1.0)) * 0.45, -2.0, 2.0))
    return pitch_shift_audio(x, sample_rate, semitones)


def _match_reference_brightness(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 32 or not references:
        return x
    source_centroid = _spectral_centroid(x, sample_rate)
    ref_centroids = [_spectral_centroid(np.asarray(ref, dtype=np.float32), sample_rate) for ref in references]
    ref_centroids = [value for value in ref_centroids if value > 0]
    if source_centroid <= 0 or not ref_centroids:
        return x

    ref_centroid = float(np.median(np.asarray(ref_centroids, dtype=np.float32)))
    brightness = float(np.clip((ref_centroid / max(source_centroid, 1.0) - 1.0) * 0.34, -0.18, 0.18))
    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    curve = np.clip(1.0 + brightness * np.clip((freqs - 0.18) / 0.75, 0.0, 1.0), 0.82, 1.22)
    shaped = np.fft.irfft(spec * curve, n=x.size).astype(np.float32)
    return (0.88 * x + 0.12 * shaped).astype(np.float32)


def _match_reference_level(audio: np.ndarray, references: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    ref_rms_values = [float(np.sqrt(np.mean(np.asarray(ref, dtype=np.float32) ** 2))) for ref in references if np.asarray(ref).size]
    source_rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if source_rms <= 1e-8 or not ref_rms_values:
        return x
    ref_rms = float(np.median(np.asarray(ref_rms_values, dtype=np.float32)))
    gain = float(np.clip((ref_rms / source_rms) ** 0.25, 0.88, 1.12))
    return (x * gain).astype(np.float32)


def speaker_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = extract_mfcc(audio, sample_rate, n_mfcc=20)
    stats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)], axis=0).astype(np.float32)
    norm = np.linalg.norm(stats) + 1e-8
    return (stats / norm).astype(np.float32)


def clone_speaker(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if not references:
        return source

    pitch_matched = _match_reference_pitch(source, sample_rate, references)
    timbre_matched = _match_reference_timbre(pitch_matched, sample_rate, references)
    bright_matched = _match_reference_brightness(timbre_matched, sample_rate, references)
    level_matched = _match_reference_level(bright_matched, references)
    return stretch_to_length(np.clip(level_matched, -1.0, 1.0), source.size)
