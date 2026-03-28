from __future__ import annotations

import time

import numpy as np


def measure_latency(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000.0


def measure_processing_time(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def compute_intelligibility(original, processed) -> float:
    original = np.asarray(original, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)
    if original.size == 0 or processed.size == 0:
        return 0.0
    min_len = min(len(original), len(processed))
    corr = np.corrcoef(original[:min_len], processed[:min_len])[0, 1]
    corr = float(np.nan_to_num(corr, nan=0.0))
    return float(np.clip(2.5 + corr * 2.5, 0.0, 5.0))


def compute_audio_quality(original, processed, sr) -> float:
    _ = sr
    original = np.asarray(original, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)
    if original.size == 0 or processed.size == 0:
        return 0.0
    min_len = min(len(original), len(processed))
    noise = original[:min_len] - processed[:min_len]
    snr = 10 * np.log10(
        np.mean(original[:min_len] ** 2) / (np.mean(noise**2) + 1e-8) + 1e-8
    )
    return float(np.clip((snr / 20.0) * 3.0 + 2.0, 1.0, 4.5))


def compute_speaker_similarity(emb1, emb2) -> float:
    e1 = np.asarray(emb1, dtype=np.float32)
    e2 = np.asarray(emb2, dtype=np.float32)
    denom = (np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-8
    return float(np.dot(e1, e2) / denom)

