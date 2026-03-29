from __future__ import annotations

import librosa
import numpy as np


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 20) -> np.ndarray:
    return librosa.feature.mfcc(y=np.asarray(audio, dtype=np.float32), sr=sample_rate, n_mfcc=n_mfcc)


def extract_pitch_contour(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    f0, _, _ = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    return np.nan_to_num(f0, nan=0.0).astype(np.float32)


def stretch_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == target_length:
        return x
    if target_length <= 1:
        return np.zeros(1, dtype=np.float32)
    index = np.linspace(0, max(0, x.size - 1), target_length, dtype=np.float32)
    return np.interp(index, np.arange(x.size, dtype=np.float32), x).astype(np.float32)
