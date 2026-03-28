from __future__ import annotations

import numpy as np
from scipy.signal import resample


def pitch_shift_simple(audio: np.ndarray, semitones: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if np.isclose(semitones, 0.0):
        return audio.copy()
    ratio = 2 ** (semitones / 12.0)
    new_len = max(1, int(len(audio) / ratio))
    shifted = resample(audio, new_len)
    return resample(shifted, len(audio)).astype(np.float32)


def time_stretch_simple(audio: np.ndarray, rate: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    rate = max(0.5, min(2.0, float(rate)))
    new_len = max(1, int(len(audio) / rate))
    return resample(audio, new_len).astype(np.float32)


def match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))

