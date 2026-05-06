from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for band in range(n_mels):
        left = bins[band]
        center = max(left + 1, bins[band + 1])
        right = max(center + 1, bins[band + 2])

        for idx in range(left, min(center, bank.shape[1])):
            bank[band, idx] = (idx - left) / max(center - left, 1)
        for idx in range(center, min(right, bank.shape[1])):
            bank[band, idx] = (right - idx) / max(right - center, 1)

    return bank


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 20) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)

    n_fft = 1024
    hop_length = 256
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(x, (0, max(0, n_fft - x.size)))

    frames: list[np.ndarray] = []
    for start in range(0, max(1, padded.size - n_fft + 1), hop_length):
        frame = padded[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        frames.append(frame * window)

    if not frames:
        frames = [np.pad(x, (0, max(0, n_fft - x.size)))[:n_fft] * window]

    stft = np.fft.rfft(np.stack(frames, axis=0), n=n_fft, axis=1)
    power = (np.abs(stft) ** 2).astype(np.float32)
    mel_bank = _mel_filter_bank(sample_rate, n_fft, 40)
    mel_spec = np.maximum(power @ mel_bank.T, 1e-8)
    log_mel = np.log(mel_spec)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs.T.astype(np.float32)


def stretch_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == target_length:
        return x
    if target_length <= 1:
        return np.zeros(1, dtype=np.float32)
    if x.size <= 1:
        return np.full(target_length, float(x[0]) if x.size else 0.0, dtype=np.float32)

    index = np.linspace(0, max(0, x.size - 1), target_length, dtype=np.float32)
    return np.interp(index, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


def time_stretch_audio(audio: np.ndarray, rate: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    safe_rate = float(np.clip(rate, 0.25, 4.0))
    target_length = max(1, int(round(x.size / safe_rate)))
    return stretch_to_length(x, target_length)


def pitch_shift_audio(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    del sample_rate  # The lightweight implementation only needs the shift ratio.

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    if abs(semitones) < 1e-4:
        return x

    ratio = float(2.0 ** (semitones / 12.0))
    shifted_length = max(1, int(round(x.size / max(ratio, 1e-4))))
    shifted = stretch_to_length(x, shifted_length)
    return stretch_to_length(shifted, x.size)


def extract_pitch_contour(audio: np.ndarray, sample_rate: int, frame_length: int | None = None) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 64 or sample_rate <= 0:
        return np.zeros(0 if x.size == 0 else 1, dtype=np.float32)

    frame_size = int(frame_length) if frame_length is not None else sample_rate // 20
    frame_size = min(max(512, frame_size), x.size)
    hop_length = max(128, frame_size // 4)
    fmin = 65.4
    fmax = 1046.5
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_size - 1, int(sample_rate / fmin))
    if min_lag >= max_lag:
        return np.zeros(1, dtype=np.float32)

    window = np.hanning(frame_size).astype(np.float32)
    contour: list[float] = []
    for start in range(0, max(1, x.size - frame_size + 1), hop_length):
        frame = x[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))

        frame = frame * window
        energy = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
        if energy < 1e-4:
            contour.append(0.0)
            continue

        autocorr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        search = autocorr[min_lag : max_lag + 1]
        if search.size == 0:
            contour.append(0.0)
            continue

        peak_offset = int(np.argmax(search))
        peak = float(search[peak_offset])
        if peak <= 0.1 * float(autocorr[0] + 1e-8):
            contour.append(0.0)
            continue

        lag = min_lag + peak_offset
        contour.append(float(sample_rate / lag))

    if not contour:
        return np.zeros(1, dtype=np.float32)
    return np.asarray(contour, dtype=np.float32)
