from __future__ import annotations

import numpy as np
from scipy.signal import stft


def _safe_librosa():
    try:
        import librosa

        return librosa
    except Exception:
        return None


def compute_energy_envelope(audio: np.ndarray, frame_length: int = 2048) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    hop = max(1, frame_length // 4)
    frames = []
    for i in range(0, len(audio) - frame_length + 1, hop):
        chunk = audio[i : i + frame_length]
        frames.append(np.sqrt(np.mean(chunk**2) + 1e-10))
    return np.asarray(frames, dtype=np.float32)


def extract_features(audio: np.ndarray, sr: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    librosa = _safe_librosa()
    if librosa is not None:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        stft_mag = np.abs(librosa.stft(audio))
        return {
            "mfcc": mfcc.astype(np.float32),
            "mel_spectrogram": mel.astype(np.float32),
            "stft": stft_mag.astype(np.float32),
            "chroma": chroma.astype(np.float32),
        }

    _, _, zxx = stft(audio, fs=sr, nperseg=1024, noverlap=768)
    mag = np.abs(zxx).astype(np.float32)
    return {
        "mfcc": np.log1p(mag[:20, :]),
        "mel_spectrogram": np.log1p(mag[:128, :]),
        "stft": mag,
        "chroma": mag[:12, :],
    }


def detect_f0(audio: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    librosa = _safe_librosa()
    if librosa is not None:
        f0, _, _ = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax)
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        return f0

    frame = 1024
    hop = 256
    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))
    contour = []
    for i in range(0, len(audio) - frame, hop):
        chunk = audio[i : i + frame]
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1 :]
        corr[:min_lag] = 0
        lag = np.argmax(corr[min_lag:max_lag]) + min_lag
        contour.append(float(sr / lag) if lag > 0 else 0.0)
    return np.asarray(contour, dtype=np.float32)


def separate_prosodic_spectral(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    _ = sr
    energy = compute_energy_envelope(audio)
    if energy.size:
        win = np.hanning(min(11, energy.size))
        win = win / np.sum(win)
        prosodic = np.convolve(energy, win, mode="same").astype(np.float32)
    else:
        prosodic = energy
    spectral = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float32))).astype(np.float32)
    return prosodic, spectral

