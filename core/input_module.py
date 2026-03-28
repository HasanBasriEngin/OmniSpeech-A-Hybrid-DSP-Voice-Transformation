from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
from scipy.signal import resample_poly

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac"}


def validate_format(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-8:
        return audio
    return audio / peak


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    if not validate_format(file_path):
        raise ValueError("Unsupported audio format. Supported: WAV, MP3, FLAC")

    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required to load audio files.") from exc

    audio, sr = sf.read(file_path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
    return normalize_amplitude(audio)


def detect_silence(audio: np.ndarray, threshold_db: float = -40) -> list[tuple]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []
    frame = 1024
    hop = 512
    threshold = 10 ** (threshold_db / 20.0)
    segments: list[tuple[int, int]] = []
    start = None
    for i in range(0, len(audio) - frame, hop):
        chunk = audio[i : i + frame]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms > threshold and start is None:
            start = i
        if rms <= threshold and start is not None:
            segments.append((start, i + frame))
            start = None
    if start is not None:
        segments.append((start, len(audio)))
    return segments


def segment_audio(audio: np.ndarray, sr: int, max_duration: float = 5.0) -> list[np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    max_samples = int(sr * max_duration)
    if max_samples <= 0:
        raise ValueError("max_duration must be positive.")
    return [audio[i : i + max_samples] for i in range(0, len(audio), max_samples)]


def start_mic_stream(sr: int = 16000, chunk: int = 1024) -> Generator:
    try:
        import pyaudio
    except Exception as exc:
        raise RuntimeError("PyAudio is required for microphone streaming.") from exc

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=chunk,
    )

    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            yield np.frombuffer(data, dtype=np.float32)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

