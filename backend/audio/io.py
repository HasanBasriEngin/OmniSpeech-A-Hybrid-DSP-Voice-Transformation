from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_path(path: str) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Audio file not found: {target}")
    if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {target.suffix}")
    return target


def load_audio_mono(path: str, sample_rate: int) -> np.ndarray:
    target = validate_audio_path(path)
    audio, _ = librosa.load(str(target), sr=sample_rate, mono=True)
    return normalize_audio(audio)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 1e-7:
        return arr
    return (arr / peak).astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> str:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), np.asarray(audio, dtype=np.float32), sample_rate)
    return str(output)


def default_output_path(input_path: str, suffix: str) -> str:
    source = Path(input_path).expanduser().resolve()
    return str(source.with_stem(f"{source.stem}_{suffix}").with_suffix(".wav"))
