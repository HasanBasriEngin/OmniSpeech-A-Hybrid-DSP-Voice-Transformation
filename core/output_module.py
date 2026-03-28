from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def synthesize_waveform(processed_audio: np.ndarray, sr: int) -> np.ndarray:
    _ = sr
    audio = np.asarray(processed_audio, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)


def export_audio(audio: np.ndarray, sr: int, path: str, fmt: str = "wav") -> None:
    fmt = fmt.lower()
    if fmt not in {"wav", "mp3", "flac"}:
        raise ValueError("Unsupported export format.")

    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required for export.") from exc

    audio = synthesize_waveform(audio, sr)
    output = Path(path)
    if output.suffix.lower() != f".{fmt}":
        output = output.with_suffix(f".{fmt}")

    if fmt in {"wav", "flac"}:
        sf.write(str(output), audio, sr)
        return

    tmp_wav = output.with_suffix(".wav")
    sf.write(str(tmp_wav), audio, sr)
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub is required for MP3 export.") from exc
    seg = AudioSegment.from_wav(str(tmp_wav))
    seg.export(str(output), format="mp3")
    tmp_wav.unlink(missing_ok=True)


def playback_realtime(audio: np.ndarray, sr: int) -> None:
    try:
        import pyaudio
    except Exception as exc:
        raise RuntimeError("PyAudio is required for playback.") from exc

    audio = synthesize_waveform(audio, sr).astype(np.float32)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    try:
        stream.write(audio.tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def save_session_log(log_entries: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps(log_entries, indent=2), encoding="utf-8")

