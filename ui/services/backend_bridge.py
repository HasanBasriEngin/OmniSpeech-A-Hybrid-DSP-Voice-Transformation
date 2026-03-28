from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.evaluation import compute_audio_quality, compute_intelligibility, measure_latency
from core.input_module import load_audio
from core.output_module import export_audio, save_session_log
from core.preprocessing import detect_f0
from modules.emotion_conversion import convert_emotion
from modules.gender_age_conversion import convert_gender_age
from modules.singing_voice import speech_to_singing
from modules.speaker_conversion import convert_speaker, export_embedding, extract_speaker_embedding
from ui.controls.parameter_panel import ParameterSnapshot
from ui.models.app_state import AppState
from utils.logger import SessionLogger


class BackendBridge:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.logger = SessionLogger()

    def load_input(self, file_path: str) -> None:
        audio = load_audio(file_path, target_sr=self.state.sample_rate)
        if len(audio) < int(self.state.sample_rate * 0.5):
            raise ValueError("Audio is too short. Minimum duration is 0.5 seconds.")
        self.state.input_path = file_path
        self.state.input_audio = audio
        self.state.processed_audio = np.array([], dtype=np.float32)
        self.logger.add("input_loaded", path=file_path, samples=int(len(audio)))

    def process(self, params: ParameterSnapshot) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.state.input_audio.size == 0:
            raise ValueError("No audio loaded.")
        x = self.state.input_audio
        sr = self.state.sample_rate

        if params.mode == "emotion":
            y = convert_emotion(x, sr, params.option)
            y = np.clip(y * params.intensity, -1.0, 1.0)
        elif params.mode == "gender_age":
            y = convert_gender_age(x, sr, params.option)
            y = np.clip(y * params.intensity, -1.0, 1.0)
        elif params.mode == "speaker":
            target = extract_speaker_embedding(x[::-1].copy(), sr)
            y = convert_speaker(x, sr, target_embedding=target)
            y = (y * (0.7 + 0.3 * params.intensity)).astype(np.float32)
        elif params.mode == "singing":
            contour = np.linspace(140, 280, 64).astype(np.float32)
            y = speech_to_singing(x, sr, contour, input_type="pitch_contour")
            y = np.clip(y * params.intensity, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported mode: {params.mode}")

        f0 = detect_f0(y, sr)
        quality = compute_audio_quality(x, y, sr)
        intelligibility = compute_intelligibility(x, y)
        latency = measure_latency(lambda: y[:1000])
        metrics = {
            "latency_ms": latency,
            "intelligibility_mos": intelligibility,
            "quality_pesq_like": quality,
        }
        self.state.processed_audio = y
        self.state.target_mode = params.mode
        self.state.processing_ms = latency
        self.logger.add("processed", params=asdict(params), metrics=metrics)
        self.state.session_log = self.logger.entries
        return y, f0, metrics

    def export_audio(self, output_path: str, fmt: str) -> Path:
        if self.state.processed_audio.size == 0:
            raise ValueError("Nothing to export. Process audio first.")
        export_audio(self.state.processed_audio, self.state.sample_rate, output_path, fmt)
        out = Path(output_path).with_suffix(f".{fmt}")
        self.logger.add("audio_exported", path=str(out), fmt=fmt)
        self.state.session_log = self.logger.entries
        return out

    def export_embedding(self, output_path: str) -> Path:
        if self.state.processed_audio.size == 0:
            raise ValueError("No processed audio for embedding export.")
        emb = extract_speaker_embedding(self.state.processed_audio, self.state.sample_rate)
        export_embedding(emb, output_path)
        out = Path(output_path).with_suffix(".npy")
        self.logger.add("embedding_exported", path=str(out))
        self.state.session_log = self.logger.entries
        return out

    def save_log(self, output_path: str) -> None:
        save_session_log(self.state.session_log, output_path)

