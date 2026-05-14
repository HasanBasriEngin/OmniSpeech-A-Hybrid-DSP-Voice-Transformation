from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from backend.audio.filtering import post_filter_voice
from backend.audio.features import extract_pitch_contour
from backend.audio.io import default_output_path, load_audio_mono, save_audio
from backend.audio.spectrogram_image import SpectrogramImageResult, preprocess_spectrogram_for_model
from backend.config import SETTINGS
from backend.modules.emotion import convert_emotion
from backend.modules.gender_age import convert_gender_age
from backend.modules.celebrity_voice import convert_celebrity
from backend.modules.rvc_adapter import (
    convert_file_with_rvc,
    convert_gender_age_with_rvc,
    get_gender_age_rvc_config,
    get_rvc_config,
)
from backend.modules.singing import convert_to_singing
from backend.modules.speaker_clone import clone_speaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineResult:
    output_path: str
    metrics: dict[str, float]


class VoiceConversionPipeline:
    """Coordinates file-based conversion tasks across modular DSP/ML services."""

    def __init__(
        self,
        sample_rate: int,
        *,
        rvc_models_dir: str | None = None,
        rvc_device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.rvc_models_dir = rvc_models_dir or SETTINGS.rvc_models_dir
        self.rvc_device = rvc_device or SETTINGS.rvc_device

    def convert_emotion_file(
        self,
        input_path: str,
        emotion: str,
        pitch_override: float | None = None,
        rate_override: float | None = None,
        energy_override: float | None = None,
        output_path: str | None = None,
    ) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)
        start = perf_counter()
        converted = convert_emotion(
            source,
            self.sample_rate,
            emotion,
            pitch_override=pitch_override,
            rate_override=rate_override,
            energy_override=energy_override,
        )
        converted = post_filter_voice(converted, self.sample_rate)
        elapsed = perf_counter() - start
        path = save_audio(output_path or default_output_path(input_path, f"emotion_{emotion}"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics["emotion_profile"] = float(sorted(["angry", "calm", "excited", "sad", "whisper"]).index(emotion))
        if pitch_override is not None:
            metrics["pitch_override"] = float(pitch_override)
        if rate_override is not None:
            metrics["rate_override"] = float(rate_override)
        if energy_override is not None:
            metrics["energy_override"] = float(energy_override)
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_gender_age_file(self, input_path: str, mode: str, output_path: str | None = None) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)
        start = perf_counter()
        prepared = preprocess_spectrogram_for_model(source, self.sample_rate)
        temp_dir: Path | None = None
        try:
            rvc_input_path = input_path
            if (
                get_gender_age_rvc_config(mode, models_dir=self.rvc_models_dir) is not None
                and prepared.metrics.get("opencv_spectrogram_applied", 0.0) > 0.0
            ):
                rvc_input_path, temp_dir = self._write_ai_preprocessed_input(prepared, self.sample_rate, "gender_age")

            rvc_result = convert_gender_age_with_rvc(
                rvc_input_path,
                mode,
                self.sample_rate,
                models_dir=self.rvc_models_dir,
                device=self.rvc_device,
            )
            if rvc_result is None:
                converted = convert_gender_age(prepared.audio, self.sample_rate, mode)
                rvc_engine = 0.0
            else:
                converted = rvc_result.audio
                rvc_engine = 1.0
            converted = post_filter_voice(converted, self.sample_rate)
            elapsed = perf_counter() - start
        finally:
            if temp_dir is not None:
                self._cleanup_temp_dir(temp_dir)
        path = save_audio(output_path or default_output_path(input_path, mode), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(prepared.metrics)
        metrics["rvc_engine"] = rvc_engine
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_speaker_clone_file(
        self,
        input_path: str,
        reference_paths: list[str],
        output_path: str | None = None,
    ) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)
        references = [load_audio_mono(path, self.sample_rate) for path in reference_paths]

        start = perf_counter()
        converted = clone_speaker(source, self.sample_rate, references)
        converted = post_filter_voice(converted, self.sample_rate)
        elapsed = perf_counter() - start

        path = save_audio(output_path or default_output_path(input_path, "speaker_clone"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics["reference_count"] = float(len(reference_paths))
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_singing_file(
        self,
        input_path: str,
        midi_path: str | None,
        pitch_contour: list[float] | None,
        output_path: str | None = None,
    ) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)

        start = perf_counter()
        converted = convert_to_singing(
            audio=source,
            sample_rate=self.sample_rate,
            midi_path=midi_path,
            pitch_contour=pitch_contour,
        )
        converted = post_filter_voice(converted, self.sample_rate)
        elapsed = perf_counter() - start

        path = save_audio(output_path or default_output_path(input_path, "singing"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_celebrity_file(self, input_path: str, celebrity: str, output_path: str | None = None) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)
        start = perf_counter()
        prepared = preprocess_spectrogram_for_model(source, self.sample_rate)
        temp_dir: Path | None = None
        try:
            rvc_input_path = input_path
            if (
                get_rvc_config("celebrity", celebrity, models_dir=self.rvc_models_dir) is not None
                and prepared.metrics.get("opencv_spectrogram_applied", 0.0) > 0.0
            ):
                rvc_input_path, temp_dir = self._write_ai_preprocessed_input(prepared, self.sample_rate, "celebrity")

            rvc_result = convert_file_with_rvc(
                rvc_input_path,
                "celebrity",
                celebrity,
                self.sample_rate,
                models_dir=self.rvc_models_dir,
                device=self.rvc_device,
            )
            if rvc_result is None:
                converted = convert_celebrity(prepared.audio, self.sample_rate, celebrity)
                rvc_engine = 0.0
            else:
                converted = rvc_result.audio
                rvc_engine = 1.0
            converted = post_filter_voice(converted, self.sample_rate)
            elapsed = perf_counter() - start
        finally:
            if temp_dir is not None:
                self._cleanup_temp_dir(temp_dir)
        path = save_audio(output_path or default_output_path(input_path, f"celebrity_{celebrity}"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(prepared.metrics)
        metrics["celebrity_profile"] = float(
            sorted(["adele", "james_earl_jones", "michael_jackson", "morgan_freeman", "taylor_swift"]).index(celebrity)
        )
        metrics["rvc_engine"] = rvc_engine
        return PipelineResult(output_path=path, metrics=metrics)

    def process_live_chunk(self, chunk: np.ndarray, task: str, options: dict[str, object]) -> np.ndarray:
        if task == "emotion":
            emotion = str(options.get("emotion", "calm"))
            return convert_emotion(
                chunk,
                self.sample_rate,
                emotion,
                pitch_override=self._optional_float(options, "pitch_override"),
                rate_override=self._optional_float(options, "rate_override"),
                energy_override=self._optional_float(options, "energy_override"),
            )

        if task == "gender_age":
            mode = str(options.get("mode", "male_to_female"))
            return convert_gender_age(chunk, self.sample_rate, mode)

        if task == "speaker_clone":
            # Live mode uses light identity preservation when no references are available.
            return clone_speaker(chunk, self.sample_rate, references=[])

        if task == "singing":
            return convert_to_singing(
                chunk,
                self.sample_rate,
                midi_path=options.get("midi_path") if isinstance(options.get("midi_path"), str) else None,
                pitch_contour=options.get("pitch_contour") if isinstance(options.get("pitch_contour"), list) else None,
            )

        if task == "celebrity":
            celebrity = str(options.get("celebrity", "michael_jackson"))
            return convert_celebrity(chunk, self.sample_rate, celebrity)

        raise ValueError(f"Unsupported live task: {task}")

    @staticmethod
    def _optional_float(options: dict[str, object], key: str) -> float | None:
        value = options.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _build_metrics(self, source: np.ndarray, converted: np.ndarray, elapsed_seconds: float) -> dict[str, float]:
        source_f0 = extract_pitch_contour(source, self.sample_rate)
        converted_f0 = extract_pitch_contour(converted, self.sample_rate)

        source_voiced = source_f0[source_f0 > 0]
        converted_voiced = converted_f0[converted_f0 > 0]
        source_pitch = float(np.median(source_voiced)) if source_voiced.size else 0.0
        converted_pitch = float(np.median(converted_voiced)) if converted_voiced.size else 0.0

        signal = np.asarray(source, dtype=np.float32)
        estimate = np.asarray(converted, dtype=np.float32)
        min_len = min(signal.size, estimate.size)
        signal = signal[:min_len]
        estimate = estimate[:min_len]

        noise = signal - estimate
        signal_power = float(np.mean(signal**2) + 1e-8)
        noise_power = float(np.mean(noise**2) + 1e-8)
        snr = 10.0 * np.log10(signal_power / noise_power)

        return {
            "processing_seconds": float(round(elapsed_seconds, 4)),
            "input_duration_seconds": float(round(source.size / self.sample_rate, 3)),
            "output_duration_seconds": float(round(converted.size / self.sample_rate, 3)),
            "input_median_f0": float(round(source_pitch, 3)),
            "output_median_f0": float(round(converted_pitch, 3)),
            "snr_estimate_db": float(round(snr, 3)),
        }

    @staticmethod
    def _resolve_ai_preprocess_temp_root() -> Path:
        raw_dir = os.getenv("OMNISPEECH_AI_PREPROCESS_TEMP_DIR", ".tmp/ai_preprocess")
        root = Path(raw_dir).expanduser()
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    @classmethod
    def _write_ai_preprocessed_input(
        cls,
        prepared: SpectrogramImageResult,
        sample_rate: int,
        prefix: str,
    ) -> tuple[str, Path]:
        temp_root = cls._resolve_ai_preprocess_temp_root()
        run_dir = temp_root / f"omnispeech-{prefix}-{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        path = run_dir / "ai_input.wav"
        save_audio(str(path), prepared.audio, sample_rate)
        return str(path), run_dir

    @staticmethod
    def _cleanup_temp_dir(run_dir: Path) -> None:
        try:
            for child in run_dir.iterdir():
                if child.is_file():
                    child.unlink()
            run_dir.rmdir()
        except OSError:
            pass
