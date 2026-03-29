from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from backend.audio.features import extract_pitch_contour
from backend.audio.io import default_output_path, load_audio_mono, save_audio
from backend.modules.gender_age import convert_gender_age
from backend.modules.singing import convert_to_singing
from backend.modules.speaker_clone import clone_speaker


@dataclass
class PipelineResult:
    output_path: str
    metrics: dict[str, float]


class VoiceConversionPipeline:
    """Coordinates file-based conversion tasks across modular DSP/ML services."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def convert_gender_age_file(self, input_path: str, mode: str, output_path: str | None = None) -> PipelineResult:
        source = load_audio_mono(input_path, self.sample_rate)
        start = perf_counter()
        converted = convert_gender_age(source, self.sample_rate, mode)
        elapsed = perf_counter() - start
        path = save_audio(output_path or default_output_path(input_path, mode), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
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
        elapsed = perf_counter() - start

        path = save_audio(output_path or default_output_path(input_path, "singing"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        return PipelineResult(output_path=path, metrics=metrics)

    def process_live_chunk(self, chunk: np.ndarray, task: str, options: dict[str, object]) -> np.ndarray:
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

        raise ValueError(f"Unsupported live task: {task}")

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
