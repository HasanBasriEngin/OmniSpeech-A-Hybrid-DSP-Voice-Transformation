from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from backend.audio.filtering import post_filter_voice
from backend.audio.dsp_metrics import measure_audio_health
from backend.audio.dsp_profiles import get_dsp_profile_settings, update_dsp_profile
from backend.audio.dsp_quality import optimize_post_filter_quality
from backend.audio.features import extract_pitch_contour
from backend.audio.io import default_output_path, load_audio_mono, save_audio
from backend.audio.pure_dsp import merge_preserving_noise_regions, prepare_clean_speech
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
from backend.modules.freevc_adapter import convert_file_with_freevc
from backend.modules.freevc_profiles import get_freevc_reference_profile
from backend.modules.singing import convert_to_singing
from backend.modules.speaker_clone import clone_speaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


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
        freevc_assets_dir: str | None = None,
        freevc_device: str | None = None,
        freevc_profiles_dir: str | None = None,
        dsp_profiles_dir: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.rvc_models_dir = rvc_models_dir or SETTINGS.rvc_models_dir
        self.rvc_device = rvc_device or SETTINGS.rvc_device
        self.freevc_assets_dir = freevc_assets_dir or SETTINGS.freevc_assets_dir
        self.freevc_device = freevc_device or SETTINGS.freevc_device
        self.freevc_profiles_dir = freevc_profiles_dir or SETTINGS.freevc_profiles_dir
        self.dsp_profiles_dir = dsp_profiles_dir or SETTINGS.dsp_profiles_dir

    def convert_emotion_file(
        self,
        input_path: str,
        emotion: str,
        pitch_override: float | None = None,
        rate_override: float | None = None,
        energy_override: float | None = None,
        output_path: str | None = None,
        *,
        use_ai_engines: bool = True,
    ) -> PipelineResult:
        profile_name = self._dsp_profile_name("emotion", emotion)
        source = load_audio_mono(input_path, self.sample_rate)
        clean = prepare_clean_speech(source, self.sample_rate)
        start = perf_counter()
        converted = convert_emotion(
            clean.audio,
            self.sample_rate,
            emotion,
            pitch_override=pitch_override,
            rate_override=rate_override,
            energy_override=energy_override,
        )
        settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
        converted = merge_preserving_noise_regions(
            clean.audio,
            converted,
            self.sample_rate,
            intensity=settings.transform_intensity,
            smoothing=settings.formant_smoothing,
        )
        converted, dsp_metrics = self._post_filter_with_autotune(converted, profile_name, engine="dsp")
        elapsed = perf_counter() - start
        path = save_audio(output_path or default_output_path(input_path, f"emotion_{emotion}"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(clean.metrics)
        metrics.update(dsp_metrics)
        metrics["ai_engines_enabled"] = 1.0 if use_ai_engines else 0.0
        metrics["emotion_profile"] = float(sorted(["angry", "calm", "excited", "sad", "whisper"]).index(emotion))
        if pitch_override is not None:
            metrics["pitch_override"] = float(pitch_override)
        if rate_override is not None:
            metrics["rate_override"] = float(rate_override)
        if energy_override is not None:
            metrics["energy_override"] = float(energy_override)
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_gender_age_file(
        self,
        input_path: str,
        mode: str,
        output_path: str | None = None,
        *,
        use_ai_engines: bool = True,
    ) -> PipelineResult:
        profile_name = self._dsp_profile_name("gender_age", mode)
        source = load_audio_mono(input_path, self.sample_rate)
        clean = prepare_clean_speech(source, self.sample_rate)
        start = perf_counter()
        prepared = preprocess_spectrogram_for_model(clean.audio, self.sample_rate)
        temp_dir: Path | None = None
        try:
            rvc_input_path = input_path
            if (
                use_ai_engines
                and get_gender_age_rvc_config(mode, models_dir=self.rvc_models_dir) is not None
                and prepared.metrics.get("opencv_spectrogram_applied", 0.0) > 0.0
            ):
                rvc_input_path, temp_dir = self._write_ai_preprocessed_input(prepared, self.sample_rate, "gender_age")

            rvc_result = None
            if use_ai_engines:
                rvc_result = convert_gender_age_with_rvc(
                    rvc_input_path,
                    mode,
                    self.sample_rate,
                    models_dir=self.rvc_models_dir,
                    device=self.rvc_device,
                )
            freevc_gender_age_refine = 0.0
            if rvc_result is None:
                rvc_engine = 0.0
                freevc_engine = 0.0
                freevc_result = None
                freevc_profile = None
                if use_ai_engines:
                    freevc_profile = get_freevc_reference_profile(
                        "gender_age",
                        mode,
                        profiles_dir=self.freevc_profiles_dir,
                    )
                if use_ai_engines and freevc_profile is not None:
                    freevc_result = convert_file_with_freevc(
                        input_path,
                        str(freevc_profile.reference_path),
                        self.sample_rate,
                        assets_dir=self.freevc_assets_dir,
                        device=self.freevc_device,
                    )

                if freevc_result is None:
                    converted = convert_gender_age(prepared.audio, self.sample_rate, mode)
                    settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
                    converted = merge_preserving_noise_regions(
                        prepared.audio,
                        converted,
                        self.sample_rate,
                        intensity=settings.transform_intensity,
                        smoothing=settings.formant_smoothing,
                    )
                else:
                    converted = freevc_result.audio
                    freevc_engine = 1.0
                    if self._freevc_gender_age_refine_enabled():
                        converted = convert_gender_age(converted, self.sample_rate, mode)
                        freevc_gender_age_refine = 1.0
            else:
                converted = rvc_result.audio
                rvc_engine = 1.0
                freevc_engine = 0.0
            engine = "rvc" if rvc_engine >= 1.0 else "freevc" if freevc_engine >= 1.0 else "dsp"
            converted, dsp_metrics = self._post_filter_with_autotune(converted, profile_name, engine=engine)
            elapsed = perf_counter() - start
        finally:
            if temp_dir is not None:
                self._cleanup_temp_dir(temp_dir)
        path = save_audio(output_path or default_output_path(input_path, mode), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(clean.metrics)
        metrics.update(dsp_metrics)
        metrics.update(prepared.metrics)
        metrics["ai_engines_enabled"] = 1.0 if use_ai_engines else 0.0
        metrics["rvc_engine"] = rvc_engine
        metrics["freevc_engine"] = freevc_engine
        metrics["freevc_gender_age_refine"] = freevc_gender_age_refine
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_speaker_clone_file(
        self,
        input_path: str,
        reference_paths: list[str],
        output_path: str | None = None,
        *,
        use_ai_engines: bool = True,
    ) -> PipelineResult:
        profile_name = self._dsp_profile_name("speaker_clone", "reference" if reference_paths else "identity")
        source = load_audio_mono(input_path, self.sample_rate)
        clean = prepare_clean_speech(source, self.sample_rate)

        start = perf_counter()
        freevc_result = None
        if use_ai_engines and reference_paths:
            freevc_result = convert_file_with_freevc(
                input_path,
                reference_paths[0],
                self.sample_rate,
                assets_dir=self.freevc_assets_dir,
                device=self.freevc_device,
            )

        if freevc_result is None:
            references = [prepare_clean_speech(load_audio_mono(path, self.sample_rate), self.sample_rate).audio for path in reference_paths]
            converted = clone_speaker(clean.audio, self.sample_rate, references)
            settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
            converted = merge_preserving_noise_regions(
                clean.audio,
                converted,
                self.sample_rate,
                intensity=settings.transform_intensity,
                smoothing=settings.formant_smoothing,
            )
            freevc_engine = 0.0
        else:
            converted = freevc_result.audio
            freevc_engine = 1.0
        engine = "freevc" if freevc_engine >= 1.0 else "dsp"
        converted, dsp_metrics = self._post_filter_with_autotune(converted, profile_name, engine=engine)
        elapsed = perf_counter() - start

        path = save_audio(output_path or default_output_path(input_path, "speaker_clone"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(clean.metrics)
        metrics.update(dsp_metrics)
        metrics["reference_count"] = float(len(reference_paths))
        metrics["ai_engines_enabled"] = 1.0 if use_ai_engines else 0.0
        metrics["freevc_engine"] = freevc_engine
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_singing_file(
        self,
        input_path: str,
        midi_path: str | None,
        pitch_contour: list[float] | None,
        output_path: str | None = None,
        *,
        use_ai_engines: bool = True,
    ) -> PipelineResult:
        profile_name = self._dsp_profile_name("singing", "midi" if midi_path else "manual")
        source = load_audio_mono(input_path, self.sample_rate)
        clean = prepare_clean_speech(source, self.sample_rate)

        start = perf_counter()
        converted = convert_to_singing(
            audio=clean.audio,
            sample_rate=self.sample_rate,
            midi_path=midi_path,
            pitch_contour=pitch_contour,
        )
        settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
        converted = merge_preserving_noise_regions(
            clean.audio,
            converted,
            self.sample_rate,
            intensity=settings.transform_intensity,
            smoothing=settings.formant_smoothing,
        )
        converted, dsp_metrics = self._post_filter_with_autotune(converted, profile_name, engine="dsp")
        elapsed = perf_counter() - start

        path = save_audio(output_path or default_output_path(input_path, "singing"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(clean.metrics)
        metrics.update(dsp_metrics)
        metrics["ai_engines_enabled"] = 1.0 if use_ai_engines else 0.0
        return PipelineResult(output_path=path, metrics=metrics)

    def convert_celebrity_file(
        self,
        input_path: str,
        celebrity: str,
        output_path: str | None = None,
        *,
        use_ai_engines: bool = True,
    ) -> PipelineResult:
        profile_name = self._dsp_profile_name("licensed_profile", celebrity)
        source = load_audio_mono(input_path, self.sample_rate)
        clean = prepare_clean_speech(source, self.sample_rate)
        start = perf_counter()
        prepared = preprocess_spectrogram_for_model(clean.audio, self.sample_rate)
        temp_dir: Path | None = None
        try:
            rvc_input_path = input_path
            if (
                use_ai_engines
                and get_rvc_config("celebrity", celebrity, models_dir=self.rvc_models_dir) is not None
                and prepared.metrics.get("opencv_spectrogram_applied", 0.0) > 0.0
            ):
                rvc_input_path, temp_dir = self._write_ai_preprocessed_input(prepared, self.sample_rate, "celebrity")

            rvc_result = None
            if use_ai_engines:
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
                settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
                converted = merge_preserving_noise_regions(
                    prepared.audio,
                    converted,
                    self.sample_rate,
                    intensity=settings.transform_intensity,
                    smoothing=settings.formant_smoothing,
                )
                rvc_engine = 0.0
            else:
                converted = rvc_result.audio
                rvc_engine = 1.0
            engine = "rvc" if rvc_engine >= 1.0 else "dsp"
            converted, dsp_metrics = self._post_filter_with_autotune(converted, profile_name, engine=engine)
            elapsed = perf_counter() - start
        finally:
            if temp_dir is not None:
                self._cleanup_temp_dir(temp_dir)
        path = save_audio(output_path or default_output_path(input_path, f"celebrity_{celebrity}"), converted, self.sample_rate)
        metrics = self._build_metrics(source, converted, elapsed)
        metrics.update(clean.metrics)
        metrics.update(dsp_metrics)
        metrics.update(prepared.metrics)
        metrics["ai_engines_enabled"] = 1.0 if use_ai_engines else 0.0
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

    @staticmethod
    def _dsp_profile_name(category: str, key: str) -> str:
        clean_category = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in category.strip().lower())
        clean_key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key.strip().lower())
        return f"{clean_category}.{clean_key}" if clean_key else clean_category

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

    def _post_filter_with_autotune(
        self,
        audio: np.ndarray,
        profile_name: str,
        *,
        engine: str,
    ) -> tuple[np.ndarray, dict[str, float]]:
        settings = get_dsp_profile_settings(profile_name, profiles_dir=self.dsp_profiles_dir)
        pre_metrics = measure_audio_health(audio, self.sample_rate, prefix="pre_post_")
        neural_safe_filter = engine in {"freevc", "rvc"}
        quality_result = optimize_post_filter_quality(
            audio,
            self.sample_rate,
            settings,
            engine=engine,
            pre_metrics=pre_metrics,
            filter_func=post_filter_voice,
        )
        filtered = quality_result.audio
        post_metrics = quality_result.metrics
        profile_base_settings = settings if neural_safe_filter else quality_result.selected_settings

        try:
            updated = update_dsp_profile(
                profile_name,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                engine=engine,
                profiles_dir=self.dsp_profiles_dir,
                base_settings=profile_base_settings,
            )
            profile_updated = 1.0
        except Exception as exc:  # pragma: no cover - defensive around local profile IO
            logger.warning("DSP AutoTune profile update failed for %s: %s", profile_name, exc)
            updated = settings
            profile_updated = 0.0

        metrics = {
            **pre_metrics,
            **post_metrics,
            "dsp_autotune_applied": 1.0,
            "dsp_profile_updated": profile_updated,
            "dsp_post_gain_db": float(updated.post_gain_db),
            "dsp_ceiling": float(updated.ceiling),
            "dsp_deess_reduction_db": float(updated.deess_reduction_db),
            "dsp_noise_gate_floor": float(updated.noise_gate_floor),
            "dsp_presence_db": float(updated.presence_db),
            "dsp_spectral_tilt_db": float(updated.spectral_tilt_db),
            "dsp_transform_intensity": float(updated.transform_intensity),
            "dsp_formant_smoothing": float(updated.formant_smoothing),
            "dsp_neural_safe_filter": 1.0 if neural_safe_filter else 0.0,
        }
        return filtered, metrics

    @staticmethod
    def _freevc_gender_age_refine_enabled() -> bool:
        return os.getenv("OMNISPEECH_FREEVC_GENDER_AGE_REFINE", "").strip().lower() in {"1", "true", "yes", "on"}

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
