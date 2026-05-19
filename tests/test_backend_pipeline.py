from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from backend.audio.filtering import LiveVoicePostFilter, post_filter_voice
from backend.audio.dsp_profiles import DSPProfileSettings, apply_user_feedback, get_dsp_profile_settings
from backend.audio.dsp_metrics import measure_audio_health
from backend.audio.dsp_quality import optimize_post_filter_quality
from backend.audio.io import normalize_audio
from backend.audio.pure_dsp import merge_preserving_noise_regions, prepare_clean_speech
from backend.audio.spectrogram_quality import analyze_spectrogram_quality
from backend.audio.spectrogram_image import SpectrogramImageResult, preprocess_spectrogram_for_model
from backend.audio.voice_analysis import analyze_pitch_confidence
from backend.modules import emotion as emotion_module
from backend.modules import gender_age as gender_age_module
from backend.modules import rvc_adapter
from backend.modules.freevc_adapter import (
    FreeVCConversionResult,
    FreeVCModelConfig,
    _fit_duration,
    get_freevc_config,
    resolve_wavlm_model,
)
from backend.modules.hf_voice_assets import import_hf_voice_assets, plan_hf_voice_asset_imports
from backend.pipeline import processor as processor_module
from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.live_session import LiveOverlapCrossfader, LiveSessionManager
from backend.tools.import_freevc_assets import import_freevc_assets


def _sine(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _workspace_tmp_dir(prefix: str) -> Path:
    tmp_dir = Path(".tmp") / "tests" / f"{prefix}_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def _write_minimal_freevc_source(root: Path, *, variant: str = "freevc-24") -> None:
    for relative in ("commons.py", "models.py", "modules.py", "utils.py", "mel_processing.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# test fixture\n", encoding="utf-8")

    speaker_dir = root / "speaker_encoder"
    (speaker_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (speaker_dir / "__init__.py").write_text("", encoding="utf-8")
    (speaker_dir / "voice_encoder.py").write_text("# test fixture\n", encoding="utf-8")
    (speaker_dir / "ckpt" / "pretrained_bak_5805000.pt").write_bytes(b"speaker")

    if variant == "freevc-24":
        config_relative = "configs/freevc-24.json"
        checkpoint_relative = "checkpoints/freevc-24.pth"
    elif variant == "freevc":
        config_relative = "configs/freevc.json"
        checkpoint_relative = "checkpoints/freevc.pth"
    else:
        config_relative = "configs/freevc-s.json"
        checkpoint_relative = "checkpoints/freevc-s.pth"

    (root / config_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / config_relative).write_text("{}", encoding="utf-8")
    (root / checkpoint_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / checkpoint_relative).write_bytes(b"checkpoint")


def test_gender_age_file_conversion():
    tmp_dir = _workspace_tmp_dir("gender")
    source = _sine()
    source_path = tmp_dir / "in.wav"
    sf.write(str(source_path), source, 22050)

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(tmp_dir / "models"),
        freevc_profiles_dir=str(tmp_dir / "profiles"),
    )
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.output_path.endswith(".wav")
    assert result.metrics["processing_seconds"] >= 0.0
    assert result.metrics["rvc_engine"] == 0.0
    assert "opencv_spectrogram_applied" in result.metrics


def test_hf_voice_asset_plan_selects_rvc_freevc_and_wavlm():
    tmp_dir = _workspace_tmp_dir("hf_plan")
    plan = plan_hf_voice_asset_imports(["all"], local_root=tmp_dir)
    names = {item.name for item in plan}

    assert {"rvc-core-v2-48k", "freevc-24-one-shot", "wavlm-large-content"} <= names
    assert all(str(tmp_dir.resolve()) in item.local_dir for item in plan)


def test_hf_voice_asset_import_writes_manifest_without_network():
    tmp_dir = _workspace_tmp_dir("hf_import")
    calls: list[tuple[str, str, str]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        repo_id = str(kwargs["repo_id"])
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "download.marker").write_text(repo_id, encoding="utf-8")
        calls.append((repo_id, str(kwargs["repo_type"]), str(kwargs["revision"])))
        return str(local_dir)

    manifest = import_hf_voice_assets(
        ["freevc"],
        local_root=tmp_dir,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert calls == [("OlaWod/FreeVC", "space", "main")]
    manifest_path = Path(str(manifest["manifest_path"]))
    assert manifest_path.exists()
    assert manifest["assets"][0]["name"] == "freevc-24-one-shot"


def test_wavlm_model_resolver_preserves_hugging_face_ids():
    assert resolve_wavlm_model("microsoft/wavlm-large") == "microsoft/wavlm-large"
    assert Path(resolve_wavlm_model("models/hf/wavlm-large")).name == "wavlm-large"


def test_freevc_config_accepts_original_freevc_layout():
    tmp_dir = _workspace_tmp_dir("freevc_original")
    _write_minimal_freevc_source(tmp_dir, variant="freevc")

    config = get_freevc_config(tmp_dir)

    assert config is not None
    assert config.model_id == "freevc-one-shot"
    assert config.config_path.name == "freevc.json"
    assert config.checkpoint_path.name == "freevc.pth"
    assert config.output_sample_rate == 16000


def test_freevc_duration_fit_matches_source_length():
    audio = _sine(duration=0.4)
    fitted = _fit_duration(audio, int(22050 * 0.25))

    assert fitted.dtype == np.float32
    assert fitted.size == int(22050 * 0.25)
    assert np.all(np.isfinite(fitted))


def test_import_freevc_assets_copies_local_original_layout():
    tmp_dir = _workspace_tmp_dir("freevc_import")
    source_dir = tmp_dir / "source"
    target_dir = tmp_dir / "target"
    _write_minimal_freevc_source(source_dir, variant="freevc")

    manifest = import_freevc_assets(
        source_root=source_dir,
        target_dir=target_dir,
        variant="freevc",
    )

    assert Path(str(manifest["manifest_path"])).exists()
    assert (target_dir / "commons.py").exists()
    assert (target_dir / "configs" / "freevc.json").exists()
    assert (target_dir / "checkpoints" / "freevc.pth").exists()
    assert (target_dir / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt").exists()

    config = get_freevc_config(target_dir)
    assert config is not None
    assert config.model_id == "freevc-one-shot"


def test_emotion_file_conversion():
    source = _sine()
    tmp_dir = Path(".tmp") / "tests"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_dir / "emotion_in.wav"
    sf.write(str(source_path), source, 22050)

    pipeline = VoiceConversionPipeline(sample_rate=22050)
    result = pipeline.convert_emotion_file(str(source_path), emotion="calm")

    assert result.output_path.endswith(".wav")
    assert result.metrics["processing_seconds"] >= 0.0


def test_speaker_clone_file_prefers_freevc_when_assets_are_available(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("freevc_speaker")
    source = _sine(duration=0.25)
    reference = _sine(duration=0.25) * 0.5
    source_path = tmp_dir / "input.wav"
    reference_path = tmp_dir / "reference.wav"
    sf.write(str(source_path), source, 22050)
    sf.write(str(reference_path), reference, 22050)

    def fake_convert_file_with_freevc(
        input_path: str,
        reference_path_arg: str,
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCConversionResult:
        assert Path(input_path) == source_path
        assert Path(reference_path_arg) == reference_path
        assert sample_rate == 22050
        assert kwargs["assets_dir"] == "models/hf/freevc-24"
        return FreeVCConversionResult(
            audio=np.asarray(source * 0.25, dtype=np.float32),
            config=FreeVCModelConfig(
                model_id="freevc-24-one-shot",
                assets_dir=tmp_dir,
                checkpoint_path=tmp_dir / "freevc-24.pth",
                config_path=tmp_dir / "freevc-24.json",
                speaker_encoder_path=tmp_dir / "speaker.pt",
                wavlm_model="microsoft/wavlm-large",
            ),
        )

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fake_convert_file_with_freevc)

    pipeline = VoiceConversionPipeline(sample_rate=22050, dsp_profiles_dir=str(tmp_dir / "dsp_profiles"))
    result = pipeline.convert_speaker_clone_file(str(source_path), [str(reference_path)])

    assert result.metrics["freevc_engine"] == 1.0
    assert result.metrics["reference_count"] == 1.0
    assert result.metrics["dsp_autotune_applied"] == 1.0
    assert result.metrics["dsp_neural_safe_filter"] == 1.0


def test_neural_engine_post_filter_uses_conservative_settings(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("freevc_safe_post")
    source = _sine(duration=0.25)
    reference = _sine(duration=0.25) * 0.5
    source_path = tmp_dir / "input.wav"
    reference_path = tmp_dir / "reference.wav"
    sf.write(str(source_path), source, 22050)
    sf.write(str(reference_path), reference, 22050)

    def fake_convert_file_with_freevc(
        input_path: str,
        reference_path_arg: str,
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCConversionResult:
        del input_path, reference_path_arg, sample_rate, kwargs
        return FreeVCConversionResult(
            audio=np.asarray(source * 0.7, dtype=np.float32),
            config=FreeVCModelConfig(
                model_id="freevc-24-one-shot",
                assets_dir=tmp_dir,
                checkpoint_path=tmp_dir / "freevc-24.pth",
                config_path=tmp_dir / "freevc-24.json",
                speaker_encoder_path=tmp_dir / "speaker.pt",
                wavlm_model="models/hf/wavlm-large",
            ),
        )

    captured_settings: list[dict[str, object]] = []

    def fake_post_filter_voice(
        audio: np.ndarray,
        sample_rate: int,
        *,
        settings: dict[str, object] | None = None,
    ) -> np.ndarray:
        del sample_rate
        captured_settings.append(dict(settings or {}))
        return np.asarray(audio, dtype=np.float32)

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fake_convert_file_with_freevc)
    monkeypatch.setattr(processor_module, "post_filter_voice", fake_post_filter_voice)

    pipeline = VoiceConversionPipeline(sample_rate=22050, dsp_profiles_dir=str(tmp_dir / "dsp_profiles"))
    result = pipeline.convert_speaker_clone_file(str(source_path), [str(reference_path)])

    assert result.metrics["dsp_neural_safe_filter"] == 1.0
    assert captured_settings
    assert captured_settings[0]["use_noisereduce"] is False
    assert captured_settings[0]["use_pedalboard"] is False
    assert captured_settings[0]["post_gain_db"] <= 0.0
    assert captured_settings[0]["deess_reduction_db"] <= 2.0


def test_dsp_autotune_updates_profile_even_when_freevc_is_used(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("dsp_autotune_freevc")
    source = _sine(duration=0.25)
    reference = _sine(duration=0.25) * 0.5
    source_path = tmp_dir / "input.wav"
    reference_path = tmp_dir / "reference.wav"
    sf.write(str(source_path), source, 22050)
    sf.write(str(reference_path), reference, 22050)

    def fake_convert_file_with_freevc(
        input_path: str,
        reference_path_arg: str,
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCConversionResult:
        del input_path, reference_path_arg, sample_rate, kwargs
        loud = source.copy()
        loud[loud.size // 2] = 1.4
        return FreeVCConversionResult(
            audio=np.asarray(loud, dtype=np.float32),
            config=FreeVCModelConfig(
                model_id="freevc-24-one-shot",
                assets_dir=tmp_dir,
                checkpoint_path=tmp_dir / "freevc-24.pth",
                config_path=tmp_dir / "freevc-24.json",
                speaker_encoder_path=tmp_dir / "speaker.pt",
                wavlm_model="models/hf/wavlm-large",
            ),
        )

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fake_convert_file_with_freevc)

    dsp_profiles_dir = tmp_dir / "dsp_profiles"
    pipeline = VoiceConversionPipeline(sample_rate=22050, dsp_profiles_dir=str(dsp_profiles_dir))
    result = pipeline.convert_speaker_clone_file(str(source_path), [str(reference_path)])

    registry = json.loads((dsp_profiles_dir / "registry.json").read_text(encoding="utf-8"))
    profile = registry["profiles"]["speaker_clone"]

    assert result.metrics["freevc_engine"] == 1.0
    assert result.metrics["dsp_profile_updated"] == 1.0
    assert result.metrics["dsp_neural_safe_filter"] == 1.0
    assert profile["last_engine"] == "freevc"
    assert profile["updated_from_runs"] == 1
    assert "post_peak" in profile["metrics_ema"]


def test_gender_age_file_uses_rvc_lazily_when_registry_matches(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("rvc_lazy")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (model_dir / "female_local.index").write_bytes(b"fake local rvc index")
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "model_id": "female_local",
                        "pitch": 2,
                        "index_rate": 0.25,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, object]] = []

    class FakeRVCInference:
        def __init__(self, device: str = "cpu") -> None:
            calls.append(("init", device))

        def load_model(self, model_path: str, index_path: str = "") -> None:
            calls.append(("load", (Path(model_path).name, Path(index_path).name)))

        def set_params(self, **kwargs: object) -> None:
            calls.append(("params", (kwargs.get("f0up_key"), kwargs.get("index_rate"))))

        def infer_file(self, input_path: str, output_path: str) -> None:
            calls.append(("infer", Path(output_path).name))
            audio, _ = sf.read(input_path, dtype="float32")
            sf.write(output_path, np.asarray(audio, dtype=np.float32) * 0.5, 22050)

    monkeypatch.setattr(rvc_adapter, "_RVC_INFERENCE_CLASS", FakeRVCInference)
    monkeypatch.setattr(rvc_adapter, "_RVC_INSTANCE_CACHE", {})

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")
    assert calls == []

    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.metrics["rvc_engine"] == 1.0
    assert calls == [
        ("init", "cpu"),
        ("load", ("female_local.pth", "female_local.index")),
        ("params", (2, 0.25)),
        ("infer", "rvc_output.wav"),
    ]


def test_rvc_registry_accepts_applio_engine_options():
    tmp_dir = _workspace_tmp_dir("rvc_applio_config")
    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "model_id": "female_local",
                        "engine": "applio",
                        "pitch": 3,
                        "index_rate": 0.4,
                        "f0_method": "fcpe",
                        "volume_envelope": 0.8,
                        "protect": 0.35,
                        "split_audio": True,
                        "f0_autotune": True,
                        "f0_autotune_strength": 0.5,
                        "proposed_pitch": True,
                        "proposed_pitch_threshold": 255.0,
                        "embedder_model": "contentvec",
                        "clean_audio": True,
                        "clean_strength": 0.25,
                        "applio_root": str(tmp_dir / "applio"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = rvc_adapter.get_gender_age_rvc_config("male_to_female", models_dir=models_dir)

    assert config is not None
    assert config.engine == "applio"
    assert config.f0_method == "fcpe"
    assert config.volume_envelope == 0.8
    assert config.protect == 0.35
    assert config.split_audio is True
    assert config.f0_autotune is True
    assert config.f0_autotune_strength == 0.5
    assert config.proposed_pitch is True
    assert config.proposed_pitch_threshold == 255.0
    assert config.clean_audio is True
    assert config.clean_strength == 0.25
    assert config.applio_root == (tmp_dir / "applio").resolve()


def test_rvc_registry_rejects_unknown_engine():
    tmp_dir = _workspace_tmp_dir("rvc_bad_engine")
    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "female_local", "engine": "mystery"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported RVC engine"):
        rvc_adapter.get_gender_age_rvc_config("male_to_female", models_dir=models_dir)


def test_gender_age_file_uses_applio_when_registry_selects_it(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("rvc_applio")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (model_dir / "female_local.index").write_bytes(b"fake local rvc index")
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "model_id": "female_local",
                        "engine": "applio",
                        "pitch": 4,
                        "index_rate": 0.45,
                        "f0_method": "rmvpe",
                        "protect": 0.3,
                        "split_audio": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, object]] = []

    def fake_preprocess(audio: np.ndarray, sample_rate: int) -> SpectrogramImageResult:
        del sample_rate
        return SpectrogramImageResult(
            audio=np.asarray(audio, dtype=np.float32),
            metrics={"opencv_spectrogram_applied": 0.0},
        )

    def fake_convert_file_with_applio(
        input_path: str | Path,
        output_path: str | Path,
        model_path: str | Path,
        index_path: str | Path | None,
        **kwargs: object,
    ) -> None:
        settings = kwargs["settings"]
        calls.append(
            (
                "applio",
                (
                    Path(input_path).name,
                    Path(model_path).name,
                    Path(index_path).name if index_path else "",
                    kwargs["pitch"],
                    kwargs["index_rate"],
                    settings.f0_method,
                    settings.protect,
                    settings.split_audio,
                ),
            )
        )
        audio, _ = sf.read(str(input_path), dtype="float32")
        sf.write(str(output_path), np.asarray(audio, dtype=np.float32) * 0.35, 22050)

    monkeypatch.setattr(processor_module, "preprocess_spectrogram_for_model", fake_preprocess)
    monkeypatch.setattr(rvc_adapter, "convert_file_with_applio", fake_convert_file_with_applio)

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.metrics["rvc_engine"] == 1.0
    assert calls == [
        (
            "applio",
            ("input.wav", "female_local.pth", "female_local.index", 4, 0.45, "rmvpe", 0.3, True),
        )
    ]


def test_rvc_auto_falls_back_to_applio_when_rvc_python_is_missing(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("rvc_auto_applio")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "female_local", "engine": "auto"}}}),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_rvc_python(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("rvc-python")
        raise RuntimeError("RVC model is configured, but rvc-python is not installed.")

    def fake_applio(
        input_path: str | Path,
        output_path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        calls.append("applio")
        audio, _ = sf.read(str(input_path), dtype="float32")
        sf.write(str(output_path), np.asarray(audio, dtype=np.float32) * 0.25, 22050)

    monkeypatch.setattr(rvc_adapter, "_infer_file_with_rvc_python", fake_rvc_python)
    monkeypatch.setattr(rvc_adapter, "is_applio_available", lambda applio_root=None: True)
    monkeypatch.setattr(rvc_adapter, "convert_file_with_applio", fake_applio)

    result = rvc_adapter.convert_gender_age_with_rvc(
        str(source_path),
        "male_to_female",
        22050,
        models_dir=models_dir,
        device="cpu",
    )

    assert result is not None
    assert result.engine == "applio"
    assert calls == ["rvc-python", "applio"]


def test_gender_age_file_uses_freevc_reference_profile_when_configured(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("freevc_gender")
    source = _sine(duration=0.25)
    reference = _sine(duration=0.25) * 0.6
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    profiles_dir = tmp_dir / "profiles"
    references_dir = profiles_dir / "references"
    references_dir.mkdir(parents=True)
    reference_path = references_dir / "female_local_reference.wav"
    sf.write(str(reference_path), reference, 22050)
    (profiles_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "profile_id": "female_local_reference",
                        "reference_path": "references/female_local_reference.wav",
                        "consent_required": True,
                        "consent_owner": "authorized_local_voice",
                        "license": "private-consent",
                        "allow_any_source": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[Path, Path, int, object]] = []

    def fake_convert_file_with_freevc(
        input_path: str,
        reference_path_arg: str,
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCConversionResult:
        calls.append((Path(input_path), Path(reference_path_arg), sample_rate, kwargs["assets_dir"]))
        return FreeVCConversionResult(
            audio=np.asarray(source * 0.4, dtype=np.float32),
            config=FreeVCModelConfig(
                model_id="freevc-24-one-shot",
                assets_dir=tmp_dir,
                checkpoint_path=tmp_dir / "freevc-24.pth",
                config_path=tmp_dir / "freevc-24.json",
                speaker_encoder_path=tmp_dir / "speaker.pt",
                wavlm_model="models/hf/wavlm-large",
            ),
        )

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fake_convert_file_with_freevc)

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(tmp_dir / "rvc"),
        freevc_assets_dir="models/hf/freevc-24",
        freevc_profiles_dir=str(profiles_dir),
    )
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.metrics["rvc_engine"] == 0.0
    assert result.metrics["freevc_engine"] == 1.0
    assert result.metrics["freevc_gender_age_refine"] == 0.0
    assert result.metrics["dsp_neural_safe_filter"] == 1.0
    assert calls == [(source_path, reference_path.resolve(), 22050, "models/hf/freevc-24")]


def test_gender_age_freevc_reference_profile_can_opt_into_dsp_refine(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("freevc_gender_refine")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    profiles_dir = tmp_dir / "profiles"
    references_dir = profiles_dir / "references"
    references_dir.mkdir(parents=True)
    reference_path = references_dir / "female_local_reference.wav"
    sf.write(str(reference_path), source * 0.6, 22050)
    (profiles_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "profile_id": "female_local_reference",
                        "reference_path": "references/female_local_reference.wav",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_convert_file_with_freevc(
        input_path: str,
        reference_path_arg: str,
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCConversionResult:
        del input_path, reference_path_arg, sample_rate, kwargs
        return FreeVCConversionResult(
            audio=np.asarray(source * 0.4, dtype=np.float32),
            config=FreeVCModelConfig(
                model_id="freevc-24-one-shot",
                assets_dir=tmp_dir,
                checkpoint_path=tmp_dir / "freevc-24.pth",
                config_path=tmp_dir / "freevc-24.json",
                speaker_encoder_path=tmp_dir / "speaker.pt",
                wavlm_model="models/hf/wavlm-large",
            ),
        )

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fake_convert_file_with_freevc)
    monkeypatch.setenv("OMNISPEECH_FREEVC_GENDER_AGE_REFINE", "1")

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(tmp_dir / "rvc"),
        freevc_profiles_dir=str(profiles_dir),
        dsp_profiles_dir=str(tmp_dir / "dsp_profiles"),
    )
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.metrics["freevc_engine"] == 1.0
    assert result.metrics["freevc_gender_age_refine"] == 1.0


def test_gender_age_freevc_reference_profile_missing_file_is_explicit():
    tmp_dir = _workspace_tmp_dir("freevc_gender_missing")
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), _sine(duration=0.25), 22050)

    profiles_dir = tmp_dir / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "profile_id": "female_local_reference",
                        "reference_path": "references/female_local_reference.wav",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(tmp_dir / "rvc"),
        freevc_profiles_dir=str(profiles_dir),
    )

    with pytest.raises(FileNotFoundError, match="FreeVC reference profile configured.*file not found"):
        pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")


def test_gender_age_rvc_receives_spectrogram_preprocessed_audio(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("rvc_preprocessed")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "female_local"}}}),
        encoding="utf-8",
    )

    observed_peaks: list[float] = []

    def fake_preprocess(audio: np.ndarray, sample_rate: int) -> SpectrogramImageResult:
        del sample_rate
        return SpectrogramImageResult(
            audio=np.asarray(audio, dtype=np.float32) * 0.1,
            metrics={"opencv_spectrogram_applied": 1.0},
        )

    class FakeRVCInference:
        def __init__(self, device: str = "cpu") -> None:
            del device

        def load_model(self, model_path: str, index_path: str = "") -> None:
            del model_path, index_path

        def infer_file(self, input_path: str, output_path: str, **kwargs: object) -> None:
            del kwargs
            audio, _ = sf.read(input_path, dtype="float32")
            observed_peaks.append(float(np.max(np.abs(audio))))
            sf.write(output_path, np.asarray(audio, dtype=np.float32), 22050)

    monkeypatch.setattr(processor_module, "preprocess_spectrogram_for_model", fake_preprocess)
    monkeypatch.setattr(rvc_adapter, "_RVC_INFERENCE_CLASS", FakeRVCInference)
    monkeypatch.setattr(rvc_adapter, "_RVC_INSTANCE_CACHE", {})

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.metrics["rvc_engine"] == 1.0
    assert result.metrics["opencv_spectrogram_applied"] == 1.0
    assert observed_peaks and observed_peaks[0] <= 0.11


def test_celebrity_file_uses_rvc_lazily_when_registry_matches(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("rvc_celebrity")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "licensed_profile_local"
    model_dir.mkdir(parents=True)
    (model_dir / "licensed_profile_local.pth").write_bytes(b"fake local rvc model")
    (model_dir / "licensed_profile_local.index").write_bytes(b"fake local rvc index")
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "celebrity": {
                    "michael_jackson": {
                        "model_id": "licensed_profile_local",
                        "pitch": 1,
                        "index_rate": 0.35,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, object]] = []

    class FakeRVCInference:
        def __init__(self, device: str = "cpu") -> None:
            calls.append(("init", device))

        def load_model(self, model_path: str, index_path: str = "") -> None:
            calls.append(("load", (Path(model_path).name, Path(index_path).name)))

        def set_params(self, **kwargs: object) -> None:
            calls.append(("params", (kwargs.get("f0up_key"), kwargs.get("index_rate"))))

        def infer_file(self, input_path: str, output_path: str) -> None:
            calls.append(("infer", Path(output_path).name))
            audio, _ = sf.read(input_path, dtype="float32")
            sf.write(output_path, np.asarray(audio, dtype=np.float32) * 0.5, 22050)

    monkeypatch.setattr(rvc_adapter, "_RVC_INFERENCE_CLASS", FakeRVCInference)
    monkeypatch.setattr(rvc_adapter, "_RVC_INSTANCE_CACHE", {})

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")
    result = pipeline.convert_celebrity_file(str(source_path), celebrity="michael_jackson")

    assert result.metrics["rvc_engine"] == 1.0
    assert calls == [
        ("init", "cpu"),
        ("load", ("licensed_profile_local.pth", "licensed_profile_local.index")),
        ("params", (1, 0.35)),
        ("infer", "rvc_output.wav"),
    ]


def test_gender_age_file_can_force_pure_dsp_even_when_ai_profiles_exist(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("pure_dsp_gender")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "female_local"}}}),
        encoding="utf-8",
    )

    profiles_dir = tmp_dir / "freevc_profiles"
    references_dir = profiles_dir / "references"
    references_dir.mkdir(parents=True)
    reference_path = references_dir / "female_reference.wav"
    sf.write(str(reference_path), source * 0.5, 22050)
    (profiles_dir / "registry.json").write_text(
        json.dumps(
            {
                "gender_age": {
                    "male_to_female": {
                        "profile_id": "female_reference",
                        "reference_path": "references/female_reference.wav",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fail_rvc(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("rvc")
        raise AssertionError("RVC should be skipped in pure DSP mode")

    def fail_freevc(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("freevc")
        raise AssertionError("FreeVC should be skipped in pure DSP mode")

    monkeypatch.setattr(processor_module, "convert_gender_age_with_rvc", fail_rvc)
    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fail_freevc)

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(models_dir),
        freevc_profiles_dir=str(profiles_dir),
    )
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female", use_ai_engines=False)

    assert calls == []
    assert result.metrics["ai_engines_enabled"] == 0.0
    assert result.metrics["rvc_engine"] == 0.0
    assert result.metrics["freevc_engine"] == 0.0


def test_speaker_clone_file_can_force_pure_dsp_even_with_references(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("pure_dsp_speaker")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    reference_path = tmp_dir / "reference.wav"
    sf.write(str(source_path), source, 22050)
    sf.write(str(reference_path), source * 0.7, 22050)

    calls: list[str] = []

    def fail_freevc(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("freevc")
        raise AssertionError("FreeVC should be skipped in pure DSP mode")

    monkeypatch.setattr(processor_module, "convert_file_with_freevc", fail_freevc)

    pipeline = VoiceConversionPipeline(sample_rate=22050, dsp_profiles_dir=str(tmp_dir / "dsp_profiles"))
    result = pipeline.convert_speaker_clone_file(
        str(source_path),
        [str(reference_path)],
        use_ai_engines=False,
    )

    assert calls == []
    assert result.metrics["ai_engines_enabled"] == 0.0
    assert result.metrics["freevc_engine"] == 0.0
    assert result.metrics["reference_count"] == 1.0


def test_celebrity_file_can_force_pure_dsp_even_when_rvc_profile_exists(monkeypatch: pytest.MonkeyPatch):
    tmp_dir = _workspace_tmp_dir("pure_dsp_celebrity")
    source = _sine(duration=0.25)
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), source, 22050)

    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "licensed_profile_local"
    model_dir.mkdir(parents=True)
    (model_dir / "licensed_profile_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"celebrity": {"michael_jackson": {"model_id": "licensed_profile_local"}}}),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fail_rvc(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("rvc")
        raise AssertionError("RVC should be skipped in pure DSP mode")

    monkeypatch.setattr(processor_module, "convert_file_with_rvc", fail_rvc)

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")
    result = pipeline.convert_celebrity_file(str(source_path), celebrity="michael_jackson", use_ai_engines=False)

    assert calls == []
    assert result.metrics["ai_engines_enabled"] == 0.0
    assert result.metrics["rvc_engine"] == 0.0


def test_gender_age_rvc_registry_missing_model_is_explicit():
    tmp_dir = _workspace_tmp_dir("rvc_missing")
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), _sine(duration=0.25), 22050)

    models_dir = tmp_dir / "rvc"
    models_dir.mkdir()
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "missing_local"}}}),
        encoding="utf-8",
    )

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(models_dir), rvc_device="cpu")

    with pytest.raises(FileNotFoundError, match="RVC model configured.*file not found"):
        pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")


def test_optional_ai_fallbacks_keep_finite_float32(monkeypatch: pytest.MonkeyPatch):
    source = _sine(duration=0.25)

    monkeypatch.setattr("backend.audio.filtering._apply_pedalboard_post_filter", lambda audio, sample_rate: None)
    filtered = post_filter_voice(source, 22050)

    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))

    monkeypatch.setattr(emotion_module, "_pitch_shift_with_parselmouth", lambda audio, sample_rate, n_steps: None)
    converted = emotion_module.convert_emotion(source, 22050, "calm")

    assert converted.dtype == np.float32
    assert np.all(np.isfinite(converted))


def test_spectrogram_preprocess_falls_back_without_opencv(monkeypatch: pytest.MonkeyPatch):
    source = _sine(duration=0.25)
    monkeypatch.setitem(sys.modules, "cv2", None)

    result = preprocess_spectrogram_for_model(source, 22050)

    assert result.metrics["opencv_spectrogram_applied"] == 0.0
    assert result.audio.dtype == np.float32
    assert np.allclose(result.audio, source)


def test_post_filter_limits_spikes_and_keeps_finite_output():
    sr = 22050
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
    source = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    source[source.size // 2] = 1.8

    filtered = post_filter_voice(source, sr)

    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))
    assert float(np.max(np.abs(filtered))) <= 0.96


def test_post_filter_softens_clustered_crackle():
    sr = 22050
    t = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False, dtype=np.float32)
    source = (0.16 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    start = int(sr * 0.16)
    cluster = (0.22 * np.sign(np.sin(2 * np.pi * 4_800 * t[: int(sr * 0.035)]))).astype(np.float32)
    source[start : start + cluster.size] += cluster

    filtered = post_filter_voice(source, sr, settings={"use_noisereduce": False, "use_pedalboard": False})

    before_jump = float(np.percentile(np.abs(np.diff(source)), 99.7))
    after_jump = float(np.percentile(np.abs(np.diff(filtered)), 99.7))
    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))
    assert after_jump < before_jump * 0.82


def test_live_post_filter_limits_chunk_energy():
    sr = 22050
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    chunk = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    chunk[chunk.size // 3] = 1.6

    filtered = LiveVoicePostFilter(sr).process(chunk)

    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))
    assert float(np.max(np.abs(filtered))) <= 0.96


def test_live_overlap_crossfader_softens_chunk_boundary():
    smoother = LiveOverlapCrossfader(22050)
    first = np.full(2048, 0.4, dtype=np.float32)
    second = np.full(2048, -0.4, dtype=np.float32)

    out_first = smoother.process(first)
    out_second = smoother.process(second)

    raw_jump = abs(float(first[-1]) - float(second[0]))
    smoothed_jump = abs(float(out_first[-1]) - float(out_second[0]))
    assert out_first.size == first.size
    assert out_second.size == second.size
    assert smoothed_jump < raw_jump * 0.2
    assert float(out_second[700]) < -0.35


def test_excited_profile_prioritizes_smooth_dsp():
    profile = emotion_module.EMOTION_PROFILES["excited"]

    assert profile["pitch_shift"] <= 1.25
    assert profile["rate"] <= 1.04
    assert profile["jitter"] == 0.0
    assert profile["vibrato_rate"] == 0.0
    assert profile["drive"] <= 1.04


def test_excited_conversion_protects_unvoiced_and_silence(monkeypatch: pytest.MonkeyPatch):
    sr = 22050
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    voiced = (0.18 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.08, int(sr * 0.14)).astype(np.float32)
    fricative = (noise - np.convolve(noise, np.ones(19, dtype=np.float32) / 19.0, mode="same")).astype(np.float32)
    silence = np.zeros(int(sr * 0.12), dtype=np.float32)
    source = np.concatenate([voiced, fricative, silence])

    monkeypatch.setattr(emotion_module, "_pitch_shift_with_parselmouth", lambda audio, sample_rate, n_steps: None)
    converted = emotion_module.convert_emotion(source, sr, "excited")

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float32) ** 2)))

    voiced_slice = slice(0, voiced.size)
    fricative_slice = slice(voiced.size, voiced.size + fricative.size)
    silence_slice = slice(voiced.size + fricative.size, None)
    voiced_delta = rms(converted[voiced_slice] - source[voiced_slice])
    fricative_delta = rms(converted[fricative_slice] - source[fricative_slice])

    assert converted.dtype == np.float32
    assert np.all(np.isfinite(converted))
    assert fricative_delta < voiced_delta * 0.35
    assert float(np.max(np.abs(converted[silence_slice]))) < 0.025


def test_normalize_audio_rms_mode_hits_target_level():
    audio = np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float32)

    normalized = normalize_audio(audio, mode="rms", target_rms=0.1)
    rms = float(np.sqrt(np.mean(normalized**2)))

    assert np.isclose(rms, 0.1, atol=1e-4)


def test_pure_dsp_clean_chain_removes_dc_and_limits_spikes():
    sr = 22050
    source = _sine(sr=sr, duration=0.35) + 0.18
    source[source.size // 2] = 1.7

    result = prepare_clean_speech(source, sr)

    assert result.metrics["pure_dsp_input_cleaned"] == 1.0
    assert abs(float(np.mean(result.audio))) < abs(float(np.mean(source)))
    assert float(np.max(np.abs(result.audio))) <= 0.95
    assert result.metrics["pure_dsp_voiced_ratio"] > 0.0


def test_pure_dsp_merge_protects_silence_regions():
    sr = 22050
    voiced = _sine(sr=sr, duration=0.25)
    source = np.concatenate([np.zeros(sr // 10, dtype=np.float32), voiced])
    converted = np.full(source.size, 0.8, dtype=np.float32)

    merged = merge_preserving_noise_regions(source, converted, sr)

    assert float(np.max(np.abs(merged[: sr // 12]))) < 0.35
    assert float(np.max(np.abs(merged[-sr // 12 :]))) > 0.2


def test_user_feedback_updates_dsp_profile_settings():
    tmp_dir = _workspace_tmp_dir("dsp_feedback")

    before = get_dsp_profile_settings("gender_age", tmp_dir)
    after = apply_user_feedback("gender_age", "robotic", profiles_dir=tmp_dir)

    registry = json.loads((tmp_dir / "registry.json").read_text(encoding="utf-8"))
    assert after.transform_intensity < before.transform_intensity
    assert after.formant_smoothing > before.formant_smoothing
    assert registry["profiles"]["gender_age"]["feedback_counts"]["robotic"] == 1


def test_submodule_dsp_profile_inherits_parent_settings():
    tmp_dir = _workspace_tmp_dir("dsp_subprofile_inherit")
    (tmp_dir / "registry.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "gender_age": {"settings": {"presence_db": 1.25, "transform_intensity": 0.82}},
                    "gender_age.male_to_female": {"settings": {"transform_intensity": 0.63}},
                }
            }
        ),
        encoding="utf-8",
    )

    settings = get_dsp_profile_settings("gender_age.male_to_female", tmp_dir)

    assert settings.presence_db == 1.25
    assert settings.transform_intensity == 0.63


def test_pipeline_updates_submodule_dsp_profile():
    tmp_dir = _workspace_tmp_dir("dsp_subprofile_update")
    source_path = tmp_dir / "input.wav"
    sf.write(str(source_path), _sine(duration=0.35), 22050)

    pipeline = VoiceConversionPipeline(
        sample_rate=22050,
        rvc_models_dir=str(tmp_dir / "rvc"),
        dsp_profiles_dir=str(tmp_dir / "dsp_profiles"),
    )
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female", use_ai_engines=False)

    registry = json.loads((tmp_dir / "dsp_profiles" / "registry.json").read_text(encoding="utf-8"))
    assert "gender_age.male_to_female" in registry["profiles"]
    assert registry["profiles"]["gender_age.male_to_female"]["parent_profile"] == "gender_age"
    assert result.metrics["dsp_profile_updated"] == 1.0
    assert result.metrics["dsp_quality_optimizer_applied"] == 1.0
    assert result.metrics["dsp_quality_candidates"] == 3.0


def test_pitch_confidence_tracks_voiced_sine():
    track = analyze_pitch_confidence(_sine(duration=0.45), 22050)
    voiced = track.f0_hz[(track.f0_hz > 0) & (track.confidence >= 0.25)]

    assert voiced.size > 0
    assert 170.0 <= float(np.median(voiced)) <= 270.0


def test_artifact_metrics_expose_scores():
    sr = 22050
    source = _sine(sr=sr, duration=0.35)
    source[source.size // 3] = 1.0

    metrics = measure_audio_health(source, sr, prefix="post_")

    assert "post_artifact_score" in metrics
    assert 0.0 <= metrics["post_artifact_score"] <= 1.0
    assert "post_artifact_pitch_score" in metrics
    assert "post_spectrogram_artifact_score" in metrics
    assert "post_artifact_spectrogram_score" in metrics


def test_spectrogram_quality_exposes_band_and_artifact_scores():
    sr = 22050
    t = np.linspace(0, 0.45, int(sr * 0.45), endpoint=False, dtype=np.float32)
    source = (0.18 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 6_200 * t)).astype(np.float32)

    metrics = analyze_spectrogram_quality(source, sr, prefix="post_")

    assert 0.0 <= metrics["post_spectrogram_sibilance_ratio"] <= 1.0
    assert metrics["post_spectrogram_sibilance_ratio"] > 0.0
    assert 0.0 <= metrics["post_spectrogram_artifact_score"] <= 1.0
    assert 0.0 <= metrics["post_spectrogram_balance_score"] <= 1.0


def test_quality_optimizer_selects_best_scored_variant():
    sr = 22050
    source = _sine(sr=sr, duration=0.45)
    source = (source + 0.025 * np.sin(2 * np.pi * 6_000 * np.linspace(0, 0.45, source.size, endpoint=False))).astype(np.float32)

    result = optimize_post_filter_quality(
        source,
        sr,
        DSPProfileSettings(post_gain_db=1.5, presence_db=2.0, spectral_tilt_db=1.5),
        engine="dsp",
    )

    assert result.candidate_count == 3
    assert result.selected_score <= result.baseline_score + 1e-8
    assert result.metrics["dsp_quality_optimizer_applied"] == 1.0
    assert result.metrics["dsp_quality_candidates"] == 3.0
    assert 0.0 <= result.metrics["post_artifact_score"] <= 1.0
    assert "post_spectrogram_artifact_score" in result.metrics
    assert "dsp_quality_candidate_0_spectrogram" in result.metrics


def test_gender_age_cepstral_formant_warp_keeps_finite_output():
    converted = gender_age_module.convert_gender_age(_sine(duration=0.35), 22050, "male_to_female")

    assert converted.dtype == np.float32
    assert np.all(np.isfinite(converted))
    assert converted.size > 0


def test_live_chunk_processing_without_virtual_mic():
    pipeline = VoiceConversionPipeline(sample_rate=22050)
    manager = LiveSessionManager(pipeline=pipeline, sample_rate=22050)

    session = manager.start_session(
        task="gender_age",
        options={"mode": "male_to_female"},
        route_to_virtual_mic=False,
        virtual_mic_device=None,
    )

    out = manager.process_chunk(session.session_id, _sine(duration=0.1))
    manager.stop_session(session.session_id)

    assert out.size > 0


def test_live_chunk_processing_for_emotion():
    pipeline = VoiceConversionPipeline(sample_rate=22050)
    manager = LiveSessionManager(pipeline=pipeline, sample_rate=22050)

    session = manager.start_session(
        task="emotion",
        options={"emotion": "excited"},
        route_to_virtual_mic=False,
        virtual_mic_device=None,
    )

    out = manager.process_chunk(session.session_id, _sine(duration=0.1))
    manager.stop_session(session.session_id)

    assert out.size > 0
