from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from backend.audio.spectrogram_image import SpectrogramImageResult
from backend.modules import rvc_adapter
from backend.pipeline import processor as processor_module
from backend.pipeline.processor import VoiceConversionPipeline
from backend.tools.import_rvc_model import import_rvc_model


def _sine(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _workspace_tmp_dir(prefix: str) -> Path:
    tmp_dir = Path(".tmp") / "tests" / f"{prefix}_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def test_rvc_config_returns_none_when_registry_is_missing():
    tmp_dir = _workspace_tmp_dir("rvc_registry_missing")

    assert rvc_adapter.get_rvc_config("gender_age", "male_to_female", models_dir=tmp_dir) is None


def test_rvc_config_applies_safe_metadata_defaults():
    tmp_dir = _workspace_tmp_dir("rvc_metadata_defaults")
    models_dir = tmp_dir / "rvc"
    model_dir = models_dir / "female_local"
    model_dir.mkdir(parents=True)
    (model_dir / "female_local.pth").write_bytes(b"fake local rvc model")
    (models_dir / "registry.json").write_text(
        json.dumps({"gender_age": {"male_to_female": {"model_id": "female_local"}}}),
        encoding="utf-8",
    )

    config = rvc_adapter.get_rvc_config("gender_age", "male_to_female", models_dir=models_dir)

    assert config is not None
    assert config.consent_required is True
    assert config.consent_owner == ""
    assert config.license == "unknown"
    assert config.allow_any_source is False
    assert config.index_path is None


def test_rvc_config_reads_metadata_fields_when_present():
    tmp_dir = _workspace_tmp_dir("rvc_metadata_explicit")
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
                        "consent_required": False,
                        "consent_owner": "licensed_profile_owner",
                        "license": "studio-license",
                        "allow_any_source": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = rvc_adapter.get_rvc_config("celebrity", "michael_jackson", models_dir=models_dir)

    assert config is not None
    assert config.pitch == 1
    assert config.index_rate == 0.35
    assert config.consent_required is False
    assert config.consent_owner == "licensed_profile_owner"
    assert config.license == "studio-license"
    assert config.allow_any_source is True
    assert config.index_path is not None
    assert config.index_path.name == "licensed_profile_local.index"


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


def test_import_tool_copies_artifacts_and_updates_registry():
    tmp_dir = _workspace_tmp_dir("rvc_import_tool")
    source_dir = tmp_dir / "source"
    source_dir.mkdir()
    pth_path = source_dir / "my_voice_source.pth"
    pth_path.write_bytes(b"fake pth bytes")
    index_path = source_dir / "my_voice_source.index"
    index_path.write_bytes(b"fake index bytes")

    models_dir = tmp_dir / "models"
    args = argparse.Namespace(
        model_id="my_voice",
        pth=str(pth_path),
        index=str(index_path),
        category="gender_age",
        key="male_to_female",
        pitch=3,
        index_rate=0.6,
        consent_required="true",
        consent_owner="authorized_local_voice",
        license="private-consent",
        allow_any_source="false",
        models_dir=str(models_dir),
    )

    registry_path = import_rvc_model(args)

    assert registry_path == models_dir.resolve() / "registry.json"
    assert (models_dir / "my_voice" / "my_voice.pth").read_bytes() == b"fake pth bytes"
    assert (models_dir / "my_voice" / "my_voice.index").read_bytes() == b"fake index bytes"

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["gender_age"]["male_to_female"] == {
        "model_id": "my_voice",
        "pitch": 3,
        "index_rate": 0.6,
        "consent_required": True,
        "consent_owner": "authorized_local_voice",
        "license": "private-consent",
        "allow_any_source": False,
    }
