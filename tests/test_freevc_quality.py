from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from backend.audio.reference_preprocess import prepare_reference_from_paths, preprocess_reference_audio
from backend.audio.reference_quality import analyze_reference_quality
from backend.modules.freevc_adapter import (
    FreeVCConversionResult,
    FreeVCModelConfig,
    FreeVCSpeakerCloneAttempt,
    attempt_speaker_clone_with_freevc,
    get_freevc_config,
)


def _sine(sr: int = 22050, duration: float = 1.0, freq: float = 220.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _workspace_tmp_dir(prefix: str) -> Path:
    tmp_dir = Path(".tmp") / "tests" / f"{prefix}_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def _write_wav(path: Path, audio: np.ndarray, sr: int = 22050) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def test_short_reference_lowers_quality_score():
    short = _sine(duration=0.4)
    report = analyze_reference_quality(short, 22050)

    assert report.duration_seconds < 10.0
    assert report.quality_score < 0.75
    assert report.low_quality_warning is True


def test_clipped_reference_sets_warning_metric():
    clipped = _sine(duration=2.0)
    clipped = np.clip(clipped * 6.0, -1.0, 1.0).astype(np.float32)

    report = analyze_reference_quality(clipped, 22050)

    assert report.clipping_warning is True
    assert report.clipping_ratio > 0.0
    assert report.quality_score < 0.85


def test_multi_reference_selection_picks_cleanest():
    tmp_path = _workspace_tmp_dir("freevc_multi_ref")
    noisy = _sine(duration=12.0) + (np.random.randn(int(22050 * 12.0)).astype(np.float32) * 0.08)
    clean = _sine(duration=12.0, amplitude=0.25)

    noisy_path = tmp_path / "noisy.wav"
    clean_path = tmp_path / "clean.wav"
    _write_wav(noisy_path, noisy)
    _write_wav(clean_path, clean)

    def loader(path: str, sample_rate: int) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            ratio = sample_rate / sr
            idx = np.linspace(0, audio.size - 1, int(audio.size * ratio)).astype(int)
            audio = audio[idx]
        return np.asarray(audio, dtype=np.float32)

    prepared = prepare_reference_from_paths([str(noisy_path), str(clean_path)], 22050, loader)
    assert prepared is not None
    assert prepared.source_index == 1
    assert prepared.metrics["selected_reference_index"] == 1.0
    assert prepared.metrics["reference_candidate_count"] == 2.0


def test_preprocess_normalizes_and_trims():
    sr = 22050
    lead = np.zeros(int(sr * 0.4), dtype=np.float32)
    body = _sine(sr=sr, duration=2.0, amplitude=0.05)
    audio = np.concatenate([lead, body, lead])

    processed = preprocess_reference_audio(audio, sr)
    report = analyze_reference_quality(processed, sr)

    assert processed.size < audio.size
    assert report.rms_db > -35.0
    assert np.max(np.abs(processed)) <= 1.0


def test_attempt_freevc_skips_without_assets():
    tmp_dir = _workspace_tmp_dir("freevc_skip")
    source_path = tmp_dir / "in.wav"
    ref_path = tmp_dir / "ref.wav"
    _write_wav(source_path, _sine(duration=1.5))
    _write_wav(ref_path, _sine(duration=12.0))

    attempt = attempt_speaker_clone_with_freevc(
        str(source_path),
        [str(ref_path)],
        22050,
        assets_dir=str(tmp_dir / "missing_freevc"),
    )

    assert attempt.result is None
    assert attempt.metrics["freevc_engine"] == 0.0
    assert "reference_quality_score" in attempt.metrics
    assert attempt.metrics.get("freevc_skipped_reason", 0.0) >= 3.0


def test_speaker_clone_pipeline_uses_freevc_metrics(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch")
    from backend.pipeline import processor as processor_module
    from backend.pipeline.processor import VoiceConversionPipeline

    tmp_dir = _workspace_tmp_dir("freevc_pipeline")
    source_path = tmp_dir / "input.wav"
    ref_path = tmp_dir / "reference.wav"
    _write_wav(source_path, _sine(duration=0.5))
    _write_wav(ref_path, _sine(duration=12.0))

    fake_config = FreeVCModelConfig(
        model_id="freevc-24-one-shot",
        assets_dir=tmp_dir,
        checkpoint_path=tmp_dir / "freevc-24.pth",
        config_path=tmp_dir / "freevc-24.json",
        speaker_encoder_path=tmp_dir / "speaker.pt",
        wavlm_model="microsoft/wavlm-large",
    )

    def fake_attempt(
        input_path: str,
        reference_paths: list[str],
        sample_rate: int,
        **kwargs: object,
    ) -> FreeVCSpeakerCloneAttempt:
        del kwargs
        assert Path(input_path) == source_path
        assert len(reference_paths) == 1
        assert sample_rate == 22050
        metrics = {
            "freevc_engine": 1.0,
            "reference_quality_score": 0.91,
            "reference_duration_seconds": 12.0,
            "selected_reference_index": 0.0,
        }
        return FreeVCSpeakerCloneAttempt(
            result=FreeVCConversionResult(
                audio=_sine(duration=0.5) * 0.3,
                config=fake_config,
                metrics=metrics,
            ),
            metrics=metrics,
        )

    monkeypatch.setattr(processor_module, "attempt_speaker_clone_with_freevc", fake_attempt)

    pipeline = VoiceConversionPipeline(sample_rate=22050)
    result = pipeline.convert_speaker_clone_file(str(source_path), [str(ref_path)])

    assert result.metrics["freevc_engine"] == 1.0
    assert result.metrics["reference_quality_score"] == pytest.approx(0.91)
    assert result.metrics["reference_duration_seconds"] == pytest.approx(12.0)


def test_get_freevc_config_returns_none_for_incomplete_assets():
    tmp_dir = _workspace_tmp_dir("freevc_cfg")
    assert get_freevc_config(assets_dir=tmp_dir) is None
