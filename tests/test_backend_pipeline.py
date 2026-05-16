from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from backend.audio.filtering import LiveVoicePostFilter, post_filter_voice
from backend.audio.io import normalize_audio
from backend.audio.spectrogram_image import preprocess_spectrogram_for_model
from backend.modules import emotion as emotion_module
from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.live_session import LiveSessionManager


def _sine(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _workspace_tmp_dir(prefix: str) -> Path:
    tmp_dir = Path(".tmp") / "tests" / f"{prefix}_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def test_gender_age_file_conversion():
    tmp_dir = _workspace_tmp_dir("gender")
    source = _sine()
    source_path = tmp_dir / "in.wav"
    sf.write(str(source_path), source, 22050)

    pipeline = VoiceConversionPipeline(sample_rate=22050, rvc_models_dir=str(tmp_dir / "models"))
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.output_path.endswith(".wav")
    assert result.metrics["processing_seconds"] >= 0.0
    assert result.metrics["rvc_engine"] == 0.0
    assert "opencv_spectrogram_applied" in result.metrics


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


def test_live_post_filter_limits_chunk_energy():
    sr = 22050
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    chunk = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    chunk[chunk.size // 3] = 1.6

    filtered = LiveVoicePostFilter(sr).process(chunk)

    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))
    assert float(np.max(np.abs(filtered))) <= 0.96


def test_normalize_audio_rms_mode_hits_target_level():
    audio = np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float32)

    normalized = normalize_audio(audio, mode="rms", target_rms=0.1)
    rms = float(np.sqrt(np.mean(normalized**2)))

    assert np.isclose(rms, 0.1, atol=1e-4)


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
