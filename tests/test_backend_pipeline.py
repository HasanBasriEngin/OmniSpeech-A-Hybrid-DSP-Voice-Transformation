from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.live_session import LiveSessionManager


def _sine(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_gender_age_file_conversion():
    source = _sine()
    tmp_dir = Path(".tmp") / "tests"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_dir / "in.wav"
    sf.write(str(source_path), source, 22050)

    pipeline = VoiceConversionPipeline(sample_rate=22050)
    result = pipeline.convert_gender_age_file(str(source_path), mode="male_to_female")

    assert result.output_path.endswith(".wav")
    assert result.metrics["processing_seconds"] >= 0.0


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
