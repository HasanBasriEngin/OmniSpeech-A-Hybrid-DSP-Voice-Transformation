import numpy as np

from core.input_module import normalize_amplitude, segment_audio


def test_normalize_amplitude():
    x = np.array([0.0, 2.0, -2.0], dtype=np.float32)
    y = normalize_amplitude(x)
    assert np.isclose(np.max(np.abs(y)), 1.0)


def test_segment_audio():
    x = np.ones(16000 * 12, dtype=np.float32)
    segments = segment_audio(x, sr=16000, max_duration=5.0)
    assert len(segments) == 3

