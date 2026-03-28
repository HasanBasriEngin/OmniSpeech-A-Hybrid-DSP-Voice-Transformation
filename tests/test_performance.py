import numpy as np

from core.evaluation import measure_processing_time
from modules.emotion_conversion import convert_emotion


def test_processing_time_under_target_for_short_audio():
    x = np.random.randn(16000 * 3).astype(np.float32) * 0.01
    t = measure_processing_time(convert_emotion, x, 16000, "calm")
    assert t < 2.0

