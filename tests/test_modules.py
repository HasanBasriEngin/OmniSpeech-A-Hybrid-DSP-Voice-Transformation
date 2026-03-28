import numpy as np

from modules.emotion_conversion import convert_emotion
from modules.gender_age_conversion import convert_gender_age
from modules.singing_voice import speech_to_singing
from modules.speaker_conversion import convert_speaker, extract_speaker_embedding


def _audio():
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    return 0.2 * np.sin(2 * np.pi * 220 * t)


def test_emotion_conversion():
    y = convert_emotion(_audio(), sr=16000, target_emotion="calm")
    assert y.shape[0] == 16000


def test_gender_age_conversion():
    y = convert_gender_age(_audio(), sr=16000, conversion_type="adult_to_child")
    assert y.shape[0] == 16000


def test_speaker_conversion():
    x = _audio()
    emb = extract_speaker_embedding(x, 16000)
    y = convert_speaker(x, 16000, emb)
    assert y.shape[0] == x.shape[0]


def test_singing_conversion():
    x = _audio()
    contour = np.linspace(180, 260, 40).astype(np.float32)
    y = speech_to_singing(x, 16000, contour, input_type="pitch_contour")
    assert y.shape[0] == x.shape[0]

