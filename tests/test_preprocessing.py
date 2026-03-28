import numpy as np

from core.preprocessing import compute_energy_envelope, extract_features


def test_energy_envelope_nonempty():
    x = np.random.randn(32000).astype(np.float32) * 0.01
    env = compute_energy_envelope(x)
    assert env.size > 0


def test_extract_features_keys():
    x = np.random.randn(32000).astype(np.float32) * 0.01
    feats = extract_features(x, sr=16000)
    assert {"mfcc", "mel_spectrogram", "stft", "chroma"} <= set(feats.keys())

