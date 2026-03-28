from __future__ import annotations

from pathlib import Path

import numpy as np

from core.preprocessing import extract_features
from utils.audio_utils import match_length


def extract_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    feats = extract_features(audio, sr)
    mfcc = feats["mfcc"]
    emb = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def separate_content_identity(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    feats = extract_features(audio, sr)
    content = np.mean(feats["mel_spectrogram"], axis=0).astype(np.float32)
    identity = extract_speaker_embedding(audio, sr)
    return content, identity


def reconstruct_with_target(content: np.ndarray, target_embedding: np.ndarray) -> np.ndarray:
    content = np.asarray(content, dtype=np.float32)
    target_embedding = np.asarray(target_embedding, dtype=np.float32)
    modulation = float(np.mean(target_embedding[: min(16, len(target_embedding))])) * 0.25
    base = np.interp(
        np.linspace(0, len(content) - 1, max(1024, len(content) * 128)),
        np.arange(len(content)),
        content,
    )
    return np.tanh(base + modulation).astype(np.float32)


def convert_speaker(audio: np.ndarray, sr: int, target_embedding: np.ndarray) -> np.ndarray:
    content, _ = separate_content_identity(audio, sr)
    converted = reconstruct_with_target(content, target_embedding)
    converted = match_length(converted, len(audio))
    mixed = 0.75 * converted + 0.25 * np.asarray(audio, dtype=np.float32)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def clone_voice(audio: np.ndarray, sr: int, reference_samples: list[np.ndarray]) -> np.ndarray:
    if not reference_samples:
        return np.asarray(audio, dtype=np.float32)
    embeddings = [extract_speaker_embedding(ref, sr) for ref in reference_samples]
    target = np.mean(np.stack(embeddings), axis=0)
    return convert_speaker(audio, sr, target)


def export_embedding(embedding: np.ndarray, path: str) -> None:
    out = Path(path)
    if out.suffix.lower() != ".npy":
        out = out.with_suffix(".npy")
    np.save(str(out), np.asarray(embedding, dtype=np.float32))

