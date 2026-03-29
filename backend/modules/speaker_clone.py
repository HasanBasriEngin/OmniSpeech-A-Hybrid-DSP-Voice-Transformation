from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn

from backend.audio.features import extract_mfcc, stretch_to_length


class SpeakerStyleAdapter(nn.Module):
    """Tiny conditioning model to inject speaker identity into the source signal."""

    def __init__(self, embedding_dim: int = 40) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gain = 0.65 + 0.7 * self.gate(embedding.unsqueeze(0)).squeeze(0)
        return torch.tanh(audio * gain)


STYLE_ADAPTER = SpeakerStyleAdapter()


def speaker_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = extract_mfcc(audio, sample_rate, n_mfcc=20)
    stats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)], axis=0).astype(np.float32)
    norm = np.linalg.norm(stats) + 1e-8
    return (stats / norm).astype(np.float32)


def clone_speaker(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if not references:
        return source

    ref_embeddings = [speaker_embedding(np.asarray(ref, dtype=np.float32), sample_rate) for ref in references]
    target = np.mean(np.stack(ref_embeddings), axis=0).astype(np.float32)

    with torch.no_grad():
        source_tensor = torch.from_numpy(source)
        embedding_tensor = torch.from_numpy(target)
        styled = STYLE_ADAPTER(source_tensor, embedding_tensor).cpu().numpy().astype(np.float32)

    shifted = librosa.effects.pitch_shift(y=styled, sr=sample_rate, n_steps=0.75)
    return stretch_to_length(np.asarray(shifted, dtype=np.float32), source.size)
