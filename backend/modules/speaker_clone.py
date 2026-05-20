from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from backend.audio.features import extract_mfcc, extract_pitch_contour, pitch_shift_audio, stretch_to_length


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


def _smooth_spectrum(magnitude: np.ndarray, taps: int = 23) -> np.ndarray:
    values = np.asarray(magnitude, dtype=np.float32)
    if values.size < 4:
        return values
    taps = min(max(5, taps), values.size if values.size % 2 == 1 else values.size - 1)
    kernel = np.hanning(taps).astype(np.float32)
    kernel /= float(np.sum(kernel)) + 1e-8
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _median_f0(audio: np.ndarray, sample_rate: int) -> float:
    contour = extract_pitch_contour(audio, sample_rate)
    voiced = contour[contour > 0]
    return float(np.median(voiced)) if voiced.size else 0.0


def _match_reference_timbre(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    del sample_rate
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 32 or not references:
        return x

    source_spec = np.fft.rfft(x)
    source_env = _smooth_spectrum(np.log(np.abs(source_spec) + 1e-6))
    ref_envs: list[np.ndarray] = []
    for ref in references:
        ref_x = stretch_to_length(np.asarray(ref, dtype=np.float32), x.size)
        ref_envs.append(_smooth_spectrum(np.log(np.abs(np.fft.rfft(ref_x)) + 1e-6)))
    if not ref_envs:
        return x

    ref_env = np.mean(np.stack(ref_envs, axis=0), axis=0).astype(np.float32)
    ratio = np.exp(np.clip(ref_env - source_env, -0.65, 0.65)).astype(np.float32)
    ratio = np.clip(0.55 * ratio + 0.45, 0.55, 1.75)
    matched = np.fft.irfft(source_spec * ratio, n=x.size).astype(np.float32)
    return (0.72 * x + 0.28 * matched).astype(np.float32)


def _match_reference_pitch(audio: np.ndarray, sample_rate: int, references: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if not references:
        return x
    source_hz = _median_f0(x, sample_rate)
    ref_hz_values = [_median_f0(ref, sample_rate) for ref in references]
    ref_hz_values = [value for value in ref_hz_values if value > 0]
    if source_hz <= 0 or not ref_hz_values:
        return x
    ref_hz = float(np.median(np.asarray(ref_hz_values, dtype=np.float32)))
    semitones = float(np.clip(12.0 * np.log2(ref_hz / max(source_hz, 1.0)) * 0.35, -2.5, 2.5))
    return pitch_shift_audio(x, sample_rate, semitones)


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

    timbre_matched = _match_reference_timbre(styled, sample_rate, references)
    pitch_matched = _match_reference_pitch(timbre_matched, sample_rate, references)
    shifted = pitch_shift_audio(pitch_matched, sample_rate, 0.45)
    return stretch_to_length(np.asarray(shifted, dtype=np.float32), source.size)
