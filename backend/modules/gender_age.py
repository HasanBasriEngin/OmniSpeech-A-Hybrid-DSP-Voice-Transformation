from __future__ import annotations

import librosa
import numpy as np
from scipy import signal
import torch
import torch.nn as nn


CONVERSION_PRESETS: dict[str, dict[str, float]] = {
    "male_to_female": {"pitch_shift": 5.8, "warp": 1.24, "brightness": 0.34, "warmth": -0.08, "breath": 0.012},
    "female_to_male": {"pitch_shift": -5.4, "warp": 0.78, "brightness": -0.28, "warmth": 0.34, "breath": 0.0},
    "adult_to_child": {"pitch_shift": 7.2, "warp": 1.38, "brightness": 0.45, "warmth": -0.2, "breath": 0.006},
    "adult_to_elderly": {"pitch_shift": -2.4, "warp": 0.9, "brightness": -0.36, "warmth": 0.18, "breath": 0.028},
    "child_to_adult": {"pitch_shift": -6.2, "warp": 0.74, "brightness": -0.18, "warmth": 0.28, "breath": 0.0},
}


class SpectralWarp(nn.Module):
    """Applies a lightweight spectral-envelope warp with PyTorch tensors."""

    def forward(self, audio: torch.Tensor, warp_factor: float) -> torch.Tensor:
        if audio.ndim != 1:
            raise ValueError("SpectralWarp expects a 1D mono tensor.")

        spec = torch.fft.rfft(audio)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        idx = torch.arange(mag.numel(), device=audio.device, dtype=torch.float32)
        src_idx = torch.clamp(idx / max(warp_factor, 1e-4), 0, mag.numel() - 1)

        low = torch.floor(src_idx).long()
        high = torch.clamp(low + 1, max=mag.numel() - 1)
        frac = src_idx - low.float()
        warped_mag = mag[low] * (1.0 - frac) + mag[high] * frac

        complex_spec = warped_mag * torch.exp(1j * phase)
        return torch.fft.irfft(complex_spec, n=audio.numel())


WARP_MODEL = SpectralWarp()


def _fft_tone_shape(audio: np.ndarray, brightness: float, warmth: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    high_shelf = 1.0 + brightness * np.clip((freqs - 0.18) / 0.62, 0.0, 1.0)
    low_shelf = 1.0 + warmth * np.exp(-((freqs - 0.09) ** 2) / (2 * 0.08**2))
    presence = 1.0 + max(brightness, 0.0) * 0.32 * np.exp(-((freqs - 0.36) ** 2) / (2 * 0.055**2))
    curve = np.clip(high_shelf * low_shelf * presence, 0.25, 2.5)
    return np.fft.irfft(spec * curve, n=x.size).astype(np.float32)


def _soft_compress(audio: np.ndarray, drive: float = 1.18) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    return np.tanh(x * drive).astype(np.float32)


def _add_breath_and_tremor(audio: np.ndarray, sample_rate: int, amount: float, mode: str) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    out = x
    if mode == "adult_to_elderly":
        t = np.arange(x.size, dtype=np.float32) / max(sample_rate, 1)
        tremor = 1.0 + 0.045 * np.sin(2 * np.pi * 5.2 * t)
        out = (out * tremor).astype(np.float32)

    if amount > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, amount, x.size).astype(np.float32)
        sos = signal.butter(2, 1800, btype="highpass", fs=sample_rate, output="sos")
        breath = signal.sosfilt(sos, noise).astype(np.float32)
        out = out + breath

    return out.astype(np.float32)


def _peak_normalize(audio: np.ndarray, peak_target: float = 0.97) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-7:
        return x
    return (x * (peak_target / peak)).astype(np.float32)


def convert_gender_age(audio: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    preset = CONVERSION_PRESETS.get(mode)
    if preset is None:
        allowed = ", ".join(sorted(CONVERSION_PRESETS))
        raise ValueError(f"Unsupported conversion mode '{mode}'. Allowed: {allowed}")

    x = np.asarray(audio, dtype=np.float32)
    pitched = librosa.effects.pitch_shift(y=x, sr=sample_rate, n_steps=preset["pitch_shift"])

    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(pitched, dtype=np.float32))
        warped = WARP_MODEL(tensor, preset["warp"]).cpu().numpy().astype(np.float32)

    shaped = _fft_tone_shape(warped, brightness=preset["brightness"], warmth=preset["warmth"])
    textured = _add_breath_and_tremor(shaped, sample_rate, preset["breath"], mode)
    compressed = _soft_compress(textured)
    return np.clip(_peak_normalize(compressed), -1.0, 1.0).astype(np.float32)
