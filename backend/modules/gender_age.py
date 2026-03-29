from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn


CONVERSION_PRESETS: dict[str, dict[str, float]] = {
    "male_to_female": {"pitch_shift": 4.5, "warp": 1.18},
    "female_to_male": {"pitch_shift": -4.0, "warp": 0.84},
    "adult_to_child": {"pitch_shift": 6.0, "warp": 1.32},
    "adult_to_elderly": {"pitch_shift": -1.5, "warp": 0.92},
    "child_to_adult": {"pitch_shift": -5.0, "warp": 0.78},
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

    peak = float(np.max(np.abs(warped))) if warped.size else 1.0
    if peak > 1e-7:
        warped = warped / peak
    return np.clip(warped, -1.0, 1.0).astype(np.float32)
