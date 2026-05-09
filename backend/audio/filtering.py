"""
Shared post-processing filters for converted voice output.

This module now covers two cases:
1. Offline file conversion via ``post_filter_voice`` / ``apply_post_filter``.
2. Chunked live processing via ``LiveVoicePostFilter``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

_SPEECH_LOW_HZ = 55.0
_SPEECH_HIGH_HZ = 8_800.0
_DECLICK_BASE_THRESH = 0.08
_DECLICK_WINDOW_MS = 2.5
_SOFT_LIMIT_CEIL = 0.96
_DEFAULT_KNEE_DB = 6.0
_DESS_FREQ_HZ = 6_800.0
_DESS_BANDWIDTH_HZ = 5_200.0
_DESS_REDUCTION_DB = 4.5


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert arbitrary audio input to safe mono float32."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _fade_edges(audio: np.ndarray, sample_rate: int, milliseconds: float = 4.0) -> np.ndarray:
    """Apply a short fade in/out to reduce boundary clicks on file output."""
    x = np.asarray(audio, dtype=np.float32).copy()
    if x.size < 4:
        return x

    fade_samples = int(sample_rate * milliseconds / 1000.0)
    fade_samples = min(max(fade_samples, 1), x.size // 2)
    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    x[:fade_samples] *= fade
    x[-fade_samples:] *= fade[::-1]
    return x


def _apply_pedalboard_post_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Run the optional Pedalboard chain for offline processing."""
    x = _as_mono_float(audio)
    if x.size < 4 or sample_rate <= 0:
        return x

    try:
        from pedalboard import Compressor, HighpassFilter, Limiter, NoiseGate, Pedalboard
    except Exception:
        return None

    try:
        board = Pedalboard(
            [
                NoiseGate(threshold_db=-34.0, ratio=1.8),
                HighpassFilter(cutoff_frequency_hz=55.0),
                Compressor(threshold_db=-20.0, ratio=3.5, attack_ms=4.0, release_ms=120.0),
                Limiter(threshold_db=-0.8, release_ms=60.0),
            ]
        )
        processed = board(x, sample_rate)
    except Exception as exc:  # pragma: no cover - depends on optional binary wheels
        logger.warning("Pedalboard post-filter failed; falling back to SciPy chain: %s", exc)
        return None

    return _as_mono_float(processed)


def _apply_noisereduce(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Run optional spectral-gate denoising for offline converted files."""
    x = _as_mono_float(audio)
    if x.size < 512 or sample_rate <= 0:
        return x

    try:
        import noisereduce as nr
    except Exception:
        return None

    try:
        reduced = nr.reduce_noise(
            y=x,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.55,
            time_constant_s=1.2,
            freq_mask_smooth_hz=420,
            time_mask_smooth_ms=70,
        )
    except TypeError:
        reduced = nr.reduce_noise(y=x, sr=sample_rate)
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("noisereduce post-filter failed; keeping previous audio: %s", exc)
        return None

    return _as_mono_float(reduced)


def _build_speech_band_sos(sample_rate: int) -> np.ndarray | None:
    """Build a reusable speech-band bandpass filter."""
    if sample_rate <= 0:
        return None

    nyquist = sample_rate / 2.0
    low_hz = _SPEECH_LOW_HZ
    high_hz = min(_SPEECH_HIGH_HZ, nyquist * 0.98)
    if low_hz >= high_hz:
        return None

    return signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")


def _speech_band_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the shared offline speech-band filter."""
    x = _as_mono_float(audio)
    if x.size < 16:
        return x

    sos = _build_speech_band_sos(sample_rate)
    if sos is None:
        warnings.warn(
            f"_speech_band_filter: sample_rate={sample_rate} is too low for filtering; returning audio unchanged.",
            stacklevel=2,
        )
        return x

    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        logger.warning("_speech_band_filter failed: %s", exc)
        return x


def _declick(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove short click spikes using an adaptive derivative threshold."""
    x = _as_mono_float(audio)
    if x.size == 0:
        return x

    win_samples = max(1, int(_DECLICK_WINDOW_MS * sample_rate / 1000.0))
    x_sq = x**2
    kernel = np.ones(win_samples, dtype=np.float32) / win_samples
    local_rms = np.sqrt(np.convolve(x_sq, kernel, mode="same").clip(min=0.0))

    global_rms = float(np.sqrt(np.mean(x_sq))) if x.size else 1e-7
    if global_rms < 1e-7:
        return x

    adaptive_thresh = np.clip(
        _DECLICK_BASE_THRESH * (local_rms / (global_rms + 1e-7)),
        _DECLICK_BASE_THRESH * 0.5,
        _DECLICK_BASE_THRESH * 4.0,
    ).astype(np.float32)

    diff = np.concatenate([[0.0], np.abs(np.diff(x))]).astype(np.float32)
    mask: np.ndarray = diff > adaptive_thresh
    if not np.any(mask):
        return x

    out = x.copy()
    click_idx = np.where(mask)[0]
    for ci in click_idx:
        left = max(0, ci - 2)
        right = min(x.size - 1, ci + 2)
        if right > left:
            out[left : right + 1] = np.linspace(x[left], x[right], right - left + 1)

    return out.astype(np.float32)


def _soft_limit(
    audio: np.ndarray,
    ceiling: float = _SOFT_LIMIT_CEIL,
    knee_db: float = _DEFAULT_KNEE_DB,
) -> np.ndarray:
    """
    Apply a soft knee below the ceiling and a safe hard cap above it.

    The old implementation attempted a tanh roll-off above the ceiling but still
    collapsed into a hard clip. This version keeps the soft knee behavior and
    uses an explicit ceiling cap for overflow samples.
    """
    x = _as_mono_float(audio)
    if x.size == 0:
        return x

    ceiling = float(np.clip(ceiling, 0.01, 1.0))
    knee_db = max(0.0, float(knee_db))
    if knee_db == 0.0:
        return np.clip(x, -ceiling, ceiling).astype(np.float32)

    knee_start = ceiling * (10.0 ** (-knee_db / 20.0))
    abs_x = np.abs(x)
    out = x.copy()

    in_knee = (abs_x > knee_start) & (abs_x <= ceiling)
    if np.any(in_knee):
        t = (abs_x[in_knee] - knee_start) / max(ceiling - knee_start, 1e-7)
        gamma = 1.0 + knee_db / 6.0
        shaped = np.power(t, gamma)
        target_mag = knee_start + shaped * (ceiling - knee_start)
        out[in_knee] = np.sign(x[in_knee]) * target_mag

    above = abs_x > ceiling
    if np.any(above):
        out[above] = np.sign(x[above]) * ceiling

    return out.astype(np.float32)


def _deess(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reduce excessive sibilance when the high-frequency band dominates."""
    x = _as_mono_float(audio)
    if x.size < 64:
        return x

    nyquist = sample_rate / 2.0
    center = _DESS_FREQ_HZ
    bw_half = _DESS_BANDWIDTH_HZ / 2.0
    low_hz = max(center - bw_half, 500.0)
    high_hz = min(center + bw_half, nyquist * 0.98)
    if low_hz >= high_hz:
        return x

    try:
        sos_band = signal.butter(3, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
        sibilant = signal.sosfiltfilt(sos_band, x).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        logger.warning("_deess failed: %s", exc)
        return x

    full_rms = float(np.sqrt(np.mean(x**2))) + 1e-7
    sib_rms = float(np.sqrt(np.mean(sibilant**2))) + 1e-7
    ratio = sib_rms / full_rms
    if ratio < 0.25:
        return x

    reduction_lin = 10.0 ** (-_DESS_REDUCTION_DB / 20.0)
    gain = 1.0 - (1.0 - reduction_lin) * np.clip((ratio - 0.25) / 0.75, 0.0, 1.0)
    return (x - sibilant + sibilant * float(gain)).astype(np.float32)


def apply_post_filter(
    audio: np.ndarray,
    sample_rate: int,
    *,
    speech_band: bool = True,
    declick: bool = True,
    soft_limit: bool = True,
    deess: bool = True,
    knee_db: float = _DEFAULT_KNEE_DB,
    ceiling: float = _SOFT_LIMIT_CEIL,
    use_noisereduce: bool = True,
    use_pedalboard: bool = True,
) -> np.ndarray:
    """Apply the shared offline post-processing chain."""
    x = _as_mono_float(audio)

    if use_noisereduce:
        denoised = _apply_noisereduce(x, sample_rate)
        if denoised is not None:
            x = denoised

    if use_pedalboard:
        pedalboard_filtered = _apply_pedalboard_post_filter(x, sample_rate)
        if pedalboard_filtered is not None:
            x = pedalboard_filtered

    if speech_band:
        x = _speech_band_filter(x, sample_rate)
    if declick:
        x = _declick(x, sample_rate)
    if deess:
        x = _deess(x, sample_rate)
    if soft_limit:
        x = _soft_limit(x, ceiling=ceiling, knee_db=knee_db)

    return np.asarray(x, dtype=np.float32)


def post_filter_voice(audio: np.ndarray, sample_rate: int, *, realtime: bool = False) -> np.ndarray:
    """
    Public convenience wrapper used by conversion pipelines.

    Offline output gets a short fade to avoid file boundary clicks.
    Live output skips that fade because chunks are stitched continuously.
    """
    x = _as_mono_float(audio)
    if x.size == 0:
        return x

    x = x - float(np.mean(x))
    x = apply_post_filter(x, sample_rate, use_noisereduce=not realtime, use_pedalboard=not realtime)

    if not realtime:
        x = _fade_edges(x, sample_rate)

    return np.asarray(x, dtype=np.float32)


class LiveVoicePostFilter:
    """Stateful speech-band filter chain for chunked live microphone sessions."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._speech_band_sos = _build_speech_band_sos(sample_rate)
        self._speech_band_zi: np.ndarray | None = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = _as_mono_float(audio)
        if x.size == 0:
            return x

        x = x - float(np.mean(x))

        if self._speech_band_sos is not None:
            if self._speech_band_zi is None:
                self._speech_band_zi = signal.sosfilt_zi(self._speech_band_sos) * x[0]
            x, self._speech_band_zi = signal.sosfilt(self._speech_band_sos, x, zi=self._speech_band_zi)
            x = np.asarray(x, dtype=np.float32)

        x = _declick(x, self.sample_rate)
        x = _deess(x, self.sample_rate)
        x = _soft_limit(x)
        return np.asarray(x, dtype=np.float32)
