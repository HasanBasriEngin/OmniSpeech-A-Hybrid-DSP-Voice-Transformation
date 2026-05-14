from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectrogramImageResult:
    audio: np.ndarray
    metrics: dict[str, float]


def _safe_mono_float(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x, dtype=np.float32)


def _base_metrics() -> dict[str, float]:
    return {
        "opencv_spectrogram_enabled": 1.0,
        "opencv_spectrogram_available": 0.0,
        "opencv_spectrogram_applied": 0.0,
    }


def preprocess_spectrogram_for_model(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_size: tuple[int, int] = (256, 256),
    blur_kernel: int = 3,
    blend: float = 0.35,
) -> SpectrogramImageResult:
    """
    Treat the STFT magnitude as an image, process it with OpenCV, and rebuild audio.

    The phase is kept from the original signal, so this behaves like a light
    spectrogram-domain denoising/preconditioning step before an AI converter.
    If OpenCV or Librosa is unavailable, the original audio is returned with
    metrics that make the fallback visible.
    """
    x = _safe_mono_float(audio)
    metrics = _base_metrics()
    metrics["opencv_spectrogram_input_samples"] = float(x.size)

    if x.size < 512 or sample_rate <= 0:
        return SpectrogramImageResult(audio=x, metrics=metrics)

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on optional wheel
        logger.debug("OpenCV spectrogram preprocessing skipped: %s", exc)
        return SpectrogramImageResult(audio=x, metrics=metrics)

    try:
        import librosa
    except Exception as exc:  # pragma: no cover - requirements path
        logger.debug("Librosa spectrogram preprocessing skipped: %s", exc)
        return SpectrogramImageResult(audio=x, metrics=metrics)

    metrics["opencv_spectrogram_available"] = 1.0

    n_fft = 1024
    hop_length = 256
    try:
        complex_spec = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window="hann")
    except Exception as exc:  # pragma: no cover - defensive around audio edge cases
        logger.warning("Could not build STFT for OpenCV preprocessing: %s", exc)
        return SpectrogramImageResult(audio=x, metrics=metrics)

    magnitude = np.abs(complex_spec).astype(np.float32)
    if magnitude.size == 0 or float(np.max(magnitude)) <= 1e-8:
        return SpectrogramImageResult(audio=x, metrics=metrics)

    phase = np.exp(1j * np.angle(complex_spec)).astype(np.complex64)
    log_mag = np.log1p(magnitude).astype(np.float32)
    log_min = float(np.min(log_mag))
    log_max = float(np.max(log_mag))
    log_range = log_max - log_min
    if log_range <= 1e-8:
        return SpectrogramImageResult(audio=x, metrics=metrics)

    image = ((log_mag - log_min) / log_range).astype(np.float32)
    original_height, original_width = image.shape
    target_width = max(8, int(target_size[0]))
    target_height = max(8, int(target_size[1]))
    safe_blend = float(np.clip(blend, 0.0, 1.0))

    try:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        kernel = int(blur_kernel)
        if kernel > 1:
            if kernel % 2 == 0:
                kernel += 1
            resized = cv2.GaussianBlur(resized, (kernel, kernel), 0)
        normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
        restored = cv2.resize(normalized, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    except Exception as exc:  # pragma: no cover - OpenCV runtime edge cases
        logger.warning("OpenCV spectrogram preprocessing failed: %s", exc)
        return SpectrogramImageResult(audio=x, metrics=metrics)

    restored = np.clip(np.asarray(restored, dtype=np.float32), 0.0, 1.0)
    restored_log = restored * log_range + log_min
    processed_mag = np.expm1(restored_log).astype(np.float32)
    mixed_mag = ((1.0 - safe_blend) * magnitude + safe_blend * processed_mag).astype(np.float32)
    rebuilt_spec = mixed_mag.astype(np.complex64) * phase

    try:
        rebuilt = librosa.istft(rebuilt_spec, hop_length=hop_length, window="hann", length=x.size)
    except Exception as exc:  # pragma: no cover - defensive around audio edge cases
        logger.warning("Could not rebuild audio after OpenCV preprocessing: %s", exc)
        return SpectrogramImageResult(audio=x, metrics=metrics)

    rebuilt = _safe_mono_float(rebuilt)
    source_rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    rebuilt_rms = float(np.sqrt(np.mean(rebuilt**2))) if rebuilt.size else 0.0
    if source_rms > 1e-8 and rebuilt_rms > 1e-8:
        rebuilt = (rebuilt * (source_rms / rebuilt_rms)).astype(np.float32)

    peak = float(np.max(np.abs(rebuilt))) if rebuilt.size else 0.0
    if peak > 1.0:
        rebuilt = (rebuilt / peak).astype(np.float32)

    metrics.update(
        {
            "opencv_spectrogram_applied": 1.0,
            "opencv_spectrogram_height": float(original_height),
            "opencv_spectrogram_width": float(original_width),
            "opencv_spectrogram_target_height": float(target_height),
            "opencv_spectrogram_target_width": float(target_width),
            "opencv_spectrogram_blur_kernel": float(max(0, int(blur_kernel))),
            "opencv_spectrogram_blend": float(round(safe_blend, 3)),
        }
    )
    return SpectrogramImageResult(audio=np.asarray(rebuilt, dtype=np.float32), metrics=metrics)
