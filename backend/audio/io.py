"""
io.py — Ses dosyası yükleme, kaydetme ve normalizasyon yardımcıları.

Kişi 3 değişiklikleri (Eren DÖNMEZ):
  - load_audio_mono: Çok kısa ses dosyası (<0.3 sn) için uyarı eklendi.
  - normalize_audio: RMS tabanlı normalizasyon seçeneği eklendi (mode="rms").
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Minimum ses süresi uyarı eşiği (saniye)
_MIN_DURATION_WARN_SEC: float = 0.3


def validate_audio_path(path: str) -> Path:
    """Dosya varlığını ve uzantısını doğrula, çözümlenmiş Path döndür."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Audio file not found: {target}")
    if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {target.suffix}")
    return target


def load_audio_mono(
    path: str,
    sample_rate: int,
    *,
    normalize_mode: str = "peak",
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Ses dosyasını mono float32 olarak yükle ve normalize et.

    Kişi 3 iyileştirmesi: Yüklenen sesin süresi ``_MIN_DURATION_WARN_SEC``
    saniyeden kısaysa bir uyarı verir. Bu, çok kısa klipler için
    DSP işlemlerinin anlamsız veya hatalı sonuç üretebileceğine dair
    erken bir sinyal görevi görür.

    Parametreler
    ------------
    path : str
        Ses dosyasının yolu.
    sample_rate : int
        Hedef örnekleme frekansı (Hz). Gerekirse yeniden örnekleme yapılır.

    Döndürür
    --------
    np.ndarray
        Peak-normalize edilmiş mono float32 ses verisi.
    """
    target = validate_audio_path(path)
    audio, _ = librosa.load(str(target), sr=sample_rate, mono=True)

    # Çok kısa ses uyarısı
    duration_sec = audio.size / max(sample_rate, 1)
    if duration_sec < _MIN_DURATION_WARN_SEC:
        warnings.warn(
            f"load_audio_mono: '{target.name}' çok kısa "
            f"({duration_sec:.3f} sn < {_MIN_DURATION_WARN_SEC} sn). "
            "DSP sonuçları güvenilir olmayabilir.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "Çok kısa ses dosyası yüklendi: %s (%.3f sn)",
            target.name,
            duration_sec,
        )

    return normalize_audio(audio, mode=normalize_mode, target_rms=target_rms)


def normalize_audio(
    audio: np.ndarray,
    mode: str = "peak",
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Ses verisini normalize et.

    Kişi 3 iyileştirmesi: ``mode`` parametresi eklendi.

    Parametreler
    ------------
    audio : np.ndarray
        Normalize edilecek ses verisi.
    mode : str
        Normalizasyon modu:
        - ``"peak"`` (varsayılan): Maksimum genliği 1.0'a çeker
          (orijinal davranış, geriye dönük uyumluluk korunur).
        - ``"rms"`` : RMS değerini ``target_rms`` değerine çeker;
          dinamik aralığı korur, zirve değerler 1.0'ı aşabilir
          ve ardından klamp'lanır.
    target_rms : float
        RMS modu için hedef RMS değeri (varsayılan 0.1 ≈ -20 dBFS).

    Döndürür
    --------
    np.ndarray
        Normalize edilmiş float32 ses verisi.

    Örnekler
    --------
    >>> normalized = normalize_audio(audio, mode="peak")
    >>> normalized_rms = normalize_audio(audio, mode="rms", target_rms=0.08)
    """
    arr = np.asarray(audio, dtype=np.float32)

    if mode not in {"peak", "rms"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    if mode == "rms":
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        if rms <= 1e-7:
            return arr
        gain = target_rms / rms
        return np.clip(arr * gain, -1.0, 1.0).astype(np.float32)

    # Varsayılan: peak normalizasyon
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 1e-7:
        return arr
    return (arr / peak).astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> str:
    """Ses verisini WAV olarak kaydet, gerekirse dizinleri oluştur."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), np.asarray(audio, dtype=np.float32), sample_rate)
    return str(output)


def default_output_path(input_path: str, suffix: str) -> str:
    """Girdi dosyasının yanında suffix ile yeni bir WAV yolu oluştur."""
    source = Path(input_path).expanduser().resolve()
    return str(source.with_stem(f"{source.stem}_{suffix}").with_suffix(".wav"))
