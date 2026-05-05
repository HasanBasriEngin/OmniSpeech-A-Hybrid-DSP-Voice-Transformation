"""
filtering.py — Ortak post-processing DSP pipeline.

Tüm ses dönüşüm modülleri (gender_age, emotion, vb.) bu dosyadaki
fonksiyonları paylaşır. Her fonksiyon saf (pure) NumPy/SciPy işlemi
olarak tutulur; PyTorch bağımlılığı yoktur.

Kişi 3 değişiklikleri (Eren DÖNMEZ):
  - _speech_band_filter: Bandpass aralığı 55 Hz – 10 500 Hz olarak genişletildi
  - _declick: Adaptif sinyal-bağımlı threshold eklendi
  - _soft_limit: Knee parametresi ile yumuşak compression eklendi
  - De-essing filtresi eklendi (yüksek frekanslı sibilant azaltma)
  - apply_post_filter: Tüm adımları sırayla uygulayan herkese açık fonksiyon
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

_SPEECH_LOW_HZ: float = 55.0      # Alt kesim frekansı (Hz) — önceki değer ~80 Hz idi
_SPEECH_HIGH_HZ: float = 10_500.0 # Üst kesim frekansı (Hz) — önceki değer ~8 kHz idi
_DECLICK_BASE_THRESH: float = 0.08 # Taban declick eşiği (sinyal enerjisine göre ölçeklenir)
_DECLICK_WINDOW_MS: float = 2.5    # Pencere uzunluğu (ms)
_SOFT_LIMIT_CEIL: float = 0.96     # Yumuşak limiter çıkış tavanı
_DEFAULT_KNEE_DB: float = 6.0      # Varsayılan knee genişliği (dB)
_DESS_FREQ_HZ: float = 6_800.0     # De-esser merkez frekansı (Hz)
_DESS_BANDWIDTH_HZ: float = 5_200.0 # De-esser bandwidth (Hz)
_DESS_REDUCTION_DB: float = 4.5    # De-esser azaltma miktarı (dB)


# ---------------------------------------------------------------------------
# Dahili yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _speech_band_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    55 Hz – 10.5 kHz bandpass filtresi.

    Çift yönlü (zero-phase) Butterworth filtre kullanır; bu nedenle
    fazda bozulma olmaz.  Nyquist sınırını geçmemek için üst frekans
    otomatik olarak klamp'lanır.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 16:
        return x

    nyquist = sample_rate / 2.0
    low = _SPEECH_LOW_HZ / nyquist
    high = min(_SPEECH_HIGH_HZ, nyquist * 0.98) / nyquist  # Nyquist taşmasını önle

    if low >= high:
        warnings.warn(
            f"_speech_band_filter: sample_rate={sample_rate} çok düşük, "
            "filtre uygulanamıyor — ses değiştirilmeden döndürülüyor.",
            stacklevel=2,
        )
        return x

    try:
        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        logger.warning("_speech_band_filter başarısız: %s", exc)
        return x


def _declick(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Adaptif sinyal-bağımlı declick.

    Sabit bir eşik yerine yerel RMS enerjisine göre dinamik bir eşik
    hesaplar. Böylece sessiz bölümlerde bile click artefaktları
    etkili biçimde bastırılır; yüksek enerjili bölümlerde ise
    gereksiz müdahale olmaz.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    win_samples = max(1, int(_DECLICK_WINDOW_MS * sample_rate / 1000.0))

    # Yerel RMS: her örnek için ±win_samples yarıçaplı pencere
    x_sq = x ** 2
    kernel = np.ones(win_samples, dtype=np.float32) / win_samples
    local_rms = np.sqrt(np.convolve(x_sq, kernel, mode="same").clip(min=0.0))

    # Adaptif eşik: taban sabit * yerel RMS ölçeği
    global_rms = float(np.sqrt(np.mean(x_sq))) if x.size else 1e-7
    if global_rms < 1e-7:
        return x

    # Eşik = taban * (yerel RMS / global RMS) ile ölçekle, en az taban kadar
    adaptive_thresh = np.clip(
        _DECLICK_BASE_THRESH * (local_rms / (global_rms + 1e-7)),
        _DECLICK_BASE_THRESH * 0.5,
        _DECLICK_BASE_THRESH * 4.0,
    ).astype(np.float32)

    # Birinci türev (discrete derivative)
    diff = np.concatenate([[0.0], np.abs(np.diff(x))]).astype(np.float32)

    # Click maske: türevin adaptif eşiği aştığı örnekler
    mask: np.ndarray = diff > adaptive_thresh
    if not np.any(mask):
        return x

    out = x.copy()
    click_idx = np.where(mask)[0]

    for ci in click_idx:
        # Her click çevresindeki 3 örneği lineer interpolasyonla yumuşat
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
    Knee parametreli yumuşak limiter.

    ``knee_db`` dB genişliğindeki geçiş bölgesinde sinyali kademeli
    olarak sıkıştırır; böylece hard-clip artefaktları oluşmaz.

    Parametreler
    ------------
    ceiling : float
        Çıkış maksimum genliği (0 < ceiling <= 1).
    knee_db : float
        Knee genişliği dB cinsinden. 0 = hard limit, büyük değer = daha yumuşak.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    knee_db = max(0.0, float(knee_db))
    ceiling = float(np.clip(ceiling, 0.01, 1.0))

    knee_lin = 10.0 ** (-knee_db / 20.0)  # Knee'nin doğrusal alt sınırı
    knee_start = ceiling * knee_lin        # Bu değerden itibaren sıkıştırma başlar

    abs_x = np.abs(x)
    out = x.copy()

    # Knee bölgesi: knee_start < |x| <= ceiling
    in_knee = (abs_x > knee_start) & (abs_x <= ceiling)
    if np.any(in_knee):
        t = (abs_x[in_knee] - knee_start) / max(ceiling - knee_start, 1e-7)
        # Smoothstep: 3t² - 2t³
        smooth = t * t * (3.0 - 2.0 * t)
        gain = 1.0 - smooth * (1.0 - ceiling / abs_x[in_knee].clip(min=1e-7))
        out[in_knee] = x[in_knee] * gain

    # Hard bölge: |x| > ceiling → tanh ile yuvarlama
    above = abs_x > ceiling
    if np.any(above):
        # tanh ile ceil değerinin üzerini kırp ama sert kesme olmadan
        scale = ceiling / abs_x[above].clip(min=1e-7)
        excess = abs_x[above] - ceiling
        soft_excess = np.tanh(excess / (ceiling + 1e-7)) * ceiling
        out[above] = np.sign(x[above]) * (ceiling + soft_excess * (1.0 - scale))
        out[above] = np.clip(out[above], -ceiling, ceiling)

    return out.astype(np.float32)


def _deess(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Sibilant azaltma (De-essing).

    Yüksek frekanslı /s/, /ş/ gibi sibilant seslerini 6.8 kHz merkez
    frekanslı dar bant bandpass filtresi ile tespit eder; aşırı enerji
    varsa o bandı seçici olarak zayıflatır.

    İşlem sırası:
      1. Sibilant bandını bandpass filtresiyle izole et.
      2. Geniş spektrum ile sibilant bandının RMS'ini karşılaştır.
      3. Sibilant bandı baskın ise azaltma katsayısını uygula.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 64:
        return x

    nyquist = sample_rate / 2.0
    center = _DESS_FREQ_HZ
    bw_half = _DESS_BANDWIDTH_HZ / 2.0

    low_f = max(center - bw_half, 500.0) / nyquist
    high_f = min(center + bw_half, nyquist * 0.98) / nyquist

    if low_f >= high_f or low_f <= 0:
        return x  # sample_rate çok düşük, de-essing atla

    try:
        sos_band = signal.butter(3, [low_f, high_f], btype="bandpass", output="sos")
        sibilant = signal.sosfiltfilt(sos_band, x).astype(np.float32)
    except Exception as exc:
        logger.warning("_deess bandpass başarısız: %s", exc)
        return x

    full_rms = float(np.sqrt(np.mean(x ** 2))) + 1e-7
    sib_rms = float(np.sqrt(np.mean(sibilant ** 2))) + 1e-7
    ratio = sib_rms / full_rms

    # Sibilant oranı 0.25'in üzerindeyse azaltmayı uygula
    if ratio < 0.25:
        return x

    reduction_lin = 10.0 ** (-_DESS_REDUCTION_DB / 20.0)
    # Azaltma miktarını orana göre ölçeklendir (aşırı bastırma önle)
    gain = 1.0 - (1.0 - reduction_lin) * np.clip((ratio - 0.25) / 0.75, 0.0, 1.0)

    # Sadece sibilant bandındaki enerjiyi zayıflat, geri kalanına dokunma
    return (x - sibilant + sibilant * float(gain)).astype(np.float32)


# ---------------------------------------------------------------------------
# Herkese açık pipeline fonksiyonu
# ---------------------------------------------------------------------------

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
) -> np.ndarray:
    """
    Tüm post-processing adımlarını sırayla uygular.

    Parametreler
    ------------
    audio : np.ndarray
        Mono float32 ses verisi.
    sample_rate : int
        Örnekleme frekansı (Hz).
    speech_band : bool
        55 Hz – 10.5 kHz bandpass filtresi uygulansın mı?
    declick : bool
        Adaptif declick uygulansın mı?
    soft_limit : bool
        Knee parametreli yumuşak limiter uygulansın mı?
    deess : bool
        De-essing filtresi uygulansın mı?
    knee_db : float
        Limiter knee genişliği (dB).
    ceiling : float
        Limiter çıkış tavanı (0–1).

    Döndürür
    --------
    np.ndarray
        İşlenmiş mono float32 ses verisi.
    """
    x = np.asarray(audio, dtype=np.float32)

    if speech_band:
        x = _speech_band_filter(x, sample_rate)
    if declick:
        x = _declick(x, sample_rate)
    if deess:
        x = _deess(x, sample_rate)
    if soft_limit:
        x = _soft_limit(x, ceiling=ceiling, knee_db=knee_db)

    return x
