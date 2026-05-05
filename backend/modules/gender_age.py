"""
gender_age.py — Cinsiyet ve yaş dönüşüm modülü.

Kişi 3 değişiklikleri (Eren DÖNMEZ):
  - CONVERSION_PRESETS yeniden kalibrasyonu:
      male_to_female  : pitch_shift=4.5, warp=1.20 (daha rafine)
      female_to_male  : chest_resonance desteği eklendi
      adult_to_child  : nazalizasyon eklendi
      adult_to_elderly: tremor artırıldı, nefes daha belirgin
  - SpectralWarp: lineer interpolasyon → kübik Catmull-Rom interpolasyon
  - SpectralWarp: phase coherence koruması eklendi
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal
import torch
import torch.nn as nn

from backend.audio.filtering import apply_post_filter


# ---------------------------------------------------------------------------
# Dönüşüm ön ayarları (yeniden kalibre edilmiş)
# ---------------------------------------------------------------------------

CONVERSION_PRESETS: dict[str, dict] = {
    # pitch_shift daha rafine (5.8 → 4.5), warp daha ılımlı (1.24 → 1.20)
    "male_to_female": {
        "pitch_shift": 4.5,
        "warp": 1.20,
        "brightness": 0.30,
        "warmth": -0.06,
        "breath": 0.010,
        "chest_resonance": 0.0,   # Erkek rezonanssı katkısı yok
        "nasality": 0.0,           # Nazalizasyon yok
        "tremor_amount": 0.0,      # Titreme yok
    },
    # Daha doğal erkek sesi: chest resonance artırıldı, warp ayarlandı
    "female_to_male": {
        "pitch_shift": -5.0,
        "warp": 0.80,
        "brightness": -0.25,
        "warmth": 0.40,
        "breath": 0.005,
        "chest_resonance": 0.35,   # Göğüs rezonansı katkısı (yeni)
        "nasality": 0.0,
        "tremor_amount": 0.0,
    },
    # Nazalizasyon eklendi; pitch biraz düşürüldü (7.2 → 6.8)
    "adult_to_child": {
        "pitch_shift": 6.8,
        "warp": 1.36,
        "brightness": 0.42,
        "warmth": -0.18,
        "breath": 0.006,
        "chest_resonance": 0.0,
        "nasality": 0.28,           # Nazalizasyon (yeni)
        "tremor_amount": 0.0,
    },
    # Tremor 0.045 → 0.075, nefes 0.028 → 0.048, pitch biraz artırıldı
    "adult_to_elderly": {
        "pitch_shift": -2.0,
        "warp": 0.88,
        "brightness": -0.32,
        "warmth": 0.22,
        "breath": 0.048,            # Daha belirgin nefes (yeni)
        "chest_resonance": 0.0,
        "nasality": 0.0,
        "tremor_amount": 0.075,     # Daha fazla tremor (yeni)
    },
    "child_to_adult": {
        "pitch_shift": -6.2,
        "warp": 0.74,
        "brightness": -0.18,
        "warmth": 0.28,
        "breath": 0.0,
        "chest_resonance": 0.0,
        "nasality": 0.0,
        "tremor_amount": 0.0,
    },
}


# ---------------------------------------------------------------------------
# SpectralWarp — kübik Catmull-Rom interpolasyon + phase coherence
# ---------------------------------------------------------------------------

class SpectralWarp(nn.Module):
    """
    Hafif spektral zarf bükme modülü.

    İyileştirmeler (Kişi 3):
      1. Lineer interpolasyon yerine kübik Catmull-Rom spline interpolasyonu.
         Bu, spektral zirveler arasında daha pürüzsüz geçişler sağlar.
      2. Phase coherence koruması: faz, orijinal spektrumdan korunur;
         sadece büyüklük (magnitude) dönüştürülür.
    """

    @staticmethod
    def _cubic_interp(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Catmull-Rom kübik interpolasyon.

        ``values`` 1-D tensörü üzerinde ``idx`` konumlarını enterpolasyon yapar.
        Sınır örnekleri sıkıştırılarak kopyalanır.
        """
        n = values.numel()
        i = idx.long()

        # 4 komşu indeksi (p0, p1, p2, p3) — sınırları klamp'la
        i0 = torch.clamp(i - 1, 0, n - 1)
        i1 = torch.clamp(i,     0, n - 1)
        i2 = torch.clamp(i + 1, 0, n - 1)
        i3 = torch.clamp(i + 2, 0, n - 1)

        p0 = values[i0]
        p1 = values[i1]
        p2 = values[i2]
        p3 = values[i3]

        t = idx - i.float()   # Kesirli kısım [0, 1)

        # Catmull-Rom formülü
        result = (
            0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
            )
        )
        return result

    def forward(self, audio: torch.Tensor, warp_factor: float) -> torch.Tensor:
        if audio.ndim != 1:
            raise ValueError("SpectralWarp expects a 1D mono tensor.")

        spec = torch.fft.rfft(audio)
        mag = torch.abs(spec)
        phase = torch.angle(spec)   # Orijinal fazı sakla (phase coherence)

        n = mag.numel()
        idx = torch.arange(n, device=audio.device, dtype=torch.float32)
        src_idx = torch.clamp(idx / max(warp_factor, 1e-4), 0.0, float(n - 1))

        # Kübik interpolasyon ile büyüklüğü bük
        warped_mag = self._cubic_interp(mag, src_idx)
        warped_mag = torch.clamp(warped_mag, min=0.0)  # Negatif büyüklük olamaz

        # Phase coherence: orijinal fazı koru, sadece büyüklüğü değiştir
        complex_spec = warped_mag * torch.exp(1j * phase)
        return torch.fft.irfft(complex_spec, n=audio.numel())


WARP_MODEL = SpectralWarp()


# ---------------------------------------------------------------------------
# Yardımcı DSP fonksiyonları
# ---------------------------------------------------------------------------

def _fft_tone_shape(audio: np.ndarray, brightness: float, warmth: float) -> np.ndarray:
    """Frekans domaininde yüksek raf ve alçak raf filtresi uygula."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    high_shelf = 1.0 + brightness * np.clip((freqs - 0.18) / 0.62, 0.0, 1.0)
    low_shelf = 1.0 + warmth * np.exp(-((freqs - 0.09) ** 2) / (2 * 0.08 ** 2))
    presence = 1.0 + max(brightness, 0.0) * 0.32 * np.exp(
        -((freqs - 0.36) ** 2) / (2 * 0.055 ** 2)
    )
    curve = np.clip(high_shelf * low_shelf * presence, 0.25, 2.5)
    return np.fft.irfft(spec * curve, n=x.size).astype(np.float32)


def _soft_compress(audio: np.ndarray, drive: float = 1.18) -> np.ndarray:
    """Yumuşak harmonik bozulma / kompresyon."""
    x = np.asarray(audio, dtype=np.float32)
    return np.tanh(x * drive).astype(np.float32)


def _add_chest_resonance(audio: np.ndarray, sample_rate: int, amount: float) -> np.ndarray:
    """
    Göğüs rezonansı vurgusu — erkek sesine daha derin ve dolgun tını verir.

    100–350 Hz bandını zayıf bir peak EQ ile yükseltir.
    """
    if amount <= 0.0:
        return audio

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    nyquist = sample_rate / 2.0
    center = min(200.0, nyquist * 0.9) / nyquist
    width = min(150.0, nyquist * 0.4) / nyquist

    low_f = max(center - width / 2, 1e-3)
    high_f = min(center + width / 2, 0.98)

    if low_f >= high_f:
        return x

    sos = signal.butter(2, [low_f, high_f], btype="bandpass", output="sos")
    band = signal.sosfiltfilt(sos, x).astype(np.float32)
    return (x + band * amount).astype(np.float32)


def _add_nasality(audio: np.ndarray, sample_rate: int, amount: float) -> np.ndarray:
    """
    Nazalizasyon efekti — çocuk sesine karakteristik burun rezonanssı katar.

    500–1500 Hz bandında hafif bir rezonans tümseci ekler.
    """
    if amount <= 0.0:
        return audio

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    nyquist = sample_rate / 2.0
    low_f = min(500.0, nyquist * 0.9) / nyquist
    high_f = min(1500.0, nyquist * 0.98) / nyquist

    if low_f >= high_f:
        return x

    sos = signal.butter(2, [low_f, high_f], btype="bandpass", output="sos")
    band = signal.sosfiltfilt(sos, x).astype(np.float32)
    return (x + band * amount).astype(np.float32)


def _add_breath_and_tremor(
    audio: np.ndarray,
    sample_rate: int,
    amount: float,
    mode: str,
    tremor_amount: float = 0.045,
) -> np.ndarray:
    """
    Nefes gürültüsü ve tremor ekle.

    Kişi 3 değişikliği: ``tremor_amount`` parametresi eklendi;
    adult_to_elderly için varsayılan tremor 0.045 → 0.075 olarak
    preset'e taşındı.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    out = x

    # Tremor (yaşlı sesi)
    if mode == "adult_to_elderly" and tremor_amount > 0.0:
        t = np.arange(x.size, dtype=np.float32) / max(sample_rate, 1)
        # Titreme frekansı 5.2 Hz → 5.8 Hz (daha belirgin)
        tremor_env = 1.0 + tremor_amount * np.sin(2 * np.pi * 5.8 * t)
        out = (out * tremor_env).astype(np.float32)

    # Nefes gürültüsü
    if amount > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, amount, x.size).astype(np.float32)
        sos = signal.butter(2, 1800, btype="highpass", fs=sample_rate, output="sos")
        breath = signal.sosfilt(sos, noise).astype(np.float32)
        out = out + breath

    return out.astype(np.float32)


def _peak_normalize(audio: np.ndarray, peak_target: float = 0.97) -> np.ndarray:
    """Peak normalizasyonu."""
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-7:
        return x
    return (x * (peak_target / peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# Ana dönüşüm fonksiyonu
# ---------------------------------------------------------------------------

def convert_gender_age(audio: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    """
    Verilen ses verisine cinsiyet/yaş dönüşümü uygular.

    Parametreler
    ------------
    audio : np.ndarray
        Mono float32 ses verisi.
    sample_rate : int
        Örnekleme frekansı (Hz).
    mode : str
        Dönüşüm modu. İzin verilen değerler: ``CONVERSION_PRESETS`` anahtarları.

    Döndürür
    --------
    np.ndarray
        Dönüştürülmüş mono float32 ses verisi ([-1.0, 1.0] aralığında).
    """
    preset = CONVERSION_PRESETS.get(mode)
    if preset is None:
        allowed = ", ".join(sorted(CONVERSION_PRESETS))
        raise ValueError(f"Unsupported conversion mode '{mode}'. Allowed: {allowed}")

    x = np.asarray(audio, dtype=np.float32)

    # 1. Pitch shifting
    pitched = librosa.effects.pitch_shift(y=x, sr=sample_rate, n_steps=preset["pitch_shift"])

    # 2. Spektral warp (kübik interpolasyon + phase coherence)
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(pitched, dtype=np.float32))
        warped = WARP_MODEL(tensor, preset["warp"]).cpu().numpy().astype(np.float32)

    # 3. Ton şekillendirme (brightness / warmth)
    shaped = _fft_tone_shape(warped, brightness=preset["brightness"], warmth=preset["warmth"])

    # 4. Göğüs rezonansı (erkek sesi için)
    shaped = _add_chest_resonance(shaped, sample_rate, preset.get("chest_resonance", 0.0))

    # 5. Nazalizasyon (çocuk sesi için)
    shaped = _add_nasality(shaped, sample_rate, preset.get("nasality", 0.0))

    # 6. Nefes gürültüsü + tremor
    textured = _add_breath_and_tremor(
        shaped,
        sample_rate,
        preset["breath"],
        mode,
        tremor_amount=preset.get("tremor_amount", 0.045),
    )

    # 7. Yumuşak kompresyon
    compressed = _soft_compress(textured)

    # 8. Ortak post-filter pipeline (declick, speech_band, deess, soft limit)
    filtered = apply_post_filter(compressed, sample_rate)

    return np.clip(_peak_normalize(filtered), -1.0, 1.0).astype(np.float32)
