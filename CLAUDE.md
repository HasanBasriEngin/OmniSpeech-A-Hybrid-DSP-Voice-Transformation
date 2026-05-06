# 🔊 OmniSpeech — CLAUDE.md (Tauri + React + Python Mimarisi)

> **Proje Adı:** OmniSpeech  
> **Ders:** CENG 384 - Intro. To Signal Processing  
> **Grup:** Group 18  
> **Üyeler:** Vural YILMAZ · Eren DÖNMEZ · Hasan Basri Engin · Emre Boz · İlker Tuğberk Evren  
> **Mimari Revizyon:** Tauri + React + Tailwind CSS + Python (PyTorch) Masaüstü Mimarisi  
> **Tarih:** 29.3.2026

---

## 1. Proje Vizyonu ve Kapsamı

**OmniSpeech**, giriş sesinin orijinal dilsel içeriğini bozmadan; duygu, cinsiyet, yaş ve konuşmacı kimliği gibi karakteristik özelliklerini değiştirmeyi amaçlayan bir yazılım platformudur. Bu revize mimaride GUI katmanı **Tauri + React + Tailwind CSS** ile yeniden inşa edilmiş; tüm DSP ve derin öğrenme çıkarım mantığı ise ayrı bir **Python (PyTorch) arka uç süreci** üzerinde koşmaktadır. İki katman, **Tauri'nin yerel IPC (inter-process communication) köprüsü** aracılığıyla haberleşir.

### 1.1 Temel Hedefler

| Hedef              | Açıklama                                                                    |
| ------------------ | --------------------------------------------------------------------------- |
| Duygu Dönüşümü     | Tarafsız konuşmayı üzgün, sinirli, heyecanlı, fısıltı veya sakin tona çevir |
| Cinsiyet / Yaş     | Formant ve pitch manipülasyonuyla demografik ses dönüşümü                   |
| Konuşmacı Klonlama | Az sayıda referans örnekten hedef ses kimliğini yeniden inşa et             |
| Şarkı Üretimi      | Konuşmayı MIDI girdisiyle senkronize şarkıya dönüştür                       |
| Dilsel Bütünlük    | Tüm dönüşümlerde orijinal kelimeleri ve anlamı koru                         |

### 1.2 Kapsam Sınırları

- **Kapsam içi:** Masaüstü yazılım (Windows 10/11, macOS), DSP algoritmalar, öğrenme tabanlı dönüşüm, modern web tabanlı GUI
- **Kapsam dışı:** Ticari ses asistanı, bulut işleme, mobil platform, gerçek zamanlı akış zorunluluğu

---

## 2. Sistem Mimarisi

Bu mimari, **üç ana katmana** bölünmüş bir **hibrit masaüstü uygulamasıdır:**

```
┌─────────────────────────────────────────────────────────────┐
│                  TAURI SHELL (Rust)                         │
│  • Yerel pencere yönetimi (WebView2 / WKWebView)            │
│  • Dosya sistemi erişimi (dialog, fs API)                   │
│  • IPC köprüsü: invoke() + Event channel                   │
│  • Python alt süreç yöneticisi (sidecar)                    │
├───────────────────────┬─────────────────────────────────────┤
│   FRONTEND KATMANI    │        BACKEND KATMANI              │
│  React 18 + Vite      │   Python 3.11 + PyTorch Sidecar     │
│  Tailwind CSS v4      │   FastAPI (IPC dinleyici)           │
│  Zustand (state)      │   Librosa / SciPy / NumPy           │
│  Recharts / D3        │   PyTorch + ONNX Runtime            │
│  Web Audio API        │   SoundFile / Pydub / PyAudio       │
└───────────────────────┴─────────────────────────────────────┘
```

### 2.1 Teknoloji Yığını (Tech Stack)

| Katman                  | Araç / Kütüphane           | Amaç                                                   |
| ----------------------- | -------------------------- | ------------------------------------------------------ |
| **Masaüstü Kabuk**      | Tauri 2.x (Rust)           | Yerel pencere, dosya erişimi, IPC yönetimi             |
| **Ön Uç Çerçevesi**     | React 18 + Vite            | Bileşen tabanlı kullanıcı arayüzü                      |
| **Stil**                | Tailwind CSS v4            | Utility-first, karanlık tema UI                        |
| **Durum Yönetimi**      | Zustand                    | Global uygulama state'i                                |
| **Veri Görselleştirme** | Recharts + D3.js           | Waveform, pitch grafiği, metrik kartlar                |
| **Ses Oynatma**         | Web Audio API              | Tarayıcı-içi playback ve gerçek zamanlı görselleştirme |
| **IPC İletişimi**       | Tauri `invoke()` + Event   | Frontend ↔ Backend komut/olay köprüsü                  |
| **Arka Uç Dili**        | Python 3.11+               | DSP ve ML işleme motoru                                |
| **HTTP/IPC Katmanı**    | FastAPI (localhost)        | Python tarafı IPC uç noktaları                         |
| **Matematik & DSP**     | NumPy / SciPy              | Matris operasyonları, matematiksel hesaplamalar        |
| **Ses İşleme**          | Librosa                    | Öznitelik çıkarımı, F0 tespiti, spektral analiz        |
| **Canlı Ses**           | PyAudio                    | Gerçek zamanlı mikrofon akışı                          |
| **Format Yönetimi**     | SoundFile / Pydub          | WAV, MP3, FLAC okuma/yazma                             |
| **Derin Öğrenme**       | PyTorch 2.x + ONNX Runtime | Duygu & konuşmacı modelleri (CPU optimize)             |
| **MIDI**                | pretty_midi                | Şarkı modu için melodi girişi                          |
| **Donanım**             | Standart CPU               | Harici DSP çipi veya GPU gerektirmez                   |
| **İşletim Sistemi**     | Windows 10/11, macOS       | Hedef platformlar                                      |

---

## 3. IPC (Inter-Process Communication) Mimarisi

Tauri + Python iletişimi iki yönlü bir kanal üzerinden yürür:

### 3.1 Komut Akışı (Frontend → Backend)

```
React Bileşeni
    │
    │  await invoke("convert_emotion", { audioPath, emotion })
    ▼
Tauri Rust Komutları  (src-tauri/src/commands.rs)
    │
    │  HTTP POST → http://127.0.0.1:8765/api/convert
    ▼
FastAPI Python Sidecar  (backend/server.py)
    │
    │  pipeline.run(task)
    ▼
DSP / PyTorch İşleme Motoru
    │
    │  JSON yanıt { outputPath, metrics }
    ▼
Tauri Rust → invoke() resolve
    │
    ▼
React State Güncellemesi (Zustand)
```

### 3.2 Olay Akışı (Backend → Frontend — İlerleme Bildirimleri)

```
Python Backend
    │
    │  POST /events  →  { event: "progress", payload: 0.65 }
    ▼
Tauri Event Emitter (app.emit_all)
    │
    │  listen("pipeline_progress", handler)
    ▼
React ProgressBar Bileşeni
```

### 3.3 Tauri Komut Tanımları (`src-tauri/src/commands.rs`)

```rust
#[tauri::command]
async fn convert_emotion(audio_path: String, emotion: String) -> Result<ConversionResult, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/api/convert/emotion")
        .json(&json!({ "audio_path": audio_path, "emotion": emotion }))
        .send().await?;
    Ok(resp.json::<ConversionResult>().await?)
}

#[tauri::command]
async fn convert_gender_age(audio_path: String, conversion_type: String) -> Result<ConversionResult, String> { ... }

#[tauri::command]
async fn clone_speaker(audio_path: String, reference_paths: Vec<String>) -> Result<ConversionResult, String> { ... }

#[tauri::command]
async fn speech_to_singing(audio_path: String, midi_path: String) -> Result<ConversionResult, String> { ... }

#[tauri::command]
async fn get_metrics(output_path: String, original_path: String) -> Result<MetricsResult, String> { ... }

#[tauri::command]
fn open_file_dialog() -> Option<String> {
    // Tauri dialog API ile dosya seçici
    ...
}
```

---

## 4. Modüler Tasarım ve Veri Akışı

Sistem, 5 ana modülden oluşan **hiyerarşik bir boru hattı (pipeline)** mimarisi izler. Python arka ucu bu hattı yönetir; React ön ucu yalnızca komut gönderir ve sonuçları görselleştirir.

### 4.1 Tam Veri Akış Diyagramı

```
╔══════════════════════════════════════════════════════╗
║            KULLANICI GİRİŞİ (React UI)               ║
║   Dosya sürükle-bırak  ·  Mikrofon Kaydı Düğmesi     ║
╚══════════════════════════════════════════════════════╝
                        │
              Tauri invoke("load_audio")
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║       GİRİŞ MODÜLÜ — Python (backend/core/input.py)  ║
║  • Otomatik sessizlik algılama (VAD)                  ║
║  • Ses segmentasyonu (max 5sn dilimler)               ║
║  • Genlik normalizasyonu                             ║
║  • Örnekleme: 16 kHz veya 22.05 kHz (mono)           ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║   ÖNİŞLEME MODÜLÜ — Python (backend/core/preproc.py) ║
║  • MFCC, mel-spektrogram, STFT çıkarımı              ║
║  • F0 (temel frekans) tespiti — librosa.pyin / CREPE ║
║  • Prosodik bileşen ayrıştırması                     ║
║  • Spektral zarf hesaplama                           ║
╚══════════════════════════════════════════════════════╝
                        │
            olay: "preprocessing_done"
            (waveform + F0 verisi React'e iletilir)
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║   DSP İŞLEME MOTORU ← Kullanıcı Modül Seçimi        ║
║   (Tauri komutuyla tetiklenir)                       ║
║                                                      ║
║  ┌─────────────┐  ┌─────────────┐                    ║
║  │  🎭 Duygu   │  │ 👤 Cinsiyet │                    ║
║  │  Dönüşümü   │  │   / Yaş     │                    ║
║  └─────────────┘  └─────────────┘                    ║
║  ┌─────────────┐  ┌─────────────┐                    ║
║  │ 🎤 Konuşmacı│  │ 🎵 Şarkı   │                    ║
║  │   Klonlama  │  │   Üretimi   │                    ║
║  └─────────────┘  └─────────────┘                    ║
╚══════════════════════════════════════════════════════╝
                        │
            olay: "pipeline_progress" (0.0 → 1.0)
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║      ÇIKIŞ MODÜLÜ — Python (backend/core/output.py)  ║
║  • Yüksek kaliteli waveform sentezi                  ║
║  • WAV / MP3 / FLAC dosya export                     ║
║  • Konuşmacı embedding export (.npy)                 ║
║  • Oturum dönüşüm logu (JSON)                        ║
╚══════════════════════════════════════════════════════╝
                        │
            Tauri invoke() resolve →
            { outputPath, waveformData, metrics }
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║       DEĞERLENDİRME MODÜLÜ — Python (backend/core/)  ║
║  • Gecikme ölçümü       → hedef < 500ms              ║
║  • İşlem süresi         → hedef < 2sn / 5sn segment  ║
║  • Anlaşılırlık skoru   → MOS >= 3.5                 ║
║  • Ses kalitesi         → PESQ >= 3.0                ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║          REACT GÖRSELLEŞTIRME KATMANI                ║
║  Recharts Waveform · D3 Pitch Grafiği                ║
║  Metrik Kartlar · Session Log Paneli                 ║
║  Web Audio API Playback                              ║
╚══════════════════════════════════════════════════════╝
```

---

## 5. Python Arka Uç Modülleri

### 5.1 Giriş Modülü (`backend/core/input.py`)

**Sorumluluk:** Ses verisini sisteme alır, temizler ve işlemeye hazır hale getirir.

```python
# backend/core/input.py

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """WAV / MP3 / FLAC yükler; mono, hedef örnekleme hızına dönüştürür."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def start_mic_stream(sr: int = 16000, chunk: int = 1024):
    """PyAudio ile gerçek zamanlı mikrofon akışı — Generator döndürür."""
    ...

def detect_silence(audio: np.ndarray, threshold_db: float = -40.0) -> list[tuple[int, int]]:
    """VAD: sessiz ve sesli bölge sınırlarını döndürür."""
    ...

def segment_audio(audio: np.ndarray, sr: int, max_duration: float = 5.0) -> list[np.ndarray]:
    """Uzun sesi en fazla max_duration saniyelik dilimlere böler."""
    ...

def normalize_amplitude(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS normalizasyonu — tutarlı sinyal gücü sağlar."""
    ...

def validate_format(file_path: str) -> bool:
    """Desteklenen format kontrolü (WAV, MP3, FLAC)."""
    ...
```

### 5.2 Ön İşleme Modülü (`backend/core/preprocessing.py`)

**Sorumluluk:** Ham ses sinyalini analiz eder, akustik öznitelikleri ve prosodik yapıyı çıkarır.

```python
# backend/core/preprocessing.py

import numpy as np
import librosa

def extract_features(audio: np.ndarray, sr: int) -> dict:
    """
    Döndürür:
      {
        "mfcc":           np.ndarray,   # (n_mfcc, T)
        "mel_spectrogram": np.ndarray,  # (n_mels, T)
        "stft":           np.ndarray,   # (1 + n_fft/2, T)  — karmaşık
        "chroma":         np.ndarray,   # (12, T)
      }
    """
    ...

def detect_f0(audio: np.ndarray, sr: int,
              fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray:
    """librosa.pyin ile sürekli F0 takibi — NaN sessizlik noktaları."""
    ...

def separate_prosodic_spectral(audio: np.ndarray,
                                sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Döndürür: (prosodic_envelope, spectral_envelope)
    Prosodik: F0 + enerji zarfı
    Spektral: Formant yapısı + VTLN bileşenleri
    """
    ...

def compute_energy_envelope(audio: np.ndarray, frame_length: int = 2048) -> np.ndarray:
    """Kare ortalama güç (RMS) enerji zarfı."""
    ...

def to_serializable(features: dict) -> dict:
    """np.ndarray → list dönüşümü (JSON serileştirme için)."""
    return {k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in features.items()}
```

### 5.3 DSP İşleme Motoru

#### 5.3.1 Duygu Dönüşümü (`backend/modules/emotion_conversion.py`)

```python
# backend/modules/emotion_conversion.py

import numpy as np
import librosa
import torch

EMOTION_PROFILES = {
    "sad":       {"pitch_shift": -2.0, "rate": 0.85, "energy_scale": 0.70, "spectral_tilt": -2.0},
    "angry":     {"pitch_shift": +1.5, "rate": 1.15, "energy_scale": 1.40, "spectral_tilt": +1.5},
    "excited":   {"pitch_shift": +3.0, "rate": 1.25, "energy_scale": 1.30, "spectral_tilt": +2.0},
    "whispered": {"pitch_shift":  0.0, "rate": 0.90, "energy_scale": 0.40, "spectral_tilt": -3.0},
    "calm":      {"pitch_shift": -0.5, "rate": 0.95, "energy_scale": 0.85, "spectral_tilt": -0.5},
}

def convert_emotion(audio: np.ndarray, sr: int,
                    target_emotion: str,
                    progress_cb=None) -> np.ndarray:
    """
    Pipeline:
      1. F0 kontur kaydırma
      2. Konuşma hızı ayarı (WSOLA)
      3. Enerji zarfı ölçekleme
      4. Spektral eğim uygulama
      5. PyTorch duygu modeli ince ayarı
    progress_cb: 0.0–1.0 aralığında ilerleme bildirimi için callback.
    """
    ...

def shift_f0_contour(f0: np.ndarray, shift_st: float,
                     modulate: bool = True) -> np.ndarray:
    """Yarı ton cinsinden pitch kaydırma + isteğe bağlı jitter modülasyonu."""
    ...

def adjust_speech_rate(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """WSOLA tabanlı konuşma hızı değiştirme (pitch değişmez)."""
    ...

def modify_energy_envelope(audio: np.ndarray, scale: float) -> np.ndarray:
    ...

def apply_spectral_tilt(audio: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    ...
```

#### 5.3.2 Cinsiyet ve Yaş Dönüşümü (`backend/modules/gender_age_conversion.py`)

```python
# backend/modules/gender_age_conversion.py

import numpy as np

CONVERSION_MAP = {
    "male_to_female":   {"formant_ratio": 1.18, "pitch_shift": +3.5, "vtl_factor": 0.85},
    "female_to_male":   {"formant_ratio": 0.85, "pitch_shift": -3.5, "vtl_factor": 1.18},
    "adult_to_child":   {"formant_ratio": 1.35, "pitch_shift": +5.0, "vtl_factor": 0.74},
    "adult_to_elderly": {"formant_ratio": 0.92, "pitch_shift": -1.5, "vtl_factor": 1.05},
    "child_to_adult":   {"formant_ratio": 0.74, "pitch_shift": -5.0, "vtl_factor": 1.35},
}

def convert_gender_age(audio: np.ndarray, sr: int,
                       conversion_type: str,
                       progress_cb=None) -> np.ndarray:
    """
    Pipeline:
      1. Formant tespiti (LPC analizi)
      2. Formant frekansı oransal kaydırma
      3. VTLN warp uygulama
      4. Pitch kaydırma (PSOLA)
      5. Spektral zarf bükme
    """
    ...

def shift_formants(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    ...

def apply_vtln(audio: np.ndarray, sr: int, warp_factor: float) -> np.ndarray:
    ...

def warp_spectral_envelope(envelope: np.ndarray, warp: float) -> np.ndarray:
    ...
```

#### 5.3.3 Konuşmacı Dönüşümü ve Ses Klonlama (`backend/modules/speaker_conversion.py`)

```python
# backend/modules/speaker_conversion.py

import numpy as np
import torch

def extract_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    d-vector / x-vector tabanlı konuşmacı kimlik vektörü.
    PyTorch modeli: models/speaker_encoder.pt (CPU optimize).
    """
    ...

def convert_speaker(audio: np.ndarray, sr: int,
                    target_embedding: np.ndarray,
                    progress_cb=None) -> np.ndarray:
    ...

def clone_voice(audio: np.ndarray, sr: int,
                reference_samples: list[np.ndarray],
                progress_cb=None) -> np.ndarray:
    """Az sayıda referans örnekten (≥3) ses klonlama."""
    ...

def separate_content_identity(audio: np.ndarray,
                               sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Döndürür: (content_repr, identity_repr)"""
    ...

def reconstruct_with_target(content: np.ndarray,
                             target_embedding: np.ndarray) -> np.ndarray:
    ...

def export_embedding(embedding: np.ndarray, path: str) -> None:
    """np.save ile .npy dosyasına kayıt."""
    np.save(path, embedding)
```

#### 5.3.4 Şarkı Sesi Üretimi (`backend/modules/singing_voice.py`)

```python
# backend/modules/singing_voice.py

import numpy as np
import pretty_midi

def speech_to_singing(audio: np.ndarray, sr: int,
                      melody_input,
                      input_type: str = "midi",
                      progress_cb=None) -> np.ndarray:
    """input_type: 'midi' | 'pitch_contour'"""
    ...

def align_to_melody(audio: np.ndarray, sr: int,
                    target_f0: np.ndarray) -> np.ndarray:
    ...

def align_rhythm(audio: np.ndarray, sr: int,
                 beat_times: list[float]) -> np.ndarray:
    ...

def apply_singing_timbre(audio: np.ndarray, sr: int) -> np.ndarray:
    ...

def load_midi_melody(midi_path: str) -> tuple[np.ndarray, list[float]]:
    """Döndürür: (f0_contour, beat_times)"""
    ...
```

### 5.4 Çıkış Modülü (`backend/core/output.py`)

```python
# backend/core/output.py

import numpy as np
import soundfile as sf
import json
from datetime import datetime

def synthesize_waveform(processed_audio: np.ndarray, sr: int) -> np.ndarray:
    """Son normalize + anti-aliasing filtre uygulanmış waveform."""
    ...

def export_audio(audio: np.ndarray, sr: int,
                 path: str, fmt: str = "wav") -> None:
    """WAV / MP3 / FLAC çıktısı — pydub aracılığıyla format dönüşümü."""
    ...

def save_session_log(log_entries: list[dict], path: str) -> None:
    """Oturum dönüşüm geçmişini JSON dosyasına yazar."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

def waveform_to_json(audio: np.ndarray,
                     downsample_factor: int = 100) -> list[float]:
    """React Recharts gösterimi için waveform verisini JSON-serileştirilebilir listeye dönüştürür."""
    return audio[::downsample_factor].tolist()
```

### 5.5 Değerlendirme Modülü (`backend/core/evaluation.py`)

```python
# backend/core/evaluation.py

import time
import numpy as np
from typing import Callable

def measure_latency(fn: Callable, *args) -> float:
    """ms cinsinden fonksiyon gecikme süresi."""
    start = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - start) * 1000

def measure_processing_time(fn: Callable, *args) -> float:
    """Saniye cinsinden toplam işlem süresi."""
    ...

def compute_intelligibility(original: np.ndarray,
                             processed: np.ndarray) -> float:
    """MOS tahmini — NISQA veya basit SNR tabanlı yaklaşım."""
    ...

def compute_audio_quality(original: np.ndarray,
                           processed: np.ndarray,
                           sr: int) -> float:
    """PESQ skoru — pesq kütüphanesi aracılığıyla."""
    ...

def compute_speaker_similarity(emb1: np.ndarray,
                                emb2: np.ndarray) -> float:
    """Kosinüs benzerliği."""
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2) / norm) if norm > 0 else 0.0

def build_metrics_report(latency_ms: float,
                          proc_time_s: float,
                          mos: float,
                          pesq: float,
                          cos_sim: float) -> dict:
    return {
        "latency_ms": round(latency_ms, 2),
        "processing_time_s": round(proc_time_s, 3),
        "mos": round(mos, 3),
        "pesq": round(pesq, 3),
        "speaker_similarity": round(cos_sim, 4),
        "targets_met": {
            "latency": latency_ms < 500,
            "processing": proc_time_s < 2.0,
            "mos": mos >= 3.5,
            "pesq": pesq >= 3.0,
        }
    }
```

### 5.6 FastAPI Sunucusu (`backend/server.py`)

```python
# backend/server.py

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="OmniSpeech Backend", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["tauri://localhost"], allow_methods=["*"])

class ConvertEmotionRequest(BaseModel):
    audio_path: str
    emotion: str

class ConvertGenderAgeRequest(BaseModel):
    audio_path: str
    conversion_type: str

class CloneSpeakerRequest(BaseModel):
    audio_path: str
    reference_paths: list[str]

class SpeechToSingingRequest(BaseModel):
    audio_path: str
    midi_path: str

class ConversionResult(BaseModel):
    output_path: str
    waveform_data: list[float]
    metrics: dict

@app.post("/api/convert/emotion", response_model=ConversionResult)
async def convert_emotion_endpoint(req: ConvertEmotionRequest): ...

@app.post("/api/convert/gender_age", response_model=ConversionResult)
async def convert_gender_age_endpoint(req: ConvertGenderAgeRequest): ...

@app.post("/api/convert/speaker", response_model=ConversionResult)
async def clone_speaker_endpoint(req: CloneSpeakerRequest): ...

@app.post("/api/convert/singing", response_model=ConversionResult)
async def speech_to_singing_endpoint(req: SpeechToSingingRequest): ...

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

---

## 6. React Ön Uç Mimarisi

### 6.1 Bileşen Hiyerarşisi

```
src/
├── App.tsx                        # Kök bileşen + Zustand sağlayıcısı
│
├── pages/
│   └── MainPage.tsx               # Tek sayfa uygulama ana düzeni
│
├── components/
│   ├── layout/
│   │   ├── TopBar.tsx             # Uygulama başlığı + tema geçiş
│   │   └── SidePanel.tsx         # Modül seçim paneli
│   │
│   ├── input/
│   │   ├── FileDropZone.tsx       # Sürükle-bırak / dosya seçici (Tauri dialog)
│   │   └── MicRecorder.tsx        # Web Audio API mikrofon kaydı
│   │
│   ├── controls/
│   │   ├── ModuleSelector.tsx     # Duygu / Cinsiyet-Yaş / Konuşmacı / Şarkı sekmeleri
│   │   ├── EmotionPanel.tsx       # Duygu seçici + parametre slider'ları
│   │   ├── GenderAgePanel.tsx     # Cinsiyet/Yaş dönüşüm seçenekleri
│   │   ├── SpeakerPanel.tsx       # Referans ses yükleme + klonlama
│   │   └── SingingPanel.tsx       # MIDI yükleme + müzikal ayarlar
│   │
│   ├── visualization/
│   │   ├── WaveformChart.tsx      # Recharts: orijinal vs. işlenmiş karşılaştırma
│   │   ├── PitchGraph.tsx         # D3.js: F0 kontur eğrisi
│   │   └── MetricCards.tsx        # MOS, PESQ, gecikme metrik kartları
│   │
│   ├── playback/
│   │   └── AudioPlayer.tsx        # Web Audio API: Play/Pause/Stop/Volume
│   │
│   ├── output/
│   │   ├── ExportPanel.tsx        # Format seçimi + export düğmesi
│   │   └── SessionLog.tsx         # Dönüşüm geçmişi listesi
│   │
│   └── feedback/
│       ├── ProgressBar.tsx        # Tauri event ile senkronize animasyonlu bar
│       └── ErrorToast.tsx         # Hata ve uyarı bildirimleri
│
├── store/
│   └── useAppStore.ts             # Zustand global state
│
├── hooks/
│   ├── useTauriInvoke.ts          # invoke() sarmalayıcı (tip güvenli)
│   ├── useAudioPlayer.ts          # Web Audio API oynatma mantığı
│   └── usePipelineEvents.ts       # Tauri event dinleyici (ilerleme)
│
├── lib/
│   ├── tauriCommands.ts           # Tüm invoke() çağrıları (merkezi)
│   └── audioUtils.ts             # Waveform dönüşüm yardımcıları
│
└── types/
    └── index.ts                   # Paylaşılan TypeScript tipleri
```

### 6.2 Zustand Global Store (`src/store/useAppStore.ts`)

```typescript
// src/store/useAppStore.ts

import { create } from "zustand";

type ActiveModule = "emotion" | "gender_age" | "speaker" | "singing";
type ProcessingStatus = "idle" | "loading" | "processing" | "done" | "error";

interface AppState {
  // Dosya durumu
  inputFilePath: string | null;
  outputFilePath: string | null;
  originalWaveform: number[];
  processedWaveform: number[];

  // Modül seçimi
  activeModule: ActiveModule;
  selectedEmotion: string;
  selectedConversionType: string;
  referenceFilePaths: string[];
  midiFilePath: string | null;

  // İşlem durumu
  status: ProcessingStatus;
  progress: number; // 0.0 – 1.0
  errorMessage: string | null;

  // Metrikler
  metrics: Record<string, number | boolean> | null;

  // Oturum logu
  sessionLog: ConversionLogEntry[];

  // Eylemler
  setInputFile: (path: string) => void;
  setActiveModule: (module: ActiveModule) => void;
  setProgress: (value: number) => void;
  setMetrics: (m: Record<string, number | boolean>) => void;
  appendLog: (entry: ConversionLogEntry) => void;
  reset: () => void;
}

interface ConversionLogEntry {
  id: string;
  timestamp: string;
  module: ActiveModule;
  params: Record<string, unknown>;
  outputPath: string;
  metrics: Record<string, number | boolean>;
}

export const useAppStore = create<AppState>((set) => ({
  inputFilePath: null,
  outputFilePath: null,
  originalWaveform: [],
  processedWaveform: [],
  activeModule: "emotion",
  selectedEmotion: "calm",
  selectedConversionType: "male_to_female",
  referenceFilePaths: [],
  midiFilePath: null,
  status: "idle",
  progress: 0,
  errorMessage: null,
  metrics: null,
  sessionLog: [],

  setInputFile: (path) => set({ inputFilePath: path, status: "loading" }),
  setActiveModule: (module) => set({ activeModule: module }),
  setProgress: (value) => set({ progress: value }),
  setMetrics: (m) => set({ metrics: m }),
  appendLog: (entry) => set((s) => ({ sessionLog: [...s.sessionLog, entry] })),
  reset: () =>
    set({
      outputFilePath: null,
      processedWaveform: [],
      status: "idle",
      progress: 0,
      errorMessage: null,
      metrics: null,
    }),
}));
```

### 6.3 Tauri Komut Sarmalayıcıları (`src/lib/tauriCommands.ts`)

```typescript
// src/lib/tauriCommands.ts

import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export interface ConversionResult {
  outputPath: string;
  waveformData: number[];
  metrics: Record<string, number | boolean>;
}

export const tauriCommands = {
  openFileDialog: async (): Promise<string | null> => {
    return await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac"] }],
    });
  },

  convertEmotion: (audioPath: string, emotion: string) =>
    invoke<ConversionResult>("convert_emotion", { audioPath, emotion }),

  convertGenderAge: (audioPath: string, conversionType: string) =>
    invoke<ConversionResult>("convert_gender_age", {
      audioPath,
      conversionType,
    }),

  cloneSpeaker: (audioPath: string, referencePaths: string[]) =>
    invoke<ConversionResult>("clone_speaker", { audioPath, referencePaths }),

  speechToSinging: (audioPath: string, midiPath: string) =>
    invoke<ConversionResult>("speech_to_singing", { audioPath, midiPath }),

  getMetrics: (outputPath: string, originalPath: string) =>
    invoke<Record<string, number | boolean>>("get_metrics", {
      outputPath,
      originalPath,
    }),
};
```

### 6.4 UX — 3 Adım Kuralı (NFR 8)

```
ADIM 1: Ses Yükle
  └── FileDropZone: Dosya sürükle-bırak  |  MicRecorder: Kayıt başlat

ADIM 2: Modül Seç + Parametreleri Ayarla
  ├── 🎭 Duygu   → Duygu seçici dropdown + pitch/rate slider'ları
  ├── 👤 Cinsiyet/Yaş → Dönüşüm tipi seçici
  ├── 🎤 Konuşmacı → Referans ses(ler) yükle
  └── 🎵 Şarkı   → MIDI dosyası yükle

ADIM 3: "Dönüştür" → Sonuçları İncele & Export Et
  ├── ProgressBar (Tauri event ile canlı)
  ├── WaveformChart (karşılaştırmalı)
  ├── PitchGraph (F0 eğrisi)
  ├── MetricCards (MOS, PESQ, gecikme)
  ├── AudioPlayer (Play/Pause/Stop)
  └── ExportPanel (WAV / MP3 / FLAC / .npy embedding)
```

### 6.5 Tasarım Dili (Tailwind CSS)

```
Tema    : Koyu (Dark) — profesyonel ses mühendisliği estetiği
Palette : Mor-mavi accent   #6c63ff  (--color-accent)
          Teal vurgu        #22d3b0  (--color-teal)
          Koyu arka plan    #0d0f12  (--color-bg)
          Kart arka planı   #161b22  (--color-surface)
          Metin             #e2e8f0  (--color-text)

Tipografi: Sora         → UI metin (başlıklar, etiketler)
           JetBrains Mono → Değer gösterimi (metrikler, log)

Bileşen Stili:
  • Dashboard metrik kartları (glassmorphism kenarlık)
  • Neumorphism kontrol paneli slider'ları
  • Canlı waveform canvas (animasyonlu, gerçek zamanlı)
  • Animasyonlu ilerleme çubuğu (gradient fill)
```

---

## 7. Proje Yapısı

```
omnispeech/
│
├── package.json                   # Node bağımlılıkları (React, Vite, Tailwind)
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── README.md
├── CLAUDE.md                      # ← Bu dosya
│
├── src/                           # React ön uç
│   ├── App.tsx
│   ├── main.tsx
│   ├── pages/
│   ├── components/
│   ├── store/
│   ├── hooks/
│   ├── lib/
│   └── types/
│
├── src-tauri/                     # Tauri Rust kabuk
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs
│       ├── lib.rs
│       └── commands.rs            # Tauri IPC komutları
│
├── backend/                       # Python DSP + ML arka ucu
│   ├── server.py                  # FastAPI girdi noktası
│   ├── requirements.txt
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── input.py
│   │   ├── preprocessing.py
│   │   ├── output.py
│   │   └── evaluation.py
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── emotion_conversion.py
│   │   ├── gender_age_conversion.py
│   │   ├── speaker_conversion.py
│   │   └── singing_voice.py
│   │
│   ├── models/
│   │   ├── speaker_encoder.pt     # PyTorch konuşmacı encoder (CPU)
│   │   ├── emotion_model.onnx     # ONNX duygu modeli
│   │   └── singing_model.pt      # Şarkı sentez modeli
│   │
│   └── utils/
│       ├── audio_utils.py
│       └── logger.py
│
└── tests/
    ├── test_input.py
    ├── test_preprocessing.py
    ├── test_modules.py
    ├── test_performance.py
    └── test_api.py                # FastAPI endpoint testleri
```

---

## 8. Fonksiyonel Gereksinimler

### 8.1 Girdi İşleme (FR 1–6)

| ID   | Gereksinim                                                                             |
| ---- | -------------------------------------------------------------------------------------- |
| FR 1 | Sistem WAV, MP3 ve FLAC formatlarında ses kayıtlarını kabul etmelidir.                 |
| FR 2 | Sistem gerçek zamanlı mikrofon girişini destekleyebilir (opsiyonel — Web Audio API).   |
| FR 3 | Sistem 16 kHz ve 22.05 kHz örnekleme hızlarıyla mono ses sinyallerini işlemelidir.     |
| FR 4 | Sistem konuşma dışı segmentleri tanımlamak için otomatik sessizlik tespiti yapmalıdır. |
| FR 5 | Sistem uzun kayıtları işlenebilir birimlere bölmek için ses segmentasyonu yapmalıdır.  |
| FR 6 | Sistem tutarlı sinyal gücü için ses genlik normalizasyonu uygulamalıdır.               |

### 8.2 Ön İşleme (FR 7–9)

| ID   | Gereksinim                                                                                                    |
| ---- | ------------------------------------------------------------------------------------------------------------- |
| FR 7 | Sistem girdi sinyalinden akustik özellikler (MFCC, mel-spektrogram, STFT) çıkarmalıdır.                       |
| FR 8 | Sistem temel frekans (F0) tespiti ve sürekli takibini yapmalıdır; F0 verisi React PitchGraph'a iletilmelidir. |
| FR 9 | Sistem prosodik bileşenleri spektral bileşenlerden ayırmalıdır.                                               |

### 8.3 Duygu Dönüşümü (FR 10–15)

| ID    | Gereksinim                                                                                         |
| ----- | -------------------------------------------------------------------------------------------------- |
| FR 10 | Sistem tarafsız konuşmayı şu duygulara dönüştürmelidir: Üzgün, Sinirli, Heyecanlı, Fısıltı, Sakin. |
| FR 11 | Sistem F0 kaydırma ve modülasyon ile pitch konturunu değiştirmelidir.                              |
| FR 12 | Sistem konuşma hızı ve fonem süresini duygusal prozodi ile eşleştirmelidir.                        |
| FR 13 | Sistem konuşma sinyalinin enerji zarfını değiştirmelidir.                                          |
| FR 14 | Sistem spektral eğim ve formant kaydırma uygulayabilmelidir.                                       |
| FR 15 | Sistem duygusal prosodiyi modellemek için PyTorch sinir ağları kullanmalıdır.                      |

### 8.4 Cinsiyet ve Yaş Dönüşümü (FR 16–21)

| ID    | Gereksinim                                                                                   |
| ----- | -------------------------------------------------------------------------------------------- |
| FR 16 | Sistem Kadın ↔ Erkek ses dönüşümünü desteklemelidir.                                         |
| FR 17 | Sistem yaş dönüşümlerini desteklemelidir: Yetişkin-Çocuk, Yetişkin-Yaşlı, Çocuk-Yetişkin.    |
| FR 18 | Sistem farklı vokal kanal boyutlarını simüle etmek için formant frekanslarını ayarlamalıdır. |
| FR 19 | Sistem hedef cinsiyet/yaşa uygun pitch aralığını değiştirmelidir.                            |
| FR 20 | Sistem vokal kanal uzunluğu özelliklerini değiştirebilmelidir (VTLN).                        |
| FR 21 | Sistem spektral zarf çarpıtma ile sesin tını karakterini ayarlamalıdır.                      |

### 8.5 Konuşmacı Dönüşümü ve Klonlama (FR 22–28)

| ID    | Gereksinim                                                                                  |
| ----- | ------------------------------------------------------------------------------------------- |
| FR 22 | Sistem bir konuşmacının ses kimliğini başka bir konuşmacınınkine dönüştürmelidir.           |
| FR 23 | Sistem sınırlı referans ses verisiyle (≥3 örnek) hedef konuşmacı sesini klonlayabilmelidir. |
| FR 24 | Sistem kimlik dönüşümü sırasında orijinal dilsel içeriği korumalıdır.                       |
| FR 25 | Sistem ses kimliğini temsil eden benzersiz konuşmacı embeddingleri çıkarmalıdır.            |
| FR 26 | Sistem konuşmacı kimliği temsillerini içerik temsillerinden ayırmalıdır.                    |
| FR 27 | Sistem hedef konuşmacı embedding'ini kullanarak konuşmayı yeniden sentezlemelidir.          |
| FR 28 | Sistem çok-birden-bire ve bir-birden-çoğa konuşmacı kimliği eşleşmesini desteklemelidir.    |

### 8.6 Şarkı Sesi Üretimi (FR 29–33)

| ID    | Gereksinim                                                                          |
| ----- | ----------------------------------------------------------------------------------- |
| FR 29 | Sistem konuşmayı şarkı sesine dönüştürmelidir.                                      |
| FR 30 | Sistem çıkış pitch'ini tanımlı müzikal melodiye göre kontrol etmelidir.             |
| FR 31 | Sistem MIDI dosyaları veya pitch kontur verisi ile melodi girişini desteklemelidir. |
| FR 32 | Sistem ritim ve konuşma süresini müzikal zamanlama ile eşleştirmelidir.             |
| FR 33 | Sistem anlamlı bir şarkı tınısı üretmelidir.                                        |

### 8.7 Çıktı Üretimi (FR 34–36)

| ID    | Gereksinim                                                                             |
| ----- | -------------------------------------------------------------------------------------- |
| FR 34 | Sistem yüksek kaliteli dalga formu sesi üretmelidir.                                   |
| FR 35 | Sistem yapılandırılabilir çıktı dosya formatlarını (WAV / MP3 / FLAC) desteklemelidir. |
| FR 36 | Sistem işlenmiş ses için Web Audio API ile gerçek zamanlı oynatma sağlamalıdır.        |

### 8.8 Kullanıcı Arayüzü (FR 37–38)

| ID    | Gereksinim                                                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------------- |
| FR 37 | Sistem tüm işleme modülleriyle etkileşim için Tauri + React tabanlı bir GUI sağlamalıdır.                     |
| FR 38 | Sistem Recharts waveform grafiğiyle orijinal ve işlenmiş sinyallerin eş zamanlı önizlemesine izin vermelidir. |

### 8.9 Hata Yönetimi ve Veri Yönetimi (FR 39–42)

| ID    | Gereksinim                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------------- |
| FR 39 | Dosya bozuksa veya desteklenmeyen formattaysa React ErrorToast ile hata mesajı gösterilmelidir.      |
| FR 40 | Ses süresi çok kısaysa (<0.5sn) kullanıcı uyarı toast'ı ile bilgilendirilmelidir.                    |
| FR 41 | Çıkarılan konuşmacı embeddingleri .npy formatında export edilebilmelidir.                            |
| FR 42 | Oturum içinde yapılan tüm dönüşümlerin geçmiş logu Zustand store'da ve JSON dosyasında tutulmalıdır. |

---

## 9. Fonksiyonel Olmayan Gereksinimler

### 9.1 Performans

| Kod   | Gereksinim Tanımı | Hedef Metrik                                              |
| ----- | ----------------- | --------------------------------------------------------- |
| NFR 1 | İşlem Süresi      | < 5sn sesler için < **2 saniye**                          |
| NFR 2 | Gecikme (Latency) | Kısa segmentler için < **500 ms**                         |
| NFR 3 | CPU Optimizasyonu | Standart CPU'da çalışmalı; GPU gerektirmemeli             |
| NFR 4 | IPC Gecikme       | Tauri invoke() → Python yanıt < **100 ms** (ek yük dahil) |

### 9.2 Güvenilirlik

| Kod   | Gereksinim                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------- |
| NFR 5 | Ses Sadakati — dijital artifakt içermeyen yüksek kaliteli ses çıkışı                                    |
| NFR 6 | Sistem çökmesiz çalışmalıdır; Python sidecar çökmesi Tauri tarafından yakalanıp yeniden başlatılmalıdır |
| NFR 7 | Anlaşılırlık — dönüşüm sonrası dilsel içeriğin korunması                                                |

### 9.3 Kullanılabilirlik

| Kod    | Gereksinim                                                                  |
| ------ | --------------------------------------------------------------------------- |
| NFR 8  | Ses yükleme, parametre ayarı ve sonuç alma için sezgisel arayüz             |
| NFR 9  | Ses dönüşümü en fazla **3 adımda** tamamlanabilmeli                         |
| NFR 10 | Yoğun işlemlerde görsel progress bar + Tauri event tabanlı anlık güncelleme |

### 9.4 Taşınabilirlik ve Uyumluluk

| Kod    | Gereksinim                                                                        |
| ------ | --------------------------------------------------------------------------------- |
| NFR 11 | Windows 10/11 ve macOS ile uyumlu (Tauri çapraz platform desteği)                 |
| NFR 12 | Çıktı dosyaları endüstri standardı formatlarda, üçüncü taraf oynatıcılarla uyumlu |
| NFR 13 | Uygulama tek bir .exe / .dmg paketine derlenebilmelidir (Tauri bundler)           |

---

## 10. Test ve Doğrulama Senaryoları

| Test Türü            | Senaryo                           | Beklenen Sonuç                                   |
| -------------------- | --------------------------------- | ------------------------------------------------ |
| **Giriş Doğrulama**  | WAV / MP3 / FLAC yükleme          | Başarılı yükleme ve normalizasyon                |
| **Giriş Doğrulama**  | Bozuk dosya yükleme               | React ErrorToast hata mesajı (FR 39)             |
| **Giriş Doğrulama**  | Desteklenmeyen format             | Hata mesajı + desteklenen format listesi         |
| **Giriş Doğrulama**  | Çok kısa ses (<0.5sn)             | Uyarı toast mesajı (FR 40)                       |
| **IPC Testi**        | invoke("convert_emotion") çağrısı | < 100ms ek yük, doğru JSON yanıtı                |
| **IPC Testi**        | Python sidecar çökmesi            | Tauri otomatik yeniden başlatma (NFR 6)          |
| **Algoritma Testi**  | Duygu dönüşümü anlaşılırlık       | Dilsel içerik korunmalı (NFR 7)                  |
| **Algoritma Testi**  | Cinsiyet dönüşümü kalitesi        | Artefaktsız yüksek ses kalitesi (NFR 5)          |
| **Algoritma Testi**  | Konuşmacı klonlama benzerliği     | Cos. sim. >= 0.85                                |
| **Performans Testi** | <5sn segment işlem süresi         | < 2 saniye (NFR 1)                               |
| **Performans Testi** | Kısa segment gecikmesi            | < 500 ms (NFR 2)                                 |
| **Performans Testi** | Çökme testi (büyük girdi)         | Sistem çökmemeli (NFR 6)                         |
| **UI Testi**         | 3 adım kullanım akışı             | Kullanıcı 3 adımda dönüşüm yapabilmeli (NFR 9)   |
| **UI Testi**         | WaveformChart güncelleme          | Dönüşüm sonrası karşılaştırmalı grafik gösterimi |

```bash
# Python arka uç testleri
pytest backend/tests/ -v

# Sadece performans testleri
pytest backend/tests/test_performance.py -v

# FastAPI endpoint testleri
pytest backend/tests/test_api.py -v

# Kapsam raporu
pytest backend/tests/ --cov=backend/core --cov=backend/modules --cov-report=html

# React ön uç testleri (Vitest)
npm run test

# Tauri entegrasyon testi
cargo test --manifest-path src-tauri/Cargo.toml
```

---

## 11. Değerlendirme Metrikleri

| Metrik               | Amaç                          | Hedef Değer       |
| -------------------- | ----------------------------- | ----------------- |
| IPC Gecikmesi        | Tauri invoke() → Python yanıt | < 100 ms          |
| İşlem Gecikmesi      | Kısa segment yanıt süresi     | < 500 ms          |
| İşlem Süresi         | <5sn segment toplam süre      | < 2 saniye        |
| Anlaşılırlık (MOS)   | Dilsel içerik korunumu        | >= 3.5 / 5.0      |
| Ses Kalitesi (PESQ)  | Artefakt yokluğu              | >= 3.0            |
| Konuşmacı Benzerliği | Klonlama başarısı             | Cos. sim. >= 0.85 |
| Çökme Oranı          | Sistem kararlılığı            | %0 çökme          |
| UI Yanıt Süresi      | React render gecikmesi        | < 16 ms (60 FPS)  |

---

## 12. Kurulum ve Çalıştırma

### 12.1 Ön Koşullar

```bash
# Node.js 20+ ve Rust toolchain gereklidir
node --version   # >= 20.0
rustc --version  # >= 1.77
python --version # >= 3.11
```

### 12.2 Bağımlılık Kurulumu

```bash
# Depoyu klonla
git clone https://github.com/group18/omnispeech.git
cd omnispeech

# Node bağımlılıkları (React + Tauri)
npm install

# Python arka uç bağımlılıkları
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 12.3 Geliştirme Modu

```bash
# Python sidecar'ı başlat (terminal 1)
cd backend && python server.py

# Tauri geliştirme sunucusu (terminal 2)
npm run tauri dev
```

### 12.4 Üretim Derlemesi

```bash
npm run tauri build
# Çıktı: src-tauri/target/release/bundle/
#   Windows → .msi / .exe
#   macOS   → .dmg / .app
```

### 12.5 `requirements.txt` (Python Arka Ucu)

```
numpy>=1.26
scipy>=1.11
librosa>=0.10
pyaudio>=0.2.14
soundfile>=0.12
pydub>=0.25
torch>=2.2
onnxruntime>=1.17
pretty_midi>=0.2
fastapi>=0.111
uvicorn[standard]>=0.29
pydantic>=2.7
pesq>=0.0.4
httpx>=0.27
```

---

## 13. Sistem Kısıtlamaları

| Kısıt Türü            | Açıklama                                                           |
| --------------------- | ------------------------------------------------------------------ |
| **Donanım**           | Yalnızca standart CPU; harici DSP çipi veya GPU gerektirmez        |
| **Lisans**            | Yalnızca açık kaynaklı araçlar (ticari lisanssız)                  |
| **Kapsam**            | Akademik ve deneysel kullanım; ticari ses asistanı değil           |
| **Platform**          | Windows 10/11 ve macOS masaüstü ortamı (Tauri destekli)            |
| **Gerçek Zamanlılık** | Tam gerçek zamanlı işleme zorunluluğu yoktur                       |
| **IPC Güvenliği**     | Python sidecar yalnızca localhost:8765'te dinler; dış ağa kapalı   |
| **Tarayıcı Motoru**   | Tauri WebView2 (Windows) / WKWebView (macOS); Chromium gerektirmez |

---

## 14. Görev Dağılımı

| Üye                 | Sorumluluk Alanı                                                         |
| ------------------- | ------------------------------------------------------------------------ |
| Vural YILMAZ        | Input Modülü (Python) + Değerlendirme Modülü + FastAPI endpoint testleri |
| Eren DÖNMEZ         | Ön İşleme (Python) + F0 Tespiti + PitchGraph (React/D3)                  |
| Hasan Basri Engin   | Duygu Dönüşümü + Cinsiyet/Yaş Dönüşümü (Python)                          |
| Emre Boz            | Konuşmacı Dönüşümü + Klonlama (Python + PyTorch)                         |
| İlker Tuğberk Evren | Şarkı Sesi Üretimi (Python) + Tauri IPC + React UI Tasarımı              |

---

_Bu belge, OmniSpeech CLAUDE.md (PyQt6 mimarisi) temel alınarak **Tauri + React + Tailwind CSS + Python (PyTorch)** yığınına uyarlanmıştır. Proje adı: **OmniSpeech** — Group 18._
