# OmniSpeech - A Hybrid DSP Voice Transformation System

Ankara Bilim University - CENG 384 Project

OmniSpeech, sesin dilsel içeriğini koruyarak duygu, cinsiyet/yaş, konuşmacı kimliği ve şarkı benzeri dönüşüm işlemleri yapan hibrit bir DSP + öğrenme tabanlı dönüşüm sistemidir.

## Kullanılan Teknolojiler

| Kategori | Teknolojiler |
|----------|-------------|
| **Core / DSP** | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white) ![Librosa](https://img.shields.io/badge/Librosa-Audio-blue?style=for-the-badge) |
| **Python Frontend** | ![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white) |
| **WPF Frontend (Opsiyonel)** | ![.NET](https://img.shields.io/badge/.NET_10-512BD4?style=for-the-badge&logo=dotnet&logoColor=white) ![C#](https://img.shields.io/badge/C%23-WPF-239120?style=for-the-badge&logo=csharp&logoColor=white) |
| **Audio I/O** | ![SoundFile](https://img.shields.io/badge/SoundFile-WAV%2FFLAC-4B5563?style=for-the-badge) ![PyDub](https://img.shields.io/badge/PyDub-MP3-4B5563?style=for-the-badge) ![PyAudio](https://img.shields.io/badge/PyAudio-Microphone-4B5563?style=for-the-badge) |
| **Model / Runtime** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white) |

## Gereksinimler

| Araç | Versiyon |
|------|----------|
| Python | 3.10+ |
| pip | güncel |
| .NET SDK (WPF için) | 10.0+ |
| Windows | 10/11 |

## Kurulum

### 1. Repo'yu klonla

```bash
git clone <repo-url>
cd OmniSpeech-A-Hybrid-DSP-Voice-Transformation
```

### 2. Python sanal ortamını oluştur ve aktif et

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. Uygulamayı başlat (Python / PyQt frontend)

```bash
python main.py
```

Alternatif tek komut (Windows):

```bash
run_omnispeech.bat
```

### 5. WPF frontend'i çalıştır (opsiyonel)

```bash
cd OmniSpeech_WPF
run_wpf_frontend.bat
```

## Temel Akış

1. Ses yükle (`WAV`, `MP3`, `FLAC`) veya MIC simülasyonu seç  
2. Modül seç (`Emotion`, `Gender/Age`, `Speaker`, `Singing`)  
3. Parametreleri ayarla (`Pitch`, `Speech Rate`, `Energy`)  
4. `Convert Audio` ile dönüşümü çalıştır  
5. Sonucu export et (`Audio`, `Embedding`, `Session Log`)

## Proje Yapısı

```text
main.py                         -> Python uygulama giriş noktası
requirements.txt                -> Python bağımlılıkları
run_omnispeech.bat              -> Python app launcher
CLAUDE.md                       -> Proje gereksinim ve mimari referansı

core/                           -> Input, preprocessing, output, evaluation
modules/                        -> Emotion, gender/age, speaker, singing dönüşüm modülleri
ui/                             -> PyQt frontend (views, controls, services, models, converters)
utils/                          -> Ortak yardımcı fonksiyonlar
tests/                          -> Temel test dosyaları

OmniSpeech_WPF/                 -> Opsiyonel WPF frontend
```

## Komutlar

```bash
python -m pytest -v                            # Testleri çalıştır
python main.py                                 # PyQt frontend başlat
run_omnispeech.bat                             # Otomatik venv + install + run
dotnet build OmniSpeech_WPF/OmniSpeech_WPF.csproj   # WPF derle
```

## Notlar

- MP3 export için sistemde uygun `ffmpeg` kurulumu gerekebilir (`pydub` için).
- Python Store alias sorunlarında, python.org üzerinden kurulum ve PATH ayarı önerilir.
- WPF frontend görsel prototip/arayüz amaçlıdır; ana DSP pipeline Python tarafındadır.
