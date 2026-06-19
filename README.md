# OmniSpeech Desktop (Tauri + React + Python)

OmniSpeech, Tauri masaustu kabugu uzerinde calisan; React tabanli arayuz ve Python/FastAPI ses donusum backend'ini birlestiren hibrit bir ses donusum uygulamasidir.

## Kullanilan Teknolojiler

| Kategori | Teknolojiler |
|----------|-------------|
| **Desktop** | ![Tauri](https://img.shields.io/badge/Tauri-FFC131?style=for-the-badge&logo=tauri&logoColor=black) ![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white) |
| **Frontend** | ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB) ![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white) |
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Librosa](https://img.shields.io/badge/Librosa-Audio-blue?style=for-the-badge) |
| **Ses I/O** | ![SoundDevice](https://img.shields.io/badge/SoundDevice-RealTime-1F6FEB?style=for-the-badge) ![SoundFile](https://img.shields.io/badge/SoundFile-WAV%2FFLAC-1F6FEB?style=for-the-badge) |

## Gereksinimler

| Arac | Versiyon |
|------|----------|
| Python | 3.10+ |
| Node.js | 20+ |
| Rust (cargo/rustup) | stable |
| Tauri CLI | 2.x (npm dependency ile gelir) |

Platform notlari:
- Windows: Visual Studio Build Tools (Desktop development with C++, MSVC v143, Windows 10/11 SDK)
- macOS: Xcode Command Line Tools

## Kurulum

### 1. Repo'yu klonla

```bash
git clone <repo-url>
cd OmniSpeech-A-Hybrid-DSP-Voice-Transformation
```

### 2. Windows tek komut kurulum ve calistirma

```bat
run_omnispeech.bat
```

Bu script su islemleri yapar:
- Python/Node/Rust kontrolu ve otomatik kurulum denemesi
- `.venv` olusturma ve `requirements.txt` kurulumu
- `npm install`
- MSVC linker kontrolu
- `npm run tauri dev` ile desktop baslatma

### 3. Manuel kurulum (Windows/macOS)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS:
source .venv/bin/activate

pip install -r requirements.txt
npm install
npm run tauri dev
```

### Opsiyonel AI kurulumu

RVC tabanli offline gender/age, FreeVC tabanli referans sesli konusmaci klonu,
Pedalboard post-filter, Noisereduce spectral-gate temizleme ve Parselmouth pitch
shift icin:

```bash
pip install -r requirements-ai.txt
```

Bu kurulum OpenCV spektrogram on-isleme katmanini da acar. Backend, sesi Librosa ile STFT spektrogramina cevirir, spektrogrami OpenCV ile 256x256 goruntu gibi resize/blur/normalize eder, orijinal faz ile tekrar waveform'a donusturur ve RVC/DSP donusumune bu hazirlanmis sesi verir. `cv2` yoksa pipeline calismaya devam eder ve metriklerde `opencv_spectrogram_applied=0` gorunur.

RVC paketleri Windows'ta Python 3.10 ortaminda daha sorunsuz kurulur:

```bash
py -3.10 -m venv .venv-rvc
.venv-rvc\Scripts\python -m pip install -r requirements-rvc.txt
```

Applio RVC motorunu denemek icin Applio reposunu yerel bir checkout olarak
`vendor/applio` altina koyun veya `OMNISPEECH_APPLIO_ROOT` ile yolunu verin:

```bash
git clone https://github.com/IAHispano/Applio vendor/applio
.venv-rvc\Scripts\python -m pip install -r requirements-applio.txt
set OMNISPEECH_RVC_ENGINE=applio
```

`OMNISPEECH_RVC_ENGINE=auto` varsayilan moddur. Bu mod once mevcut
`rvc-python` adaptoruyle dener; `rvc-python` kurulu degilse ve Applio checkout'u
bulunursa Applio motoruna duser. Belirli bir model icin `models/rvc/registry.json`
icinde `engine: "applio"` kullanarak dogrudan Applio secilebilir.

### Hugging Face RVC/FreeVC import

OmniSpeech, Hugging Face'ten secilmis RVC ve FreeVC varliklarini yerel
`models/hf/` altina indirebilir:

```bash
# Indirmeden once secilecek repo ve dosyalari goster
python -m backend.tools.import_hf_voice_assets --bundle all --dry-run

# RVC core, FreeVC 24 kHz ve WavLM Large varliklarini indir
python -m backend.tools.import_hf_voice_assets --bundle all
```

Secilen varsayilanlar:
- RVC icin `AEmotionStudio/rvc-models`: MIT lisansli RVC v2 core varliklari,
  48 kHz F0/RMVPE yolu.
- FreeVC icin `OlaWod/FreeVC`: FreeVC 24 kHz one-shot checkpoint ve calisma
  dosyalari.
- FreeVC content encoder icin `microsoft/wavlm-large`: Space'in kullandigi
  WavLM Large encoder.

OpenVoice veya FreeVC varliklari mevcutsa `speaker_clone` dosya donusumu, ilk
referans sesini hedef stil olarak kullanir. Siralama OpenVoice, FreeVC ve DSP
fallback seklindedir; model veya checkpoint eksikse mevcut hafif DSP/ML fallback
aynen calisir.

### OpenVoice entegrasyonu

OpenVoice opsiyoneldir ve uygulama OpenVoice kurulu degilken de calisir. Yerel
kurulum icin OpenVoice reposunu proje icine klonlayip checkpointleri ayni
dizindeki `checkpoints` klasorune yerlestirin:

```bash
git clone https://github.com/myshell-ai/OpenVoice.git vendor/OpenVoice
pip install -r vendor/OpenVoice/requirements.txt
```

Ardindan OpenVoice checkpoint zip dosyasini indirip acin ve converter
checkpointlerinin `vendor/OpenVoice/checkpoints/converter/config.json` ve
`vendor/OpenVoice/checkpoints/converter/checkpoint.pth` olarak gorundugunu
dogrulayin. Farkli konum kullaniyorsaniz:

```cmd
set OMNISPEECH_OPENVOICE_ROOT=vendor/OpenVoice
set OMNISPEECH_OPENVOICE_CHECKPOINTS_DIR=vendor/OpenVoice/checkpoints
```

OpenVoice hazir oldugunda `speaker_clone`, `/api/convert/voice-clone` ve yerel
referans WAV bulunan `celebrity` donusumleri tone-color converter'i kullanir.
Uygulama metriklerinde `openvoice_engine=1` gorunur. Checkpointleri otomatik
indirmek isterseniz `OMNISPEECH_OPENVOICE_AUTO_DOWNLOAD=1` verilebilir.

RVC modelleri git'e eklenmez. Lisansli veya riza alinmis yerel modelleri `models/rvc/<model_id>/<model_id>.pth` duzeninde yerlestirin ve `models/rvc/registry.json` ile mode eslestirmesi yapin. Ornek sema `models/rvc/registry.example.json` icindedir.
Hugging Face uzerindeki hazir RVC hedef ses modelleri kalite ve izin acisindan
cok degiskendir; bu yuzden importer varsayilan olarak yalnizca core varliklari
indirir. Hedef ses icin sadece lisansli veya riza alinmis `.pth/.index`
modellerini kullanin.

### Hibrit AI/DSP Pipeline Plani

Bu mimari uygulanabilir ve repo su an RVC-oncelikli sekilde hazirdir:

1. Girdi ses dosyasi Librosa/SoundFile ile mono float32 olarak yuklenir.
2. Duygu modulu Parselmouth varsa Praat pitch manipulation kullanir; yoksa DSP tabanli PSOLA/FFT fallback'e iner.
3. Gender-age ve lisansli profil yollarinda spektrogram OpenCV ile goruntu gibi islenir.
4. `models/rvc/registry.json` icinde uygun model varsa RVC inference calisir; yoksa ayni hazirlanmis ses hafif DSP fallback ile islenir.
5. Cikis post-filter, de-click, de-ess ve limiter zincirinden gecirilip WAV olarak kaydedilir.

OpenVoice tone-color converter ve FreeVC 24 kHz entegrasyonu speaker-clone akisi
icin opsiyonel olarak eklidir.
RVC ise hedef sese ozel yerel `.pth/.index` model registry'si ile calisir.

## Gelistirme Ortam Degiskenleri

| Degisken | Aciklama |
|----------|----------|
| `OMNISPEECH_FORCE_INSTALL=1` | Python ve npm bagimliliklarini zorla tekrar kurar |
| `OMNISPEECH_SETUP_ONLY=1` | Sadece kurulum yapar, uygulamayi acmaz |
| `OMNISPEECH_AUTO_INSTALL=1` | Python/Node/Rust otomatik kurulum denemelerini acik tutar |
| `OMNISPEECH_AUTO_INSTALL_MSVC=1` | Windows'ta MSVC Build Tools otomatik kurulum denemesini acik tutar |
| `OMNISPEECH_RVC_MODELS_DIR=models/rvc` | Yerel RVC registry ve model kok dizini |
| `OMNISPEECH_RVC_DEVICE=cpu` | RVC inference cihaz secimi |
| `OMNISPEECH_RVC_ENGINE=auto` | `auto`, `rvc-python` veya `applio` motor secimi |
| `OMNISPEECH_APPLIO_ROOT=vendor/applio` | Opsiyonel Applio checkout dizini |
| `OMNISPEECH_RVC_TEMP_DIR=.tmp/rvc` | RVC gecici WAV calisma dizini |
| `OMNISPEECH_FREEVC_ASSETS_DIR=models/hf/freevc-24` | Hugging Face'ten import edilen FreeVC Space dosyalari |
| `OMNISPEECH_FREEVC_DEVICE=cpu` | FreeVC inference cihaz secimi |
| `OMNISPEECH_WAVLM_MODEL=models/hf/wavlm-large` | FreeVC WavLM encoder yolu veya HF model id |
| `OMNISPEECH_FREEVC_TEMP_DIR=.tmp/freevc` | FreeVC gecici WAV calisma dizini |
| `OMNISPEECH_OPENVOICE_ROOT=vendor/OpenVoice` | Yerel OpenVoice repo checkout dizini |
| `OMNISPEECH_OPENVOICE_CHECKPOINTS_DIR=vendor/OpenVoice/checkpoints` | OpenVoice converter checkpoint dizini |
| `OMNISPEECH_OPENVOICE_DEVICE=cpu` | OpenVoice inference cihaz secimi |
| `OMNISPEECH_OPENVOICE_TEMP_DIR=.tmp/openvoice` | OpenVoice gecici WAV calisma dizini |
| `OMNISPEECH_OPENVOICE_AUTO_DOWNLOAD=1` | Converter checkpointleri yoksa otomatik indirmeyi acar |
| `OMNISPEECH_AI_PREPROCESS_TEMP_DIR=.tmp/ai_preprocess` | RVC oncesi OpenCV spektrogram on-isleme gecici dizini |

## Ozellikler

- Canli mikrofon veya dosya tabanli ses donusumu
- Cinsiyet/Yas donusumu
- Konusmaci klonlama
- Sarkiya donusturme (MIDI destekli akis)
- Sanal mikrofon cihaz listesi ve route islemleri
- Session log ve canli islem durumu

## Proje Yapisi

```text
backend/                  -> FastAPI tabanli ses donusum backend'i
backend/api/              -> Route ve schema tanimlari
backend/audio/            -> Ses okuma/yazma ve ozellik cikarimi
backend/modules/          -> Donusum modulleri (gender-age, clone, singing)
backend/pipeline/         -> Islem orkestrasyonu
backend/services/         -> Live session ve virtual mic servisleri
src/                      -> React uygulamasi (UI + state + Tauri bridge)
src-tauri/                -> Rust/Tauri desktop katmani
tests/                    -> Backend pipeline testleri
run_omnispeech.bat        -> Windows otomatik kurulum + calistirma scripti
requirements.txt          -> Python bagimliliklari
package.json              -> Node/Tauri bagimliliklari
```

## Komutlar

```bash
# Frontend gelistirme
npm run dev

# Desktop (Tauri) gelistirme
npm run tauri dev

# Frontend production build
npm run build

# Backend test
python -m pytest tests/test_backend_pipeline.py
```

## Notlar

- Uygulama masaustu (Tauri) olarak calisir. Gelistirme modunda Vite dev server arka planda `127.0.0.1:1420` kullanir; paketli surumde tarayici uygulamasi gibi davranmaz.
- `pytest-cache-files-*`, `.tmp/`, `__pycache__/`, `*.tsbuildinfo` gibi yerel cache dosyalari `.gitignore` ile dislanmistir.
