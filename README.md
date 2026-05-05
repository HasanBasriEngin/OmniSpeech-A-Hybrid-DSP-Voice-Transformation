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

## Gelistirme Ortam Degiskenleri

| Degisken | Aciklama |
|----------|----------|
| `OMNISPEECH_FORCE_INSTALL=1` | Python ve npm bagimliliklarini zorla tekrar kurar |
| `OMNISPEECH_SETUP_ONLY=1` | Sadece kurulum yapar, uygulamayi acmaz |
| `OMNISPEECH_AUTO_INSTALL=1` | Python/Node/Rust otomatik kurulum denemelerini acik tutar |
| `OMNISPEECH_AUTO_INSTALL_MSVC=1` | Windows'ta MSVC Build Tools otomatik kurulum denemesini acik tutar |

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
