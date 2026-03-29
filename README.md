# OmniSpeech 2.0 (Tauri + React + Python)

OmniSpeech, masaustu ses donusumu icin Tauri tabanli bir desktop shell, React tabanli bir arayuz ve Python/FastAPI tabanli bir ses isleme backend'i birlestirir.

## Guncel Durum

Bu repodaki guncel uygulama:
- Tauri desktop uygulamasi olarak acilir (browser odakli localhost uygulamasi degildir).
- `run_omnispeech.bat` ile tek komutta kurulum + calistirma akisini destekler.
- UI'daki ana butonlar backend aksiyonlarina baglidir:
  - Dosya secimi
  - Referans dosya secimi
  - MIDI secimi
  - Conversion calistirma
  - Live session baslat/durdur
  - Virtual mic cihaz listesi yenileme

## Teknoloji Yigini

- Desktop shell: `Tauri 2.x (Rust)`
- Frontend: `React 18 + Vite`
- UI stili: custom CSS (mockup tabanli koyu tema)
- Backend: `FastAPI + PyTorch + Librosa`
- Ses I/O: `soundfile`, `sounddevice`

## Gereksinimler

### Tum platformlar

- Python `3.10+`
- Node.js `20+`

### Windows

- Rust toolchain (`cargo`, `rustup`)
- Visual Studio Build Tools (C++ workload)
  - Desktop development with C++
  - MSVC v143 x64/x86
  - Windows 10/11 SDK

### macOS

- Rust toolchain (`cargo`, `rustup`)
- Xcode Command Line Tools

## Onerilen Baslatma (Windows)

Repodaki otomasyon scripti:

```bat
run_omnispeech.bat
```

Script su islemleri yapar:
- Python bulunmasi / `.venv` olusturma
- `requirements.txt` kurulumu
- `npm install` kontrolu
- esbuild runtime onarimi (gerektiginde)
- Rust/cargo kontrolu
- MSVC linker (`link.exe`) kontrolu
- `1420` portu doluysa stale process temizligi
- `npm run tauri dev` baslatma

### Script icin kullanisli ortam degiskenleri

```bat
set OMNISPEECH_FORCE_INSTALL=1
set OMNISPEECH_SETUP_ONLY=1
set OMNISPEECH_INSTALL_RUST=1
```

- `OMNISPEECH_FORCE_INSTALL=1`: Python/npm bagimliliklarini yeniden kurar
- `OMNISPEECH_SETUP_ONLY=1`: sadece kurulum yapar, uygulamayi acmaz
- `OMNISPEECH_INSTALL_RUST=1`: Rust otomatik kurulumunu dener

## Manuel Calistirma

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
npm install
npm run tauri dev
```

## UI Butonlari ve Islevleri

### Sol / Ust alan

- `Workspace / Evaluation / Settings`: sayfa gorunumu degistirir
- Modul secimi (`Emotion`, `Gender/Age`, `Speaker/Clone`, `Singing`): aktif conversion tipini degistirir

### Audio input karti

- `FILE`: dosya odakli akis
- `MIC`: live session baslat/durdur denemesi
- Drop area: dosya seciciyi tetikler
- `Select Source`: ana ses dosyasi secer
- `References`: speaker clone referans dosyalari secer
- `MIDI`: singing modu icin MIDI dosyasi secer
- `Refresh VMic`: virtual mic cihazlarini backend'den tekrar ceker
- Play/progress: UI preview oynatma simulasyonu

### Sag panel

- Parametre sliderlari (`Pitch`, `Speech Rate`, `Energy`): conversion payload parametrelerini gunceller
- Emotion chip'leri: emotion->mode eslemesini degistirir
- `Route to virtual mic`: live session payload'ina route flag ekler
- Virtual mic secimi: live session payload'ina cihaz adi ekler
- `Start/Stop Live Session`: backend live session endpoint'lerini cagirir
- `Convert Audio`: secili module gore conversion endpoint'ini cagirir
- Session log: tum aksiyonlari zaman damgali listeler

## Backend Endpointleri

- `GET /health`
- `POST /api/convert/gender-age`
- `POST /api/convert/speaker-clone`
- `POST /api/convert/singing`
- `GET /api/live/virtual-mics`
- `POST /api/live/start`
- `POST /api/live/chunk`
- `POST /api/live/stop`

## Proje Dizini (Kisa)

```text
backend/
  api/
  audio/
  modules/
  pipeline/
  services/
  server.py

src/
  App.tsx
  index.css
  lib/tauri.ts

src-tauri/
  src/
    main.rs
    commands.rs
    backend.rs
    types.rs
  tauri.conf.json

run_omnispeech.bat
requirements.txt
package.json
```

## Bilinen Sinirlar

- Live session UI tarafinda backend'de session acma/kapama aktif; surekli mikrofon chunk push akisi su an sinirli/gelistirme asamasindadir.
- Waveform ve bazi playback davranislari UI simulasyonu ile desteklenir.

## Sorun Giderme

### Port 1420 in use

Script otomatik temizlemeyi dener. Yetmezse:

```powershell
netstat -ano | findstr :1420
taskkill /PID <PID> /F
```

### `cargo not found`

```powershell
winget install -e --id Rustlang.Rustup
```

Ardindan yeni terminal acip tekrar deneyin.

### `MSVC linker link.exe not found`

Visual Studio Build Tools + C++ workload kurulmalidir.

### Uygulama acilmiyor / gorunmuyor

- Tum `omnispeech_desktop` process'lerini kapatin
- `run_omnispeech.bat` ile temiz baslatin

