# 🔊 OmniSpeech — CLAUDE.md

> **Proje Adı:** OmniSpeech  
> **Ders:** CENG 384 - Intro. To Signal Processing  
> **Grup:** Group 18  
> **Üyeler:** Vural YILMAZ · Eren DÖNMEZ · Hasan Basri Engin · Emre Boz · İlker Tuğberk Evren  
> **Tarih:** 24.3.2026

---

## 1. Proje Vizyonu ve Kapsamı

**OmniSpeech**, giriş sesinin orijinal dilsel içeriğini bozmadan; duygu, cinsiyet, yaş ve konuşmacı kimliği gibi karakteristik özelliklerini değiştirmeyi amaçlayan bir yazılım platformudur. Proje, **dijital sinyal işleme (DSP)** prensiplerini modern öğrenme temelli yaklaşımlarla birleştirerek **doğal ve anlaşılabilir ses çıktıları** üretmeyi hedefler.

### 1.1 Temel Hedefler

| Hedef | Açıklama |
|---|---|
| Duygu Dönüşümü | Tarafsız konuşmayı üzgün, sinirli, heyecanlı, fısıltı veya sakin tona çevir |
| Cinsiyet / Yaş | Formant ve pitch manipülasyonuyla demografik ses dönüşümü |
| Konuşmacı Klonlama | Az sayıda referans örnekten hedef ses kimliğini yeniden inşa et |
| Şarkı Üretimi | Konuşmayı MIDI girdisiyle senkronize şarkıya dönüştür |
| Dilsel Bütünlük | Tüm dönüşümlerde orijinal kelimeleri ve anlamı koru |

### 1.2 Kapsam Sınırları

- **Kapsam içi:** Masaüstü yazılım, DSP algoritmalar, öğrenme tabanlı dönüşüm, GUI
- **Kapsam dışı:** Ticari ses asistanı, bulut işleme, mobil platform, gerçek zamanlı akış zorunluluğu

---

## 2. Sistem Mimarisi

OmniSpeech, standart kişisel bilgisayarlarda çalışacak şekilde optimize edilmiş bir **Masaüstü Mimarisi** üzerine kuruludur. Tüm DSP ve derin öğrenme çıkarımı, harici donanım gerektirmeksizin standart CPU üzerinde yürütülür.

### 2.1 Teknoloji Yığını (Tech Stack)

| Katman | Araç / Kütüphane | Amaç |
|---|---|---|
| **Dil** | Python 3.10+ | Ana geliştirme dili (açık kaynak, ticari lisanssız) |
| **Dil (alternatif)** | C# | GUI entegrasyonu için seçenek |
| **Matematik & DSP** | NumPy / SciPy | Matris operasyonları ve matematiksel hesaplamalar |
| **Ses İşleme** | Librosa | Öznitelik çıkarımı, F0 tespiti, spektral analiz |
| **Canlı Ses** | PyAudio | Gerçek zamanlı mikrofon akışı ve oynatma |
| **Format Yönetimi** | SoundFile / Pydub | WAV, MP3, FLAC okuma/yazma |
| **Derin Öğrenme** | PyTorch / ONNX Runtime | Duygu & konuşmacı modelleri (CPU optimize) |
| **GUI** | PyQt6 | Masaüstü arayüz çerçevesi |
| **Görselleştirme** | Pyqtgraph / Matplotlib | Gerçek zamanlı waveform ve pitch grafiği |
| **MIDI** | pretty_midi | Şarkı modu için melodi girişi |
| **Donanım** | Standart CPU | Harici DSP çipi veya GPU gerektirmez |
| **İşletim Sistemi** | Windows 10/11, macOS | Hedef platformlar |

---

## 3. Modüler Tasarım ve Veri Akışı

Sistem, 5 ana modülden oluşan **hiyerarşik bir boru hattı (pipeline)** mimarisi izler:

### 3.1 Veri Akış Diyagramı

```
╔══════════════════════════════════════════════════════╗
║                 KULLANICI GİRİŞİ                     ║
║   WAV / MP3 / FLAC dosyası  ·  Mikrofon Akışı        ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║              GİRİŞ MODÜLÜ (Input)                    ║
║  • Otomatik sessizlik algılama (VAD)                  ║
║  • Ses segmentasyonu (max 5sn dilimler)               ║
║  • Genlik normalizasyonu                             ║
║  • Örnekleme: 16 kHz veya 22.05 kHz (mono)           ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║           ÖNİŞLEME MODÜLÜ (Preprocessing)            ║
║  • Akustik öznitelik çıkarımı (MFCC, mel-spektrogram)║
║  • F0 (temel frekans) tespiti ve sürekli takibi      ║
║  • Prosodik bileşen ayrıştırması (zamanlama/tonlama) ║
║  • Spektral bileşen ayrıştırması (vocal tract)       ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║          DSP İŞLEME MOTORU  ←  Kullanıcı Seçimi     ║
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
                        ▼
╔══════════════════════════════════════════════════════╗
║             ÇIKIŞ MODÜLÜ (Output)                    ║
║  • Yüksek kaliteli waveform sentezi                  ║
║  • Gerçek zamanlı oynatma (playback)                 ║
║  • Dosya export: WAV / MP3 / FLAC                    ║
║  • Konuşmacı embedding export (.npy)                 ║
║  • Oturum dönüşüm logu                               ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║           DEĞERLENDİRME MODÜLÜ (Evaluation)          ║
║  • Gecikme ölçümü       → hedef < 500ms              ║
║  • İşlem süresi         → hedef < 2sn / 5sn segment  ║
║  • Anlaşılırlık skoru   → MOS >= 3.5                 ║
║  • Ses kalitesi         → PESQ >= 3.0                ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║                    GUI KATMANI                       ║
║  Waveform · Pitch Grafiği · Progress Bar · Log       ║
╚══════════════════════════════════════════════════════╝
```

### 3.2 Giriş Modülü (Input Module)

**Sorumluluk:** Ses verisini sisteme alır, temizler ve işlemeye hazır hale getirir.

- **Format Desteği:** WAV, MP3, FLAC; gerçek zamanlı mikrofon akışı (opsiyonel)
- **Sinyal:** Mono, 16 kHz ve 22.05 kHz örnekleme hızları
- **Fonksiyonlar:** Otomatik sessizlik algılama (VAD), ses segmentasyonu, genlik normalizasyonu

```python
# core/input_module.py

def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray: ...
def start_mic_stream(sr: int = 16000, chunk: int = 1024) -> Generator: ...
def detect_silence(audio: np.ndarray, threshold_db: float = -40) -> list[tuple]: ...
def segment_audio(audio: np.ndarray, sr: int, max_duration: float = 5.0) -> list[np.ndarray]: ...
def normalize_amplitude(audio: np.ndarray) -> np.ndarray: ...
def validate_format(file_path: str) -> bool: ...
```

### 3.3 Önişleme Modülü (Preprocessing Module)

**Sorumluluk:** Ham ses sinyalini analiz eder, akustik öznitelikleri ve prosodik yapıyı çıkarır.

- **Analiz:** MFCC, mel-spektrogram, kısa zaman Fourier dönüşümü (STFT)
- **F0 Takibi:** `librosa.pyin` veya CREPE ile sürekli temel frekans tespiti
- **Ayrıştırma:** Prozodik bileşenler (zamanlama/tonlama) vs. Spektral bileşenler (vocal tract)

```python
# core/preprocessing.py

def extract_features(audio: np.ndarray, sr: int) -> dict:
    # Döndürür: {"mfcc", "mel_spectrogram", "stft", "chroma"}
    ...

def detect_f0(audio: np.ndarray, sr: int,
              fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray: ...

def separate_prosodic_spectral(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    # Döndürür: (prosodic_envelope, spectral_envelope)
    ...

def compute_energy_envelope(audio: np.ndarray, frame_length: int = 2048) -> np.ndarray: ...
```

### 3.4 DSP İşleme Motoru (DSP Engine)

#### 3.4.1 Duygu Dönüşümü

**Perde konturu modülasyonu**, enerji zarfı değişimi ve sinir ağı tabanlı duygu modelleme.

```python
# modules/emotion_conversion.py

EMOTION_PROFILES = {
    "sad":       {"pitch_shift": -2.0, "rate": 0.85, "energy_scale": 0.70, "spectral_tilt": -2.0},
    "angry":     {"pitch_shift": +1.5, "rate": 1.15, "energy_scale": 1.40, "spectral_tilt": +1.5},
    "excited":   {"pitch_shift": +3.0, "rate": 1.25, "energy_scale": 1.30, "spectral_tilt": +2.0},
    "whispered": {"pitch_shift":  0.0, "rate": 0.90, "energy_scale": 0.40, "spectral_tilt": -3.0},
    "calm":      {"pitch_shift": -0.5, "rate": 0.95, "energy_scale": 0.85, "spectral_tilt": -0.5},
}

def convert_emotion(audio: np.ndarray, sr: int, target_emotion: str) -> np.ndarray: ...
def shift_f0_contour(f0: np.ndarray, shift_st: float, modulate: bool = True) -> np.ndarray: ...
def adjust_speech_rate(audio: np.ndarray, sr: int, rate: float) -> np.ndarray: ...
def modify_energy_envelope(audio: np.ndarray, scale: float) -> np.ndarray: ...
def apply_spectral_tilt(audio: np.ndarray, sr: int, tilt_db: float) -> np.ndarray: ...
```

#### 3.4.2 Cinsiyet ve Yaş Dönüşümü

**Formant frekanslarının kaydırılması**, ses yolu uzunluğu simülasyonu (VTLN) ve spektral zarf bükme.

```python
# modules/gender_age_conversion.py

CONVERSION_MAP = {
    "male_to_female":    {"formant_ratio": 1.18, "pitch_shift": +3.5, "vtl_factor": 0.85},
    "female_to_male":    {"formant_ratio": 0.85, "pitch_shift": -3.5, "vtl_factor": 1.18},
    "adult_to_child":    {"formant_ratio": 1.35, "pitch_shift": +5.0, "vtl_factor": 0.74},
    "adult_to_elderly":  {"formant_ratio": 0.92, "pitch_shift": -1.5, "vtl_factor": 1.05},
    "child_to_adult":    {"formant_ratio": 0.74, "pitch_shift": -5.0, "vtl_factor": 1.35},
}

def convert_gender_age(audio: np.ndarray, sr: int, conversion_type: str) -> np.ndarray: ...
def shift_formants(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray: ...
def apply_vtln(audio: np.ndarray, sr: int, warp_factor: float) -> np.ndarray: ...
def warp_spectral_envelope(envelope: np.ndarray, warp: float) -> np.ndarray: ...
```

#### 3.4.3 Konuşmacı Dönüşümü ve Ses Klonlama

**Benzersiz "Speaker Embeddings" çıkarımı**, kimliğin içerikten ayrıştırılması ve hedef kimlikle yeniden inşa.

```python
# modules/speaker_conversion.py

def extract_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    # d-vector veya x-vector tabanlı kimlik temsili
    ...

def convert_speaker(audio: np.ndarray, sr: int,
                    target_embedding: np.ndarray) -> np.ndarray: ...

def clone_voice(audio: np.ndarray, sr: int,
                reference_samples: list[np.ndarray]) -> np.ndarray:
    # Sinirli referans örneklerden ses klonlama
    ...

def separate_content_identity(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    # Döndürür: (content_repr, identity_repr)
    ...

def reconstruct_with_target(content: np.ndarray,
                             target_embedding: np.ndarray) -> np.ndarray: ...

def export_embedding(embedding: np.ndarray, path: str) -> None: ...
```

**Desteklenen eşleme modları:**
- Çok-birden-bire (many-to-one): Birden fazla kaynak → tek hedef kimlik
- Bir-birden-çoğa (one-to-many): Tek kaynak → birden fazla hedef kimlik

#### 3.4.4 Şarkı Sesi Üretimi

**MIDI veya pitch konturu verisiyle senkronizasyon**; ritim ve sürenin müzikal zamana uyarlanması.

```python
# modules/singing_voice.py

def speech_to_singing(audio: np.ndarray, sr: int,
                      melody_input, input_type: str = "midi") -> np.ndarray:
    # input_type: "midi" | "pitch_contour"
    ...

def align_to_melody(audio: np.ndarray, sr: int,
                    target_f0: np.ndarray) -> np.ndarray: ...

def align_rhythm(audio: np.ndarray, sr: int,
                 beat_times: list[float]) -> np.ndarray: ...

def apply_singing_timbre(audio: np.ndarray, sr: int) -> np.ndarray: ...

def load_midi_melody(midi_path: str) -> tuple[np.ndarray, list[float]]:
    # Döndürür: (f0_contour, beat_times)
    ...
```

### 3.5 Çıkış Modülü (Output Module)

```python
# core/output_module.py

def synthesize_waveform(processed_audio: np.ndarray, sr: int) -> np.ndarray: ...
def export_audio(audio: np.ndarray, sr: int, path: str, fmt: str = "wav") -> None: ...
def playback_realtime(audio: np.ndarray, sr: int) -> None: ...
def save_session_log(log_entries: list[dict], path: str) -> None: ...
```

### 3.6 Değerlendirme Modülü (Evaluation Module)

```python
# core/evaluation.py

def measure_latency(fn, *args) -> float: ...           # ms cinsinden
def measure_processing_time(fn, *args) -> float: ...   # saniye cinsinden
def compute_intelligibility(original, processed) -> float: ...  # MOS skoru
def compute_audio_quality(original, processed, sr) -> float:    # PESQ skoru
    ...
def compute_speaker_similarity(emb1, emb2) -> float: ...        # Kosinus benzerligi
```

---

## 4. Frontend ve Kullanıcı Arayüzü (GUI)

### 4.1 UX — 3 Adım Kuralı (NFR 8)

```
ADIM 1: Ses Yükle (Dosya / Mikrofon)
ADIM 2: Modül Seç + Parametreleri Ayarla
ADIM 3: "Dönüştür" → Dinle & Export Et
```

### 4.2 Arayüz Bileşenleri

| Bileşen | Açıklama | Gereksinim |
|---|---|---|
| Dosya Yükleme Alanı | Sürükle-bırak / dosya seçici | FR 1, FR 37 |
| Modül Seçim Paneli | Duygu / Cinsiyet-Yaş / Konuşmacı / Şarkı | FR 37 |
| Parametre Slider'ları | Pitch ratio, tempo, enerji yoğunluğu | NFR 7 |
| Waveform Görüntüleyici | Orijinal ve işlenmiş ses karşılaştırması | FR 38 |
| Pitch Grafiği | Gerçek zamanlı F0 eğrisi | FR 8 |
| İlerleme Çubuğu | Yoğun işlemlerde görsel geri bildirim | NFR 9 |
| Oynatma Kontrolleri | Play / Pause / Stop / Ses seviyesi | FR 36 |
| Session Log | Oturum içi dönüşüm geçmişi | FR 42 |
| Export Düğmesi | Ses dosyası ve embedding çıktısı | FR 35, FR 41 |
| Hata Bildirimleri | Bozuk dosya / kısa ses uyarıları | FR 39, FR 40 |

### 4.3 OmniSpeech Tasarım Dili

- **Tema:** Koyu (Dark) — profesyonel, ses mühendisliği estetiği
- **Renk Paleti:** Mor-mavi accent `#6c63ff` · Teal vurgu `#22d3b0` · Koyu arka plan `#0d0f12`
- **Tipografi:** Sora (UI metin) + JetBrains Mono (kod/değer gösterimi)
- **Stil:** Dashboard-odaklı metrik kartlar + Neumorphism kontrol paneli
- **Görselleştirme:** Canlı waveform canvas, F0 pitch kontur grafiği, animasyonlu progress bar

---

## 5. Fonksiyonel Gereksinimler

### 5.1 Girdi İşleme (FR 1–6)

| ID | Gereksinim |
|---|---|
| FR 1 | Sistem WAV, MP3 ve FLAC formatlarında ses kayıtlarını kabul etmelidir. |
| FR 2 | Sistem gerçek zamanlı mikrofon girişini destekleyebilir (opsiyonel). |
| FR 3 | Sistem 16 kHz ve 22.05 kHz örnekleme hızlarıyla mono ses sinyallerini işlemelidir. |
| FR 4 | Sistem konuşma dışı segmentleri tanımlamak için otomatik sessizlik tespiti yapmalıdır. |
| FR 5 | Sistem uzun kayıtları işlenebilir birimlere bölmek için ses segmentasyonu yapmalıdır. |
| FR 6 | Sistem tutarlı sinyal gücü için ses genlik normalizasyonu uygulamalıdır. |

### 5.2 Ön İşleme (FR 7–9)

| ID | Gereksinim |
|---|---|
| FR 7 | Sistem girdi sinyalinden akustik özellikler çıkarmalıdır. |
| FR 8 | Sistem temel frekans (F0) tespiti ve sürekli takibini yapmalıdır. |
| FR 9 | Sistem prosodik bileşenleri spektral bileşenlerden ayırmalıdır. |

### 5.3 Duygu Dönüşümü (FR 10–15)

| ID | Gereksinim |
|---|---|
| FR 10 | Sistem tarafsız konuşmayı şu duygulara dönüştürmelidir: Üzgün, Sinirli, Heyecanlı, Fısıltı, Sakin. |
| FR 11 | Sistem F0 kaydırma ve modülasyon ile pitch konturunu değiştirmelidir. |
| FR 12 | Sistem konuşma hızı ve fonem süresini duygusal prozodi ile eşleştirmelidir. |
| FR 13 | Sistem konuşma sinyalinin enerji zarfını değiştirmelidir. |
| FR 14 | Sistem spektral eğim ve formant kaydırma uygulayabilmelidir. |
| FR 15 | Sistem duygusal prosodiyi modellemek için sinir ağları kullanmalıdır. |

### 5.4 Cinsiyet ve Yaş Dönüşümü (FR 16–21)

| ID | Gereksinim |
|---|---|
| FR 16 | Sistem Kadin - Erkek ses dönüsümünü desteklemelidir. |
| FR 17 | Sistem yas dönüsümlerini desteklemelidir: Yetiskin-Çocuk, Yetiskin-Yasli, Çocuk-Yetiskin. |
| FR 18 | Sistem farklı vokal kanal boyutlarını simüle etmek için formant frekanslarını ayarlamalıdır. |
| FR 19 | Sistem hedef cinsiyet/yasa uygun pitch aralıgını degistirmelidir. |
| FR 20 | Sistem vokal kanal uzunlugu özelliklerini degistirebilmelidir. |
| FR 21 | Sistem spektral zarf çarpıtma ile sesin tını karakterini ayarlamalıdır. |

### 5.5 Konuşmacı Dönüşümü ve Klonlama (FR 22–28)

| ID | Gereksinim |
|---|---|
| FR 22 | Sistem bir konuşmacının ses kimliğini başka bir konuşmacınınkine dönüştürmelidir. |
| FR 23 | Sistem sınırlı referans ses verisiyle hedef konuşmacı sesini klonlayabilmelidir. |
| FR 24 | Sistem kimlik dönüşümü sırasında orijinal dilsel içeriği korumalıdır. |
| FR 25 | Sistem ses kimliğini temsil eden benzersiz konuşmacı embeddingleri çıkarmalıdır. |
| FR 26 | Sistem konuşmacı kimliği temsillerini içerik temsillerinden ayırmalıdır. |
| FR 27 | Sistem hedef konuşmacı embedding'ini kullanarak konuşmayı yeniden sentezlemelidir. |
| FR 28 | Sistem çok-birden-bire ve bir-birden-çoğa konuşmacı kimliği eşleşmesini desteklemelidir. |

### 5.6 Şarkı Sesi Üretimi (FR 29–33)

| ID | Gereksinim |
|---|---|
| FR 29 | Sistem konuşmayı şarkı sesine dönüştürmelidir. |
| FR 30 | Sistem çıkış pitch'ini tanımlı müzikal melodiye göre kontrol etmelidir. |
| FR 31 | Sistem MIDI dosyaları veya pitch kontur verisi ile melodi girişini desteklemelidir. |
| FR 32 | Sistem ritim ve konuşma süresini müzikal zamanlama ile eşleştirmelidir. |
| FR 33 | Sistem anlamlı bir şarkı tınısı üretmelidir. |

### 5.7 Çıktı Üretimi (FR 34–36)

| ID | Gereksinim |
|---|---|
| FR 34 | Sistem yüksek kaliteli dalga formu sesi üretmelidir. |
| FR 35 | Sistem yapılandırılabilir çıktı dosya formatlarını desteklemelidir. |
| FR 36 | Sistem işlenmiş ses için gerçek zamanlı oynatma sağlamalıdır. |

### 5.8 Kullanıcı Arayüzü (FR 37–38)

| ID | Gereksinim |
|---|---|
| FR 37 | Sistem tüm işleme modülleriyle etkileşim için bir GUI sağlamalıdır. |
| FR 38 | Sistem orijinal ve işlenmiş sinyallerin eş zamanlı önizlemesine izin vermelidir. |

### 5.9 Hata Yönetimi ve Veri Yönetimi (FR 39–42)

| ID | Gereksinim |
|---|---|
| FR 39 | Dosya bozuksa veya desteklenmeyen formattaysa hata mesajı gösterilmelidir. |
| FR 40 | Ses süresi çok kısaysa kullanıcı bilgilendirilmelidir. |
| FR 41 | Çıkartılan konuşmacı embeddingleri gelecek klonlama için export edilebilmelidir. |
| FR 42 | Oturum içinde yapılan tüm dönüşümlerin geçmiş logu tutulmalıdır. |

---

## 6. Fonksiyonel Olmayan Gereksinimler

### 6.1 Performans

| Kod | Gereksinim Tanımı | Hedef Metrik |
|---|---|---|
| NFR 1 | İşlem Süresi | < 5sn sesler için < **2 saniye** |
| NFR 2 | Gecikme (Latency) | Kısa segmentler için < **500 ms** |
| NFR 3 | CPU Optimizasyonu | Standart CPU'da çalışmalı; GPU gerektirmemeli |

### 6.2 Güvenilirlik

| Kod | Gereksinim |
|---|---|
| NFR 4 | Ses Sadakati — dijital artifakt içermeyen yüksek kaliteli ses çıkışı |
| NFR 5 | Sistem çökmesiz çalışmalıdır (yüksek kararlılık) |
| NFR 6 | Anlaşılırlık — dönüşüm sonrası dilsel içeriğin korunması |

### 6.3 Kullanılabilirlik

| Kod | Gereksinim |
|---|---|
| NFR 7 | Ses yükleme, parametre ayarı ve sonuç alma için sezgisel arayüz |
| NFR 8 | Ses dönüşümü en fazla **3 adımda** tamamlanabilmeli |
| NFR 9 | Geri Bildirim — yoğun işlemlerde görsel progress bar desteği |

### 6.4 Taşınabilirlik ve Uyumluluk

| Kod | Gereksinim |
|---|---|
| NFR 10 | Windows 10/11 ve macOS ile uyumlu |
| NFR 11 | Çıktı dosyaları endüstri standardı formatlarda, üçüncü taraf oynatıcılarla uyumlu |

---

## 7. Test ve Doğrulama Senaryoları

| Test Türü | Senaryo | Beklenen Sonuç |
|---|---|---|
| **Giriş Doğrulama** | WAV / MP3 / FLAC yükleme | Başarılı yükleme ve normalizasyon |
| **Giriş Doğrulama** | Bozuk dosya yükleme | Kullanıcıya hata mesajı (FR 39) |
| **Giriş Doğrulama** | Desteklenmeyen format | Hata mesajı + desteklenen format listesi |
| **Giriş Doğrulama** | Çok kısa ses (<0.5sn) | Uyarı mesajı (FR 40) |
| **Algoritma Testi** | Duygu dönüşümü anlaşılırlık | Dilsel içerik korunmalı (NFR 6) |
| **Algoritma Testi** | Cinsiyet dönüşümü kalitesi | Artefaktsız yüksek ses kalitesi (NFR 4) |
| **Algoritma Testi** | Konuşmacı klonlama benzerliği | Cos. sim. >= 0.85 |
| **Performans Testi** | <5sn segment işlem süresi | < 2 saniye (NFR 1) |
| **Performans Testi** | Kısa segment gecikmesi | < 500 ms (NFR 2) |
| **Performans Testi** | Çökme testi (büyük girdi) | Sistem çökmemeli (NFR 5) |
| **UI Testi** | 3 adım kullanım akışı | Kullanıcı 3 adimda dönüşüm yapabilmeli (NFR 8) |

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Sadece performans testleri
pytest tests/test_performance.py -v

# Kapsam raporu
pytest tests/ --cov=core --cov=modules --cov-report=html
```

---

## 8. Değerlendirme Metrikleri

| Metrik | Amaç | Hedef Değer |
|---|---|---|
| İşlem Gecikmesi | Kısa segment yanıt süresi | < 500 ms |
| İşlem Süresi | <5sn segment toplam süre | < 2 saniye |
| Anlaşılırlık (MOS) | Dilsel içerik korunumu | >= 3.5 / 5.0 |
| Ses Kalitesi (PESQ) | Artefakt yokluğu | >= 3.0 |
| Konuşmacı Benzerliği | Klonlama başarısı | Cos. sim. >= 0.85 |
| Çökme Oranı | Sistem kararlılığı | %0 çökme |

---

## 9. Sistem Kısıtlamaları

| Kısıt Türü | Açıklama |
|---|---|
| **Donanım** | Yalnızca standart CPU; harici DSP çipi veya GPU gerektirmez |
| **Lisans** | Yalnızca açık kaynaklı araçlar (ticari lisanssız) |
| **Kapsam** | Akademik ve deneysel kullanım; ticari ses asistanı değil |
| **Platform** | Windows 10/11 ve macOS masaüstü ortamı |
| **Gerçek Zamanlılık** | Tam gerçek zamanlı işleme zorunluluğu yoktur |

---

## 10. Proje Yapısı

```
omnispeech/
│
├── main.py
├── requirements.txt
├── README.md
├── CLAUDE.md
│
├── core/
│   ├── input_module.py
│   ├── preprocessing.py
│   ├── output_module.py
│   └── evaluation.py
│
├── modules/
│   ├── emotion_conversion.py
│   ├── gender_age_conversion.py
│   ├── speaker_conversion.py
│   └── singing_voice.py
│
├── ui/
│   ├── main_window.py
│   ├── waveform_widget.py
│   ├── pitch_graph.py
│   └── parameter_panel.py
│
├── utils/
│   ├── audio_utils.py
│   └── logger.py
│
└── tests/
    ├── test_input.py
    ├── test_preprocessing.py
    ├── test_modules.py
    └── test_performance.py
```

---

## 11. Kurulum ve Çalıştırma

```bash
git clone https://github.com/group18/omnispeech.git
cd omnispeech
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**requirements.txt:**
```
numpy>=1.24
scipy>=1.10
librosa>=0.10
pyaudio>=0.2.13
soundfile>=0.12
pydub>=0.25
torch>=2.0
onnxruntime>=1.16
PyQt6>=6.5
pyqtgraph>=0.13
matplotlib>=3.7
pretty_midi>=0.2
```

---

## 12. Görev Dağılımı

| Üye | Sorumluluk Alanı |
|---|---|
| Vural YILMAZ | Input Modülü + Değerlendirme Modülü |
| Eren DÖNMEZ | Ön İşleme + F0 Tespiti |
| Hasan Basri Engin | Duygu Dönüşümü + Cinsiyet/Yaş Dönüşümü |
| Emre Boz | Konuşmacı Dönüşümü + Klonlama |
| İlker Tuğberk Evren | Şarkı Sesi Üretimi + GUI Tasarımı |

---

*Bu belge SDD_GROUP18.pdf ve SRS_GROUP18.pdf kaynak dokümanları temel alınarak hazırlanmıştır. Proje adı: **OmniSpeech** — Group 18.*
