# FreeVC, RVC ve OpenCV Aktivasyon Plani

Bu planin hedefi OmniSpeech icinde FreeVC, RVC ve OpenCV yollarini sadece hazir
durumdan cikarip, test edilebilir ve kullaniciya metriklerle gorunur sekilde
aktif hale getirmektir.

## Mevcut Durum

| Bilesen | Durum | Aciklama |
| --- | --- | --- |
| OpenCV | Aktif | `.venv` icinde `cv2` kurulu. `gender_age` ve `celebrity` dosya donusumlerinde spektrogram on-isleme icin kullaniliyor. |
| FreeVC | Aktif | `models/hf/freevc-24` varliklari mevcut. `speaker_clone` dosya donusumunde referans ses varsa FreeVC deneniyor. |
| RVC | Hazirlik var, hedef model yok | `models/hf/rvc-core-v2` core varliklari mevcut. Ancak `models/rvc/registry.json` ve hedef `.pth/.index` modeli olmadigi icin aktif RVC donusumu calismiyor. |

## Hedef Durum

- OpenCV varsa `opencv_spectrogram_applied = 1.0` metriginde net gorunmeli.
- FreeVC, speaker clone dosya donusumunde `freevc_engine = 1.0` uretmeli.
- RVC, gercek ve izinli hedef ses modeli eklendiginde `rvc_engine = 1.0` uretmeli.
- Model eksikse sistem sessizce bozulmamali; fallback veya acik hata davranisi testlerle garanti edilmeli.
- README, kullaniciya hangi motorun aktif oldugunu nasil anlayacagini gostermeli.

## 1. Ortam Kontrolu

### Amac

IDE, test runner ve uygulama ayni Python ortamlarini kullanmali.

### Komutlar

```powershell
.\.venv\Scripts\python -c "import sys; print(sys.executable)"
.\.venv\Scripts\python -c "import cv2, numpy, soundfile, torch; print(cv2.__version__)"
.\.venv\Scripts\python -c "from backend.modules.freevc_adapter import get_freevc_config; print(bool(get_freevc_config()))"
```

### Beklenen Sonuc

- Python yolu `.venv\Scripts\python.exe` olmali.
- OpenCV import edilebilmeli.
- FreeVC config `True` donmeli.

## 2. OpenCV Aktivasyonunu Guclendir

### Yapilacaklar

- `backend/audio/spectrogram_image.py` icindeki OpenCV preprocess yolu korunacak.
- `gender_age` dosya donusumu icin OpenCV aktif test eklenecek.
- `celebrity` dosya donusumu icin OpenCV aktif test eklenecek.
- OpenCV yoksa fallback davranisinin bozulmadigi test edilecek.

### Kabul Kriterleri

- OpenCV kurulu ortamda metrik:

```text
opencv_spectrogram_applied = 1.0
```

- OpenCV yokmus gibi simule edilen testte metrik:

```text
opencv_spectrogram_applied = 0.0
```

## 3. FreeVC Aktivasyonunu Guclendir

### Yapilacaklar

- `backend/modules/freevc_adapter.py` icin asset tamlik kontrolu net tutulacak.
- `speaker_clone` dosya donusumunde ilk referans ses FreeVC hedef stili olarak kullanilacak.
- FreeVC gercek smoke test komutu dokumante edilecek.
- Hata durumunda davranis netlestirilecek:
  - Dosya donusumunde onerilen davranis: acik hata.
  - Canli modda onerilen davranis: fallback.

### Smoke Test

```powershell
.\.venv\Scripts\python -m backend.tools.run_freevc_inference `
  --assets-dir models\hf\freevc-24 `
  --wavlm-model models\hf\wavlm-large `
  --input models\hf\freevc-24\p225_001.wav `
  --reference models\hf\freevc-24\p226_002.wav `
  --output .tmp\freevc_smoke.wav `
  --device cpu
```

### Kabul Kriterleri

- Smoke test `.tmp\freevc_smoke.wav` uretmeli.
- Pipeline metriği:

```text
freevc_engine = 1.0
```

## 4. RVC Hedef Modelini Ekle

### Eksik Parca

RVC icin core varliklar hazir, fakat hedef ses modeli eksik. Gercek donusum icin
izinli veya riza alinmis bir hedef ses modeli gerekir.

### Beklenen Dosya Yapisi

```text
models/rvc/my_voice/
  my_voice.pth
  my_voice.index
models/rvc/registry.json
```

### Ornek Registry

```json
{
  "gender_age": {
    "male_to_female": {
      "model_id": "my_voice",
      "pitch": 0,
      "index_rate": 0.5
    }
  },
  "celebrity": {
    "licensed_profile": {
      "model_id": "my_voice",
      "pitch": 0,
      "index_rate": 0.5
    }
  }
}
```

### Kabul Kriterleri

- `models/rvc/registry.json` mevcut olmali.
- Registry icindeki `model_id`, ayni isimde klasore ve `.pth` dosyasina isaret etmeli.
- RVC calistiginda metrik:

```text
rvc_engine = 1.0
```

## 5. RVC Model Kaynagi Karari

### Guvenli Secenek

En guvenli yol, izinli veya riza alinmis bir sesle kendi RVC modelini egitmektir.

### Hazir Model Kullanilacaksa

Kontrol edilmesi gerekenler:

- Model lisansi
- Veri sahibinin rizasi
- Model kalitesi
- `.pth` ve varsa `.index` dosyalarinin uyumlulugu
- Ticari veya demo kullanim sinirlari

Hazir, lisansi belirsiz unlu/karakter modelleri projeye varsayilan olarak
eklenmemelidir.

## 6. API ve UI Motor Gosterimi

### Yapilacaklar

Donusum sonucunda metrikler UI tarafinda daha gorunur hale getirilecek:

```text
opencv_spectrogram_applied
freevc_engine
rvc_engine
```

### Kabul Kriterleri

Kullanici sonuc ekraninda hangi motorun calistigini anlayabilmeli:

- `1.0`: motor calisti
- `0.0`: fallback veya pasif

## 7. Test Matrisi

| Test | Beklenen |
| --- | --- |
| OpenCV import testi | `cv2` import edilir |
| OpenCV aktif pipeline testi | `opencv_spectrogram_applied = 1.0` |
| OpenCV fallback testi | `opencv_spectrogram_applied = 0.0` |
| FreeVC config testi | `get_freevc_config()` dolu doner |
| FreeVC speaker clone testi | `freevc_engine = 1.0` |
| RVC registry yok testi | fallback veya `rvc_engine = 0.0` |
| RVC model yok testi | acik `FileNotFoundError` |
| RVC mock inference testi | `rvc_engine = 1.0` |
| Full backend test | tum testler gecer |

## 8. Dogrulama Komutu

```powershell
.\.venv\Scripts\python -m pytest tests/test_backend_pipeline.py --basetemp=.tmp\pytest-base -p no:cacheprovider
```

Beklenen:

```text
17 passed
```

## 9. Siradaki Uygulama Adimlari

1. OpenCV aktif testlerini daha acik hale getir.
2. FreeVC gercek smoke testini opsiyonel integration test olarak ayir.
3. RVC icin `models/rvc/registry.json` template dosyasi olustur.
4. Kullanici izinli bir RVC `.pth/.index` modeli ekleyince RVC aktivasyon testini calistir.
5. UI metrik panelinde `opencv`, `freevc`, `rvc` durumlarini goster.

