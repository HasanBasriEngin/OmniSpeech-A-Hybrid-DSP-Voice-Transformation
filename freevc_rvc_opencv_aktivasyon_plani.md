# FreeVC, RVC ve OpenCV Aktivasyon Plani

Bu planin hedefi OmniSpeech icinde FreeVC, RVC ve OpenCV yollarini sadece hazir
durumdan cikarip, test edilebilir, guvenli ve kullaniciya metriklerle gorunur
sekilde aktif hale getirmektir.

Senin rolu: Hasan Basri Engin olarak diger ekip uyelerinin yaptiklarini
kontrol etmek, entegrasyonu gozden gecirmek, kaliteyi iyilestirmek ve son karar
vermek. Sana dogrudan gelistirme gorevi verilmeyecek.

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
- FreeVC referans sesi temizlenmeli, kalite puani uretilmeli ve kotu referanslarda kullanici uyarilmali.
- RVC icin registry, model import ve data egitimden gelen cikti baglama akisi net olmali.
- Gender/Age, Emotion, Singing, Licensed Profile ve Live mode modulleri kendi amaclarina uygun ve testli sekilde saglam calismali.
- Model eksikse sistem sessizce bozulmamali; fallback veya acik hata davranisi testlerle garanti edilmeli.
- UI, hangi motorun calistigini ve hangisinin fallback'e dustugunu acik gostermeli.
- Kotuye kullanim riskini azaltmak icin consent, lisans bilgisi ve metadata/watermark katmani eklenmeli.
- Implementasyon sirasinda harici API ile ses cekimi, dataset toplama, voice clone, TTS veya cloud inference kullanilmayacak.

## Calisma Kurallari

- Her ekip uyesi yalnizca kendi sahiplik alanindaki dosyalara dokunacak.
- Ortak dosya gerekiyorsa once not dusulecek, sonra entegrasyon asamasinda birlestirilecek.
- Buyuk model dosyalari git'e eklenmeyecek.
- Testler mumkun oldugunca ayri dosyalara bolunecek; tek `tests/test_backend_pipeline.py` dosyasinda herkes ayni anda calismayacak.
- Her ekip uyesi kendi isinin sonunda kisa bir "tamamlandi / risk / test" notu birakacak.
- Sen sadece review, duzenleme ve kalite kontrol yapacaksin.

## Harici API ve Ses Cekimi Yasagi

- Ekip uyeleri bu gorevleri implemente ederken harici API kullanmayacak.
- API ile ses cekimi, ses datasi toplama, TTS uretimi, voice clone servisi veya cloud inference yasak.
- Hugging Face, YouTube, sosyal medya, streaming servisleri veya baska platformlardan API/scraping ile ses cekilmeyecek.
- Test ve egitim sesleri sadece yerel dosya olarak, izinli/rizali kaynaklardan gelecek.
- Projenin kendi lokal FastAPI endpointleri ve Tauri-backend cagrilari bu yasagin disindadir; bunlar uygulamanin ic sozlesmesidir, harici veri kaynagi degildir.
- Model dosyalari ve ses datasi git'e eklenmeyecek; yalnizca README, registry ornekleri, test fixture'lari ve izinli kucuk sentetik dosyalar eklenebilir.

## 4 Kisilik Gorev Dagilimi

| Ekip Uyesi | Ana Alan | Yazma Sahipligi | Cakisma Notu |
| --- | --- | --- | --- |
| Emre Boz | FreeVC kalite ve referans hazirlama | `backend/modules/freevc_adapter.py`, `backend/audio/reference_quality.py`, `backend/audio/reference_preprocess.py`, `tests/test_freevc_quality.py` | RVC ve UI dosyalarina dokunmayacak. |
| Vural YILMAZ | RVC model import, registry ve egitim cikti baglama | `backend/modules/rvc_adapter.py`, `backend/tools/import_rvc_model.py`, `models/rvc/README.md`, `models/rvc/registry.example.json`, `tests/test_rvc_activation.py` | FreeVC adapter'a dokunmayacak. |
| Eren DONMEZ | OpenCV, DSP metrikleri ve backend test matrisi | `backend/audio/spectrogram_image.py`, `backend/audio/filtering.py`, `tests/test_engine_metrics.py`, `tests/fixtures/` | Pipeline davranisini degistirmeden metrik/test ekleyecek. |
| Ilker Tugberk Evren | UI ve lokal backend motor gostergesi, consent ve dokumantasyon | `src/App.tsx`, `src/components/`, `src/types/`, `backend/api/schemas.py`, `README.md`, `freevc_rvc_opencv_aktivasyon_plani.md` | Backend motor kodlarina dokunmayacak; harici API kullanmayacak. |

## Modul Bazli Eksik Ihtiyac Listesi

Bu bolum, her gorev sahibinin kendi alaninda yalnizca motoru baglamasini degil,
ilgili modulu saglam ve kalite kontrolu yapilabilir hale getirmesini hedefler.

### 1. FreeVC'yi Guclendirmek

Sahip: Emre Boz

Yapilacaklar:

- Referans ses kalite analizi.
- Referans temizleme pipeline'i.
- Coklu referans destegi.
- FreeVC hata/fallback politikasi.
- UI'da FreeVC metrikleri icin backend metrik hazirligi.

Gerekli veri:

- 10-30 saniyelik temiz referans sesler.
- Sesler API ile cekilmeyecek; kullanici tarafindan yerel ve izinli dosya olarak saglanacak.
- Farkli kalite ornekleri:
  - temiz
  - gurultulu
  - cok kisa
  - clipping olan

Saglam calisma kriteri:

- Referans kotuyse sistem bunu metrikle gosterir.
- Referans temizse FreeVC speaker clone akisi `freevc_engine = 1.0` ile calisir.
- FreeVC hata verirse davranis tahmin edilebilir olur: acik hata veya kontrollu fallback.

### 2. RVC'yi Aktif Etmek

Sahip: Vural YILMAZ

Yapilacaklar:

- Izinli/rizali hedef ses datasindan gelen modeli sisteme baglama.
- RVC training araci veya training pipeline kararini dokumante etme.
- Egitim sonrasi cikan dosyalari import etme:

```text
model.pth
model.index
models/rvc/registry.json
```

Gerekli veri:

- Izinli/rizali hedef ses datasi.
- 20-40 dakika temiz WAV kayit.
- Tek kisi, temiz ortam, clipping olmayan kayit.
- Data API/scraping ile toplanmayacak; yerel ve izinli kayit olarak alinacak.

Saglam calisma kriteri:

- Registry yoksa sistem fallback'e duser.
- Registry var ama model yoksa acik hata verir.
- Model dogruysa `rvc_engine = 1.0` uretir.
- Model dosyalari git'e eklenmez.

### 3. Gender/Age Modulunu Iyilestirmek

Sahip: Eren DONMEZ

Yapilacaklar:

- Erkek, kadin, cocuk, yasli referans profilleriyle presetleri kontrol etme.
- Pitch/formant preset kalibrasyonu.
- Metrik karsilastirma testleri.
- OpenCV preprocess metriklerinin gender/age akisi icinde dogru gorunmesini saglama.

Gerekli veri:

- Erkek, kadin, cocuk, yasli profil ornekleri.
- Her profil icin temiz kisa test sesleri.
- Profil ornekleri API ile cekilmeyecek; izinli lokal test dosyalari olacak.

Saglam calisma kriteri:

- Her mod icin cikis finite, normalize ve dinlenebilir olur.
- Pitch/formant farklari metriklerde anlamli sekilde gorunur.
- OpenCV varsa `opencv_spectrogram_applied = 1.0`, yoksa kontrollu fallback gorunur.

### 4. Celebrity Yerine Licensed Profile Modulunu Guvenli Yapmak

Sahip: Vural YILMAZ ve Ilker Tugberk Evren

Yapilacaklar:

- "Celebrity" yerine "licensed profile" mantigini dokumante edip UI'da yansitma.
- Sadece izinli ses profilleriyle calisma.
- Harici API ile unlu/ucuncu kisi sesi cekmeyi acikca yasaklama.
- Consent bilgisi.
- Model registry'de lisans/izin alanlari.
- UI'da uyari ve kilit.

Saglam calisma kriteri:

- Consent yoksa licensed profile/RVC akisi baslamaz.
- Registry lisans ve izin metadata'si tasir.
- Kullanici modelin izinli profil oldugunu arayuzde gorur.

### 5. Emotion Modulunu Iyilestirmek

Sahip: Hasan Basri Engin review rolunde, uygulama icin ayri ekip karari gerekir.

Yapilacaklar:

- Duygu ornekleriyle kalibrasyon:
  - calm
  - sad
  - angry
  - excited
  - whisper
- Pitch, hiz ve enerji presetlerinin kalibrasyonu.
- Daha dogal prosody ayarlari.
- Test sesleriyle karsilastirma.

Gerekli veri:

- Her duygu icin temiz test ornekleri.
- Ayni cumlenin farkli duygu tonlariyla okunmus ornekleri ideal olur.
- Duygu ornekleri API ile cekilmeyecek; yerel, izinli veya sentetik test dosyasi olacak.

Saglam calisma kriteri:

- Duygular birbirinden metrik ve dinleme testinde ayrisabilir.
- Cikislar clipping yapmaz.
- Live DSP akisi bozulmaz.

### 6. Singing Modulunu Saglamlastirmak

Sahip: Ekip disi sonraki faz veya ayrica atanacak kisi.

Yapilacaklar:

- MIDI + vokal test ciftleri.
- Pitch contour dogrulama.
- Sarki icin ayri voice conversion modeli degerlendirme.
- FreeVC'nin burada ana cozum olmadigini dokumante etme.
- MIDI/vokal ornekleri API ile cekilmeyecek; izinli lokal dosyalar kullanilacak.

Saglam calisma kriteri:

- MIDI/pitch contour girdileri beklenen cikisi uretir.
- FreeVC singing icin otomatik ana motor olarak kullanilmaz.
- Singing pipeline hata durumunda acik mesaj verir.

### 7. Live Mode

Sahip: Eren DONMEZ backend metrikleri, Ilker Tugberk Evren UI gostergesi.

Yapilacaklar:

- Latency olcumu.
- Chunk boyutu optimizasyonu.
- FreeVC/RVC live icin agir kalabilecegi icin DSP live ve AI file mode ayrimini netlestirme.
- UI'da live mode icin "DSP live / AI file mode" ayrimini gosterme.

Saglam calisma kriteri:

- Live mode dusuk latency ile DSP uzerinden calisir.
- FreeVC/RVC live modda yanlis beklenti olusturmaz.
- Chunk cikislari finite ve guvenli olur.

### 8. Guvenlik ve Kotuye Kullanim Onleme

Sahip: Vural YILMAZ ve Ilker Tugberk Evren

Yapilacaklar:

- Consent checkbox.
- Model registry izin bilgisi.
- Output WAV metadata.
- Speaker verification opsiyonunu planlama.
- Model dosyalarini git disinda tutma.
- Harici API ile ses cekimini, dataset toplamayi ve cloud voice servislerini yasaklama.

Saglam calisma kriteri:

- Izin bilgisi olmayan model profili aktif edilmez.
- Output dosyasinda motor/model metadata'si tutulabilir.
- `.pth`, `.index`, `.safetensors`, buyuk HF varliklari git'e girmez.

### 9. Test ve Dogrulama

Sahip: Eren DONMEZ backend test matrisi, herkes kendi alan testinden sorumlu.

Yapilacaklar:

- OpenCV aktif/fallback testleri.
- FreeVC gercek smoke test.
- RVC mock + gercek model test.
- UI motor gostergeleri.
- End-to-end test sesleri.
- Test seslerinin kaynak/izin bilgisini dokumante etme.

Saglam calisma kriteri:

- Her motor icin aktif/pasif/fallback test yolu vardir.
- Gercek ses ornekleriyle kalite test seti tutulur.
- Finalde tum backend testleri ve frontend build gecer.

## En Kritik Eksikler

Oncelik sirasi:

1. RVC icin izinli egitim datasi veya hazir `.pth/.index` model.
2. FreeVC referans kalite/temizleme sistemi.
3. UI'da motorlarin aktif/pasif gorunmesi.
4. Consent/guvenlik katmani.
5. Gercek ses ornekleriyle kalite test seti.

## Emre Boz: FreeVC Kalite ve Referans Hazirlama

### Amac

FreeVC'nin sadece calismasini degil, iyi referansla daha kaliteli calismasini
saglamak.

### Yapilacaklar

- Referans ses kalite analiz modu ekle.
- Referans ses icin sessizlik orani, clipping orani, RMS seviyesi, sure ve tahmini gurultu puani hesapla.
- Referans sesi FreeVC'ye vermeden once normalize et, sessizlikleri kirp ve gerekiyorsa hafif noise reduction uygula.
- Coklu referans gelirse en kaliteli referansi sec.
- FreeVC sonuc metriklerine su alanlari ekle:

```text
freevc_engine
reference_quality_score
reference_duration_seconds
reference_silence_ratio
reference_clipping_ratio
reference_rms_db
selected_reference_index
```

### Dosya Sahipligi

```text
backend/modules/freevc_adapter.py
backend/audio/reference_quality.py
backend/audio/reference_preprocess.py
tests/test_freevc_quality.py
```

### Kabul Kriterleri

- 10 saniyeden kisa referans icin kalite puani dusmeli.
- Clipping olan referans icin uyarici metrik uretilmeli.
- Coklu referansta en temiz olan secilmeli.
- FreeVC calisirsa `freevc_engine = 1.0` kalmali.
- FreeVC assetleri yoksa mevcut fallback bozulmamali.

## Vural YILMAZ: RVC Model Import, Registry ve Egitim Cikti Baglama

### Amac

RVC'nin hazirlik durumundan cikmasi icin `.pth/.index` model ciktilarini
projeye temiz ve guvenli sekilde baglamak.

### Yapilacaklar

- RVC model import araci ekle.
- Import araci `.pth` ve opsiyonel `.index` dosyasini `models/rvc/<model_id>/` altina yerlestirsin.
- `registry.json` olusturma/guncelleme destegi ekle.
- Registry icine consent ve lisans metadata alanlari ekle.
- Model yoksa acik hata; registry yoksa fallback davranisi korunsun.
- RVC data egitiminden gelen ciktilarin nasil eklenecegi dokumante edilsin.

### Ornek Registry

```json
{
  "gender_age": {
    "male_to_female": {
      "model_id": "my_voice",
      "pitch": 0,
      "index_rate": 0.5,
      "consent_required": true,
      "consent_owner": "authorized_local_voice",
      "license": "private-consent",
      "allow_any_source": false
    }
  }
}
```

### Dosya Sahipligi

```text
backend/modules/rvc_adapter.py
backend/tools/import_rvc_model.py
models/rvc/README.md
models/rvc/registry.example.json
tests/test_rvc_activation.py
```

### Kabul Kriterleri

- `models/rvc/registry.json` yoksa sistem fallback'e duser.
- Registry var ama `.pth` yoksa acik `FileNotFoundError` verir.
- Mock RVC inference ile `rvc_engine = 1.0` testi gecer.
- Import araci model dosyalarini git'e eklemeye calismaz.
- Registry lisans/consent alanlarini destekler.

## Eren DONMEZ: OpenCV, DSP Metrikleri ve Backend Test Matrisi

### Amac

OpenCV ve DSP destek katmanlarinin aktif/pasif durumunu daha iyi olcmek ve
regresyonlari yakalamak.

### Yapilacaklar

- OpenCV aktifken `opencv_spectrogram_applied = 1.0` testini ayri dosyada netlestir.
- OpenCV yokmus gibi simule edilen fallback testini koru.
- Post-filter cikisinda clipping, peak, RMS ve finite kontrol metriklerini guclendir.
- Test fixture klasoru olusturup kisa sentetik sesleri ortak kullanilabilir hale getir.
- `pytest` komutunu repo icinde izin sorunu cikarmayacak sekilde dokumante et.

### Dosya Sahipligi

```text
backend/audio/spectrogram_image.py
backend/audio/filtering.py
tests/test_engine_metrics.py
tests/fixtures/
```

### Kabul Kriterleri

- OpenCV kurulu ortamda aktif test `1.0` metrigini dogrular.
- OpenCV olmayan ortam simule edildiginde fallback test gecer.
- Post-filter spike testleri finite ve guvenli cikis uretir.
- Backend testleri tek komutla gecer.

## Ilker Tugberk Evren: UI ve Lokal Backend Motor Gosterimi, Consent ve Dokumantasyon

### Amac

Kullanici hangi motorun calistigini, hangi motorun pasif kaldigini ve hangi
ses/model icin izin gerektigini arayuzden gorebilmeli.

### Yapilacaklar

- UI sonuc bolumunde motor durumlarini goster:

```text
OpenCV: aktif/pasif
FreeVC: aktif/pasif
RVC: aktif/pasif
Fallback: kullanildi/kullanilmadi
```

- Consent checkbox ekle:

```text
Bu referans/model icin gerekli izinlere sahibim.
```

- RVC/FreeVC kullanilan ciktilarda metadata bilgisini goster.
- README'ye aktiflik kontrolu, FreeVC referans onerileri ve RVC model import adimlarini ekle.
- Lokal backend schema gerekiyorsa consent ve model metadata alanlarini ekle.
- UI'da harici API ile ses cekiminin yasak oldugunu ve sadece izinli lokal dosyalarin kullanilacagini belirt.

### Dosya Sahipligi

```text
src/App.tsx
src/components/
src/types/
backend/api/schemas.py
README.md
freevc_rvc_opencv_aktivasyon_plani.md
```

### Kabul Kriterleri

- Kullanici donusumden sonra `freevc_engine`, `rvc_engine`, `opencv_spectrogram_applied` degerlerini gorebilir.
- Consent onayi olmadan RVC/izinli profil akisina gecilmez.
- README yeni akisi aciklar.
- UI metinleri teknik ama anlasilir olur.

## RVC Data Egitimi Icin Gerekenler

RVC egitimi proje icinde henuz yok. Kisa vadede dis RVC trainer ile model
egitilip `.pth/.index` ciktilari OmniSpeech'e import edilecek.

### Gerekli Data

- 20-40 dakika temiz WAV kayit.
- Tek kisi, tek mikrofon, mumkunse ayni oda.
- Arka plan muzik yok.
- Clipping yok.
- Farkli cumleler, dogal tonlama.
- Sessizlik ve nefes bolumleri makul seviyede.
- Kayitlar API/scraping ile cekilmis olmayacak; riza alinmis yerel kayit olacak.

### Egitimden Beklenen Ciktilar

```text
my_voice.pth
my_voice.index
```

### Projeye Ekleme

```text
models/rvc/my_voice/
  my_voice.pth
  my_voice.index
models/rvc/registry.json
```

## Guvenlik ve Kotuye Kullanim Onlemleri

- RVC varsayilan olarak hedef model yokken pasif kalacak.
- Model dosyalari git'e eklenmeyecek.
- Registry consent/lisans bilgisi tasiyacak.
- UI consent onayi isteyecek.
- Ciktilara metadata/watermark eklenmesi degerlendirilecek.
- Ileride speaker verification ile "input sesi izinli kisiye benziyor mu" kontrolu eklenebilir.
- Harici API ile ses cekimi, TTS/voice clone servisi ve cloud inference kullanimi yasak kalacak.

## Test Matrisi

| Alan | Test | Beklenen |
| --- | --- | --- |
| Ortam | `cv2`, `torch`, `soundfile` import | Basarili |
| OpenCV | Aktif preprocess | `opencv_spectrogram_applied = 1.0` |
| OpenCV | Fallback preprocess | `opencv_spectrogram_applied = 0.0` |
| FreeVC | Config bulma | `get_freevc_config()` dolu doner |
| FreeVC | Referans kalite analizi | kalite metrikleri uretilir |
| FreeVC | Coklu referans secimi | en iyi referans secilir |
| FreeVC | Speaker clone | `freevc_engine = 1.0` |
| RVC | Registry yok | fallback veya `rvc_engine = 0.0` |
| RVC | Model yok | acik `FileNotFoundError` |
| RVC | Mock inference | `rvc_engine = 1.0` |
| UI | Motor gosterimi | aktif/pasif durumlari gorunur |
| Guvenlik | Consent yok | izinli profil/RVC engellenir |
| Guvenlik | Harici API yasagi | API ile ses cekimi veya cloud voice servisi kullanilmaz |
| Full backend | Tum testler | gecer |

## Dogrulama Komutlari

```powershell
.\.venv\Scripts\python -m pytest tests/test_backend_pipeline.py --basetemp=.tmp\pytest-base -p no:cacheprovider
.\.venv\Scripts\python -m pytest tests/test_freevc_quality.py --basetemp=.tmp\pytest-base -p no:cacheprovider
.\.venv\Scripts\python -m pytest tests/test_rvc_activation.py --basetemp=.tmp\pytest-base -p no:cacheprovider
.\.venv\Scripts\python -m pytest tests/test_engine_metrics.py --basetemp=.tmp\pytest-base -p no:cacheprovider
```

## Entegrasyon Sirasi

1. Eren DONMEZ OpenCV ve metrik testlerini ayri dosyaya alir.
2. Emre Boz FreeVC kalite ve referans hazirlama katmanini ekler.
3. Vural YILMAZ RVC import/registry akisini tamamlar.
4. Ilker Tugberk Evren UI/lokal backend motor gosterimi ve consent katmanini ekler.
5. Sen tum degisiklikleri kontrol eder, cakisan noktalar varsa duzenler, kalite
   iyilestirmesi yapar ve final testleri calistirirsin.

## Final Kabul Kriterleri

- Backend testleri gecer.
- FreeVC referans kalitesi olculur ve speaker clone aktif calisir.
- OpenCV aktif/pasif durumu metriklerle dogrulanir.
- RVC hedef model eklendiginde registry uzerinden aktif calisir.
- UI hangi motorun calistigini gosterir.
- Consent olmadan izinli profil/RVC akisi baslamaz.
- Harici API ile ses cekimi, dataset toplama, TTS/voice clone servisi veya cloud inference kullanilmaz.
- Buyuk model dosyalari git'e girmez.
