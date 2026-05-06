# Eksik Gorev Uygulama Plani

Tarih: 7 Mayis 2026
Hedef branch: `integration/team-merge-2026-05-06`
Ana hedef: Gorev dagilimi planinda eksik kalan kisimlari kapatip branch'i merge-ready hale getirmek.

Durum: Uygulandi.

## Oncelik Sirasi

1. Emotion DSP eksiklerini tamamla
2. Boot timeout/polling suresini kisalt
3. UI dil ve son polish islerini temizle
4. Test ve build dogrulamasi yap

## 1. Emotion DSP

Dosyalar:
- `backend/modules/emotion.py`
- `backend/audio/features.py`

Yapilacaklar:
- `EMOTION_PROFILES` degerlerini planla hizala.
- `sad` icin daha agresif pitch ve daha yavas rate uygula.
- `angry` icin daha sert attack ve daha guclu distortion ekle.
- `whisper` icin breath/unvoiced hissini artir.
- `excited` icin jitter veya pitch varyasyonu ekle.
- `calm` icin daha duz pitch konturu ve daha stabil enerji kur.
- `_psola_pitch_shift()` fonksiyonu ekle.
- `convert_emotion()` icinde gecici `pitch_shift_audio(...)` yerine PSOLA tabanli akis kullan.
- Duyguya ozel formant manipulesi ekle.
- `_apply_prosody()` fonksiyonunu sinus tabanli yapidan daha dogal enerji/prosody zarfi yapisina cevir.
- `extract_pitch_contour()` icin opsiyonel `frame_length` parametresi ekle.

Bitis kriteri:
- Kodda plan maddelerine karsilik gelen yardimci fonksiyonlar acikca gorunmeli.
- Bes duygu ciktilari birbirinden daha belirgin ayrismali.
- Mevcut cagri yapilari bozulmamali.

## 2. Boot ve Startup

Dosyalar:
- `src-tauri/src/backend.rs`

Yapilacaklar:
- `wait_for_backend_ready()` polling dizisini kisalt.
- Toplam bekleme suresini yaklasik 3 saniye civarina indir.
- Ilk denemeleri kisa tut, sonra artan aralik kullan.
- Var olan lazy init akisina dokunma:
  - `backend/server.py`
  - `backend/config.py`
  - `src-tauri/src/main.rs`

Bitis kriteri:
- Backend hazir olma bekleme suresi plan hedefiyle uyumlu olmali.
- Uygulama erken acilma davranisi korunmali.

## 3. UI Dil ve Son Polish

Dosyalar:
- `src/App.tsx`
- Gerekirse `src/index.css`
- Gerekirse `src/components/*`
- Gerekirse `src/main.tsx`

Yapilacaklar:
- Kalan Ingilizce durum metinlerini Turkcelestir.
- `live`, `ready`, `offline` gibi durum yaziari tek dilde olsun.
- Log mesajlarini ayni dil ve tonda toparla.
- Hata mesajlarini daha tutarli hale getir.
- Celebrity secim loglarini da Turkceye cevir.
- UI tarafinda mantik degil, sadece gorunum/dil temizligi yap.

Bitis kriteri:
- Arayuzde karisik Turkce-Ingilizce yapi kalmamali.
- Var olan celebrity secimi ve canli mod akisi bozulmamali.

## 4. Dikkat Edilecekler

- Eren tarafindaki su dosyalari gereksiz yere bozma:
  - `backend/modules/gender_age.py`
  - `backend/audio/filtering.py`
  - `backend/audio/io.py`
- Emre tarafindaki celebrity backend/Tauri entegrasyonunu bozma:
  - `backend/modules/celebrity_voice.py`
  - `backend/api/routes.py`
  - `backend/api/schemas.py`
  - `backend/pipeline/processor.py`
  - `src-tauri/src/commands.rs`
  - `src/lib/tauri.ts`
- UI tarafinda is mantigini degistirmen gerekmiyorsa degistirme.

## 5. Dogrulama

Calistir:

```bash
python -m pytest tests/test_backend_pipeline.py -q
npm run build
```

Manuel smoke test:
- Emotion conversion
- Celebrity conversion
- App acilis hizi
- Live session start/stop

## Kisa Todo Checklist

- [x] `emotion.py` profil kalibrasyonlarini tamamla
- [x] `_psola_pitch_shift()` ekle
- [x] Formant manipulesi ekle
- [x] Prosody akisisini gelistir
- [x] `extract_pitch_contour(..., frame_length=...)` destegi ekle
- [x] `backend.rs` polling suresini kisalt
- [x] UI metinlerini tek dilde toparla
- [x] `pytest` calistir
- [x] `npm run build` calistir
- [x] Kisa manuel smoke test yap
