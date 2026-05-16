from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str


# ---------------------------------------------------------------------------
# Engine metadata — returned alongside every conversion result so the UI
# can display which engines were active and which fell back.
# ---------------------------------------------------------------------------

class EngineStatus(BaseModel):
    """Motor durum bilgisi — her dönüşüm sonucunda UI'a iletilir."""
    freevc_engine: float = Field(
        default=0.0,
        description="1.0 = FreeVC aktif olarak kullanıldı, 0.0 = fallback/pasif",
        ge=0.0,
        le=1.0,
    )
    rvc_engine: float = Field(
        default=0.0,
        description="1.0 = RVC aktif olarak kullanıldı, 0.0 = fallback/pasif",
        ge=0.0,
        le=1.0,
    )
    opencv_spectrogram_applied: float = Field(
        default=0.0,
        description="1.0 = OpenCV spektrogram ön-işleme uygulandı, 0.0 = atlandı",
        ge=0.0,
        le=1.0,
    )
    fallback_used: bool = Field(
        default=False,
        description="Herhangi bir motor fallback'e düştüyse True",
    )


class ModelMetadata(BaseModel):
    """Çıktıda kullanılan model hakkında lisans ve izin bilgisi."""
    model_id: str | None = Field(default=None, description="Kullanılan modelin ID'si")
    license: str | None = Field(default=None, description="Model lisansı (ör. 'private-consent', 'MIT')")
    consent_owner: str | None = Field(
        default=None,
        description="Sesi izin verilen kişi (ör. 'authorized_local_voice')",
    )
    is_licensed_profile: bool = Field(
        default=False,
        description="Bu model izinli/lisanslı bir profil mi?",
    )


class ConversionResponse(BaseModel):
    output_path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    engine_status: EngineStatus = Field(default_factory=EngineStatus)
    model_metadata: ModelMetadata = Field(default_factory=ModelMetadata)


class EmotionRequest(BaseModel):
    input_path: str
    emotion: Literal["sad", "angry", "excited", "whisper", "calm"]
    pitch_override: float | None = None
    rate_override: float | None = None
    energy_override: float | None = None
    output_path: str | None = None


class GenderAgeRequest(BaseModel):
    input_path: str
    mode: str
    output_path: str | None = None


class SpeakerCloneRequest(BaseModel):
    input_path: str
    reference_paths: list[str] = Field(default_factory=list)
    output_path: str | None = None


class SingingRequest(BaseModel):
    input_path: str
    midi_path: str | None = None
    pitch_contour: list[float] | None = None
    output_path: str | None = None


# ---------------------------------------------------------------------------
# Licensed Profile — replaces the old CelebrityRequest.
# Requires explicit consent before the conversion pipeline starts.
# External API-based voice fetching is strictly forbidden.
# ---------------------------------------------------------------------------

class LicensedProfileRequest(BaseModel):
    """
    İzinli profil dönüşüm isteği.

    Kural: Bu endpoint yalnızca yerel, izin alınmış ses profilleriyle çalışır.
    Harici API ile ses çekimi, TTS veya cloud inference kesinlikle yasaktır.
    """
    input_path: str
    profile_id: str = Field(
        description=(
            "models/rvc/registry.json içinde tanımlı, izinli profil ID'si. "
            "Ünlü/üçüncü kişi sesi içeren profiller kabul edilmez."
        )
    )
    consent_confirmed: bool = Field(
        default=False,
        description=(
            "Kullanıcı 'Bu referans/model için gerekli izinlere sahibim' onayını verdiyse True. "
            "False ise dönüşüm başlamaz."
        ),
    )
    output_path: str | None = None


# Backward-compat alias — eski /celebrity endpoint'ini besleyen kodlar kırılmasın.
class CelebrityRequest(BaseModel):
    input_path: str
    celebrity: Literal["michael_jackson", "morgan_freeman", "adele", "james_earl_jones", "taylor_swift"]
    output_path: str | None = None


class LiveSessionStartRequest(BaseModel):
    task: Literal["emotion", "gender_age", "speaker_clone", "singing", "celebrity"]
    options: dict[str, object] = Field(default_factory=dict)
    route_to_virtual_mic: bool = False
    virtual_mic_device: str | None = None
    consent_confirmed: bool = Field(
        default=False,
        description="İzinli profil/RVC akışı kullanılıyorsa consent onayı gereklidir.",
    )


class LiveSessionStartResponse(BaseModel):
    session_id: str


class LiveChunkRequest(BaseModel):
    session_id: str
    chunk: list[float]


class LiveChunkResponse(BaseModel):
    chunk: list[float]


class LiveSessionStopRequest(BaseModel):
    session_id: str


class VirtualMicDevicesResponse(BaseModel):
    devices: list[str]
