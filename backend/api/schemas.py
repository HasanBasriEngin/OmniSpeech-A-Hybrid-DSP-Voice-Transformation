from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str


class ConversionResponse(BaseModel):
    output_path: str
    metrics: dict[str, float] = Field(default_factory=dict)


class DSPFeedbackRequest(BaseModel):
    profile_name: str
    feedback: Literal["clean", "muffled", "harsh", "robotic", "too_thin", "too_thick", "noisy", "unnatural"]


class DSPFeedbackResponse(BaseModel):
    profile_name: str
    feedback: str
    settings: dict[str, object] = Field(default_factory=dict)


class EmotionRequest(BaseModel):
    input_path: str
    emotion: Literal["sad", "angry", "excited", "whisper", "calm"]
    pitch_override: float | None = None
    rate_override: float | None = None
    energy_override: float | None = None
    use_ai_engines: bool = True
    output_path: str | None = None


class GenderAgeRequest(BaseModel):
    input_path: str
    mode: str
    use_ai_engines: bool = True
    output_path: str | None = None


class SpeakerCloneRequest(BaseModel):
    input_path: str
    reference_paths: list[str] = Field(default_factory=list)
    use_ai_engines: bool = True
    output_path: str | None = None


class SingingRequest(BaseModel):
    input_path: str
    midi_path: str | None = None
    pitch_contour: list[float] | None = None
    use_ai_engines: bool = True
    output_path: str | None = None


class CelebrityRequest(BaseModel):
    input_path: str
    celebrity: Literal["michael_jackson", "morgan_freeman", "adele", "james_earl_jones", "taylor_swift"]
    use_ai_engines: bool = True
    output_path: str | None = None


class LiveSessionStartRequest(BaseModel):
    task: Literal["emotion", "gender_age", "speaker_clone", "singing", "celebrity"]
    options: dict[str, object] = Field(default_factory=dict)
    route_to_virtual_mic: bool = False
    virtual_mic_device: str | None = None


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
