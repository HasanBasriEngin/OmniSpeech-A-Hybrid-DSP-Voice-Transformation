from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str


class ConversionResponse(BaseModel):
    output_path: str
    metrics: dict[str, float] = Field(default_factory=dict)


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


class LiveSessionStartRequest(BaseModel):
    task: Literal["gender_age", "speaker_clone", "singing"]
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
