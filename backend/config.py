from __future__ import annotations

from pydantic import BaseModel, Field


class BackendSettings(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765)
    model_sample_rate: int = Field(default=22050)
    live_sample_rate: int = Field(default=22050)


SETTINGS = BackendSettings()
