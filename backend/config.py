from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field


class BackendSettings(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765)
    model_sample_rate: int = Field(default=22050)
    live_sample_rate: int = Field(default=22050)
    rvc_models_dir: str = Field(default_factory=lambda: os.getenv("OMNISPEECH_RVC_MODELS_DIR", "models/rvc"))
    rvc_device: str = Field(default_factory=lambda: os.getenv("OMNISPEECH_RVC_DEVICE", "cpu"))
    freevc_assets_dir: str = Field(default_factory=lambda: os.getenv("OMNISPEECH_FREEVC_ASSETS_DIR", "models/hf/freevc-24"))
    freevc_device: str = Field(default_factory=lambda: os.getenv("OMNISPEECH_FREEVC_DEVICE", os.getenv("OMNISPEECH_RVC_DEVICE", "cpu")))
    freevc_profiles_dir: str = Field(
        default_factory=lambda: os.getenv("OMNISPEECH_FREEVC_PROFILES_DIR", "models/freevc_profiles")
    )
    dsp_profiles_dir: str = Field(default_factory=lambda: os.getenv("OMNISPEECH_DSP_PROFILES_DIR", "models/dsp_profiles"))


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def load_settings() -> BackendSettings:
    return BackendSettings(
        host=os.getenv("OMNISPEECH_HOST", "127.0.0.1"),
        port=_read_int("OMNISPEECH_PORT", 8765),
        model_sample_rate=_read_int("OMNISPEECH_MODEL_SAMPLE_RATE", 22050),
        live_sample_rate=_read_int("OMNISPEECH_LIVE_SAMPLE_RATE", 22050),
        rvc_models_dir=os.getenv("OMNISPEECH_RVC_MODELS_DIR", "models/rvc"),
        rvc_device=os.getenv("OMNISPEECH_RVC_DEVICE", "cpu"),
        freevc_assets_dir=os.getenv("OMNISPEECH_FREEVC_ASSETS_DIR", "models/hf/freevc-24"),
        freevc_device=os.getenv("OMNISPEECH_FREEVC_DEVICE", os.getenv("OMNISPEECH_RVC_DEVICE", "cpu")),
        freevc_profiles_dir=os.getenv("OMNISPEECH_FREEVC_PROFILES_DIR", "models/freevc_profiles"),
        dsp_profiles_dir=os.getenv("OMNISPEECH_DSP_PROFILES_DIR", "models/dsp_profiles"),
    )


SETTINGS = load_settings()
