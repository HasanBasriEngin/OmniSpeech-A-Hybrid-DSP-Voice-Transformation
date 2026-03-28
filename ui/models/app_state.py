from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AppState:
    sample_rate: int = 16000
    input_path: str = ""
    input_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    processed_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    target_mode: str = "emotion"
    session_log: list[dict] = field(default_factory=list)
    processing_ms: float = 0.0

