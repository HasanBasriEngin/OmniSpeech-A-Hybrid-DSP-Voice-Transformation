from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np


def _safe_import_sounddevice():
    try:
        import sounddevice as sd

        return sd
    except Exception:
        return None


VIRTUAL_DEVICE_HINTS = ("cable", "virtual", "blackhole", "loopback", "vb-audio")


@dataclass
class VirtualMicRouter:
    """Writes processed audio frames into a user-selected output device."""

    sample_rate: int
    channels: int = 1
    _stream: object | None = None
    _lock: Lock = field(default_factory=Lock)

    def list_candidate_devices(self) -> list[str]:
        sd = _safe_import_sounddevice()
        if sd is None:
            return []

        candidates: list[str] = []
        for device in sd.query_devices():
            name = str(device.get("name", ""))
            outputs = int(device.get("max_output_channels", 0))
            if outputs < 1:
                continue
            if any(hint in name.lower() for hint in VIRTUAL_DEVICE_HINTS):
                candidates.append(name)
        return candidates

    def open(self, preferred_device: str | None = None) -> None:
        sd = _safe_import_sounddevice()
        if sd is None:
            raise RuntimeError("sounddevice is required for virtual microphone routing.")

        with self._lock:
            if self._stream is not None:
                return

            device_index = None
            if preferred_device:
                for idx, device in enumerate(sd.query_devices()):
                    if preferred_device.lower() in str(device.get("name", "")).lower():
                        if int(device.get("max_output_channels", 0)) > 0:
                            device_index = idx
                            break

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device_index,
                blocksize=0,
            )
            self._stream.start()

    def write(self, chunk: np.ndarray) -> None:
        with self._lock:
            if self._stream is None:
                return
            frame = np.asarray(chunk, dtype=np.float32).reshape(-1, self.channels)
            self._stream.write(frame)

    def close(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
