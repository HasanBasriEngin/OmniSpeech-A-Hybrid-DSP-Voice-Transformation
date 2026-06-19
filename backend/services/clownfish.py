from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import sys

from backend.audio.clownfish_presets import (
    effective_pitch_semitones,
    get_clownfish_preset,
    list_clownfish_presets,
    normalize_clownfish_preset_key,
)


WINDOW_CLASS = "CLOWNFISHVOICECHANGER"
WINDOW_TITLE = "Clownfish Voice Changer"
WM_COPYDATA = 0x004A


@dataclass(frozen=True)
class ClownfishCommandResult:
    supported: bool
    available: bool
    command_sent: bool
    command: str
    message: str


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", wintypes.LPARAM),
        ("cbData", wintypes.DWORD),
        ("lpData", ctypes.c_void_p),
    ]


def is_supported() -> bool:
    return sys.platform.startswith("win")


def _find_window() -> int:
    if not is_supported():
        return 0
    return int(ctypes.windll.user32.FindWindowW(WINDOW_CLASS, WINDOW_TITLE))


def is_available() -> bool:
    return _find_window() != 0


def send_command(command: str) -> ClownfishCommandResult:
    if not is_supported():
        return ClownfishCommandResult(False, False, False, command, "Clownfish API is Windows-only.")

    hwnd = _find_window()
    if hwnd == 0:
        return ClownfishCommandResult(True, False, False, command, "Clownfish window was not found.")

    payload = command.encode("utf-8")
    buffer = ctypes.create_string_buffer(payload)
    cds = COPYDATASTRUCT()
    cds.dwData = 42
    cds.cbData = len(payload)
    cds.lpData = ctypes.cast(buffer, ctypes.c_void_p)
    ctypes.windll.user32.SendMessageW(hwnd, WM_COPYDATA, 0, ctypes.byref(cds))
    return ClownfishCommandResult(True, True, True, command, "Command sent to Clownfish.")


def set_enabled(enabled: bool) -> ClownfishCommandResult:
    return send_command(f"2|{1 if enabled else 0}")


def set_voice_effect(effect_id: int) -> ClownfishCommandResult:
    safe_effect = max(0, min(14, int(effect_id)))
    return send_command(f"3|{safe_effect}")


def set_custom_pitch(pitch: float) -> ClownfishCommandResult:
    safe_pitch = float(max(-15.0, min(15.0, pitch)))
    return send_command(f"3|13|{safe_pitch:.2f}")


def apply_preset(preset_key: str | None, custom_pitch: float | None = None, *, enable: bool = True) -> ClownfishCommandResult:
    key = normalize_clownfish_preset_key(preset_key)
    preset = get_clownfish_preset(key)

    if enable:
        enabled = set_enabled(True)
        if not enabled.command_sent:
            return enabled

    if key == "custom_pitch":
        return set_custom_pitch(effective_pitch_semitones(key, custom_pitch))
    return set_voice_effect(preset.effect_id)


def status_payload() -> dict[str, object]:
    return {
        "supported": is_supported(),
        "available": is_available(),
        "window_class": WINDOW_CLASS,
        "window_title": WINDOW_TITLE,
        "presets": list_clownfish_presets(),
    }
