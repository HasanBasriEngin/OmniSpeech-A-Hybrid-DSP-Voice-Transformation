from __future__ import annotations

from pathlib import Path


def short_path(path: str) -> str:
    if not path:
        return "No file selected"
    p = Path(path)
    if len(str(p)) <= 52:
        return str(p)
    return f"...{str(p)[-49:]}"


def ms_label(ms: float) -> str:
    return f"{ms:.1f} ms"

