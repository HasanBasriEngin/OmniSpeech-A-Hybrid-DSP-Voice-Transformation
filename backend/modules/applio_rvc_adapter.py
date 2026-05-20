from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import logging
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APPLIO_LOCK = RLock()
_APPLIO_CONVERTER_CACHE: dict[Path, Any] = {}


@dataclass(frozen=True)
class ApplioRVCSettings:
    f0_method: str = "rmvpe"
    volume_envelope: float = 1.0
    protect: float = 0.5
    split_audio: bool = False
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    proposed_pitch: bool = False
    proposed_pitch_threshold: float = 155.0
    embedder_model: str = "contentvec"
    embedder_model_custom: str | None = None
    clean_audio: bool = False
    clean_strength: float = 0.5


def resolve_applio_root(applio_root: str | os.PathLike[str] | None = None) -> Path:
    raw_root = applio_root or os.getenv("OMNISPEECH_APPLIO_ROOT", "vendor/applio")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    return root.resolve()


def is_applio_available(applio_root: str | os.PathLike[str] | None = None) -> bool:
    root = resolve_applio_root(applio_root)
    return (root / "rvc" / "infer" / "infer.py").exists()


@contextmanager
def _applio_context(applio_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_str = str(applio_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(applio_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _load_voice_converter(applio_root: Path) -> Any:
    cached = _APPLIO_CONVERTER_CACHE.get(applio_root)
    if cached is not None:
        return cached

    if not is_applio_available(applio_root):
        raise RuntimeError(
            "Applio RVC engine is selected, but Applio was not found. "
            "Set OMNISPEECH_APPLIO_ROOT to an Applio checkout or place it under vendor/applio."
        )

    with _applio_context(applio_root):
        try:
            module = importlib.import_module("rvc.infer.infer")
        except ImportError as exc:
            raise RuntimeError(
                "Applio RVC engine is selected, but its Python dependencies are not installed. "
                "Install optional dependencies with: pip install -r requirements-applio.txt"
            ) from exc

        converter_class = getattr(module, "VoiceConverter", None)
        if converter_class is None:
            raise RuntimeError("Applio rvc.infer.infer does not expose VoiceConverter.")

        converter = converter_class()
        _APPLIO_CONVERTER_CACHE[applio_root] = converter
        return converter


def convert_file_with_applio(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] | None,
    *,
    pitch: int = 0,
    index_rate: float = 0.5,
    applio_root: str | os.PathLike[str] | None = None,
    settings: ApplioRVCSettings | None = None,
    device: str | None = None,
) -> None:
    del device  # Applio chooses device through its own Config object.
    root = resolve_applio_root(applio_root)
    params = settings or ApplioRVCSettings()

    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    index = Path(index_path).expanduser().resolve() if index_path else None

    with _APPLIO_LOCK:
        converter = _load_voice_converter(root)
        with _applio_context(root):
            converter.convert_audio(
                audio_input_path=str(source),
                audio_output_path=str(target),
                model_path=str(model),
                index_path=str(index) if index is not None else "",
                pitch=pitch,
                f0_method=params.f0_method,
                index_rate=index_rate,
                volume_envelope=params.volume_envelope,
                protect=params.protect,
                split_audio=params.split_audio,
                f0_autotune=params.f0_autotune,
                f0_autotune_strength=params.f0_autotune_strength,
                proposed_pitch=params.proposed_pitch,
                proposed_pitch_threshold=params.proposed_pitch_threshold,
                embedder_model=params.embedder_model,
                embedder_model_custom=params.embedder_model_custom,
                clean_audio=params.clean_audio,
                clean_strength=params.clean_strength,
                export_format="WAV",
                post_process=False,
                sid=0,
            )


def clear_applio_engine_cache() -> None:
    _APPLIO_CONVERTER_CACHE.clear()
