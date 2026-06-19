from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import sys
from uuid import uuid4
import zipfile

import numpy as np
from scipy import signal
import soundfile as sf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_URL = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
_CONVERTER_FILES = ("config.json", "checkpoint.pth")

# Celebrity reference WAV paths (relative to project root). These files are not
# bundled; users should provide licensed or consented reference clips locally.
CELEBRITY_REFERENCES: dict[str, str] = {
    "michael_jackson": "backend/assets/celebrity_references/michael_jackson.wav",
    "morgan_freeman": "backend/assets/celebrity_references/morgan_freeman.wav",
    "adele": "backend/assets/celebrity_references/adele.wav",
    "james_earl_jones": "backend/assets/celebrity_references/james_earl_jones.wav",
    "taylor_swift": "backend/assets/celebrity_references/taylor_swift.wav",
}

_tone_color_converter = None
_tone_color_converter_key: tuple[str, str] | None = None
_openvoice_available: bool | None = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def resolve_openvoice_root(openvoice_root: str | os.PathLike[str] | None = None) -> Path | None:
    """Return a local OpenVoice checkout if one is configured or vendored."""
    raw_root = openvoice_root or os.getenv("OMNISPEECH_OPENVOICE_ROOT")
    candidates: list[str | os.PathLike[str]] = []
    if raw_root:
        candidates.append(raw_root)
    candidates.extend(
        [
            _PROJECT_ROOT / "vendor" / "OpenVoice",
            _PROJECT_ROOT / "OpenVoice",
            _PROJECT_ROOT / "openvoice",
        ]
    )

    for candidate in candidates:
        root = _resolve_project_path(candidate)
        if (root / "openvoice").is_dir():
            return root
    return None


def _prepare_openvoice_import(openvoice_root: str | os.PathLike[str] | None = None) -> Path | None:
    root = resolve_openvoice_root(openvoice_root)
    if root is not None:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    return root


def is_openvoice_available(
    *,
    openvoice_root: str | os.PathLike[str] | None = None,
    force_refresh: bool = False,
) -> bool:
    """Check if OpenVoice can be imported from pip or a local checkout."""
    global _openvoice_available
    if force_refresh or _openvoice_available is None:
        _prepare_openvoice_import(openvoice_root)
        try:
            from openvoice.api import ToneColorConverter  # noqa: F401

            _openvoice_available = True
        except ImportError:
            _openvoice_available = False
            logger.info(
                "OpenVoice is not installed. Set OMNISPEECH_OPENVOICE_ROOT to a local "
                "OpenVoice checkout or install it in the active Python environment."
            )
    return bool(_openvoice_available)


def _iter_checkpoint_roots(
    *,
    checkpoints_dir: str | os.PathLike[str] | None = None,
    openvoice_root: str | os.PathLike[str] | None = None,
) -> list[Path]:
    roots: list[Path] = []
    raw_dir = checkpoints_dir or os.getenv("OMNISPEECH_OPENVOICE_CHECKPOINTS_DIR")
    if raw_dir:
        roots.append(_resolve_project_path(raw_dir))

    root = resolve_openvoice_root(openvoice_root)
    if root is not None:
        roots.extend([root / "checkpoints", root / "checkpoints_v2"])

    roots.extend(
        [
            _PROJECT_ROOT / "models" / "openvoice" / "checkpoints",
            _PROJECT_ROOT / "checkpoints",
            _PROJECT_ROOT / "vendor" / "OpenVoice" / "checkpoints",
            _PROJECT_ROOT / "OpenVoice" / "checkpoints",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for root_path in roots:
        resolved = root_path.resolve()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _is_converter_checkpoint_dir(path: Path) -> bool:
    return all((path / filename).exists() for filename in _CONVERTER_FILES)


def _find_converter_checkpoint_dir(
    *,
    checkpoints_dir: str | os.PathLike[str] | None = None,
    openvoice_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    for root in _iter_checkpoint_roots(checkpoints_dir=checkpoints_dir, openvoice_root=openvoice_root):
        if not root.exists():
            continue

        direct_candidates = [root, root / "converter"]
        for candidate in direct_candidates:
            if _is_converter_checkpoint_dir(candidate):
                return candidate

        for config_path in root.rglob("config.json"):
            candidate = config_path.parent
            if not _is_converter_checkpoint_dir(candidate):
                continue
            if any(part.lower() == "converter" for part in candidate.parts):
                return candidate
    return None


def _download_checkpoints(destination: Path) -> None:
    import urllib.request

    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "checkpoints_1226.zip"
    url = os.getenv("OMNISPEECH_OPENVOICE_CHECKPOINT_URL", _CHECKPOINT_URL)
    logger.info("Downloading OpenVoice checkpoints to %s", destination)
    urllib.request.urlretrieve(url, str(zip_path))

    with zipfile.ZipFile(str(zip_path), "r") as archive:
        archive.extractall(str(destination))
    zip_path.unlink(missing_ok=True)


def _ensure_converter_checkpoint_dir(
    *,
    checkpoints_dir: str | os.PathLike[str] | None = None,
    openvoice_root: str | os.PathLike[str] | None = None,
) -> Path:
    converter_dir = _find_converter_checkpoint_dir(checkpoints_dir=checkpoints_dir, openvoice_root=openvoice_root)
    if converter_dir is not None:
        return converter_dir

    if _truthy(os.getenv("OMNISPEECH_OPENVOICE_AUTO_DOWNLOAD")):
        download_root = _resolve_project_path(checkpoints_dir or os.getenv("OMNISPEECH_OPENVOICE_CHECKPOINTS_DIR") or "models/openvoice")
        _download_checkpoints(download_root)
        converter_dir = _find_converter_checkpoint_dir(checkpoints_dir=download_root, openvoice_root=openvoice_root)
        if converter_dir is not None:
            return converter_dir

    searched = ", ".join(str(path) for path in _iter_checkpoint_roots(checkpoints_dir=checkpoints_dir, openvoice_root=openvoice_root))
    raise FileNotFoundError(
        "OpenVoice converter checkpoint not found. Place checkpoints under an OpenVoice "
        "checkout, set OMNISPEECH_OPENVOICE_CHECKPOINTS_DIR, or set "
        f"OMNISPEECH_OPENVOICE_AUTO_DOWNLOAD=1. Searched: {searched}"
    )


def _get_tone_color_converter(
    *,
    device: str | None = None,
    openvoice_root: str | os.PathLike[str] | None = None,
    checkpoints_dir: str | os.PathLike[str] | None = None,
):
    """Lazy-load the OpenVoice ToneColorConverter."""
    global _tone_color_converter, _tone_color_converter_key

    openvoice_device = device or os.getenv("OMNISPEECH_OPENVOICE_DEVICE", os.getenv("OMNISPEECH_RVC_DEVICE", "cpu"))
    if not is_openvoice_available(openvoice_root=openvoice_root):
        return None

    try:
        from openvoice.api import ToneColorConverter

        converter_dir = _ensure_converter_checkpoint_dir(checkpoints_dir=checkpoints_dir, openvoice_root=openvoice_root)
        cache_key = (str(converter_dir), openvoice_device)
        if _tone_color_converter is not None and _tone_color_converter_key == cache_key:
            return _tone_color_converter

        converter = ToneColorConverter(
            str(converter_dir / "config.json"),
            device=openvoice_device,
            enable_watermark=False,
        )
        converter.load_ckpt(str(converter_dir / "checkpoint.pth"))
        _tone_color_converter = converter
        _tone_color_converter_key = cache_key
        logger.info("OpenVoice ToneColorConverter loaded from %s", converter_dir)
        return _tone_color_converter
    except Exception as exc:
        logger.warning("OpenVoice ToneColorConverter is unavailable: %s", exc)
        return None


def get_celebrity_reference_path(celebrity: str) -> str | None:
    """Get the absolute path to a local celebrity reference WAV file."""
    rel_path = CELEBRITY_REFERENCES.get(celebrity)
    if rel_path is None:
        return None
    full_path = _PROJECT_ROOT / rel_path
    if full_path.exists():
        return str(full_path)
    logger.info("OpenVoice celebrity reference not found: %s", full_path)
    return None


def _resolve_temp_dir() -> Path:
    raw_dir = os.getenv("OMNISPEECH_OPENVOICE_TEMP_DIR", ".tmp/openvoice")
    temp_dir = _resolve_project_path(raw_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _create_temp_run_dir() -> Path:
    temp_root = _resolve_temp_dir()
    for _ in range(20):
        run_dir = temp_root / f"omnispeech-openvoice-{uuid4().hex}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create an OpenVoice temp directory under: {temp_root}")


def _cleanup_temp_run_dir(run_dir: Path) -> None:
    try:
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
        run_dir.rmdir()
    except OSError:
        logger.debug("Could not clean up OpenVoice temp directory: %s", run_dir, exc_info=True)


def _load_output_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1, dtype=np.float32)

    if source_rate != sample_rate and x.size > 0:
        divisor = math.gcd(int(source_rate), int(sample_rate))
        up = int(sample_rate // divisor)
        down = int(source_rate // divisor)
        x = signal.resample_poly(x, up, down).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _estimate_source_samples(path: Path, sample_rate: int) -> int:
    try:
        info = sf.info(str(path))
    except Exception:
        return 0
    if info.samplerate <= 0 or info.frames <= 0:
        return 0
    return int(round((info.frames / info.samplerate) * sample_rate))


def _fit_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0 or target_samples <= 0:
        return x
    if x.size == target_samples:
        return x

    length_delta = abs(x.size - target_samples)
    if length_delta <= max(128, int(target_samples * 0.015)):
        if x.size > target_samples:
            return x[:target_samples].astype(np.float32)
        return np.pad(x, (0, target_samples - x.size)).astype(np.float32)

    fitted = signal.resample(x, target_samples).astype(np.float32)
    return np.asarray(fitted, dtype=np.float32)


def _stabilize_converted_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x

    x = x - float(np.mean(x))
    try:
        from backend.audio.filtering import apply_post_filter

        x = apply_post_filter(
            x,
            sample_rate,
            speech_band=True,
            declick=True,
            soft_limit=True,
            deess=True,
            use_noisereduce=False,
            use_pedalboard=False,
            deess_reduction_db=2.5,
            ceiling=0.90,
            knee_db=5.0,
        )
    except Exception:
        x = np.asarray(x, dtype=np.float32)

    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms > 1e-7:
        x = x * float(min(1.0, 0.075 / rms))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.90:
        x = x * float(0.90 / peak)
    return np.asarray(x, dtype=np.float32)


def clone_voice_with_openvoice(
    source_path: str,
    reference_path: str,
    sample_rate: int,
    output_path: str | None = None,
    *,
    device: str | None = None,
    openvoice_root: str | os.PathLike[str] | None = None,
    checkpoints_dir: str | os.PathLike[str] | None = None,
) -> np.ndarray | None:
    """Clone tone color using OpenVoice if the package and checkpoints exist."""
    converter = _get_tone_color_converter(
        device=device,
        openvoice_root=openvoice_root,
        checkpoints_dir=checkpoints_dir,
    )
    if converter is None:
        return None

    source = Path(source_path).expanduser().resolve()
    reference = Path(reference_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"OpenVoice source audio not found: {source}")
    if not reference.exists():
        raise FileNotFoundError(f"OpenVoice reference audio not found: {reference}")

    temp_dir: Path | None = None
    if output_path is None:
        temp_dir = _create_temp_run_dir()
        output = temp_dir / "openvoice_output.wav"
    else:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    try:
        reference_se = converter.extract_se(str(reference))
        source_se = converter.extract_se(str(source))
        converter.convert(
            audio_src_path=str(source),
            src_se=source_se,
            tgt_se=reference_se,
            output_path=str(output),
        )

        if not output.exists():
            raise RuntimeError(f"OpenVoice inference did not create an output file: {output}")

        audio = _load_output_audio(output, sample_rate)
        audio = _fit_duration(audio, _estimate_source_samples(source, sample_rate))
        audio = _stabilize_converted_audio(audio, sample_rate)
        return np.asarray(audio, dtype=np.float32)
    except Exception as exc:
        logger.warning("OpenVoice conversion failed: %s", exc)
        return None
    finally:
        if temp_dir is not None:
            _cleanup_temp_run_dir(temp_dir)


def clear_openvoice_cache() -> None:
    global _tone_color_converter, _tone_color_converter_key, _openvoice_available
    _tone_color_converter = None
    _tone_color_converter_key = None
    _openvoice_available = None
