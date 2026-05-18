from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
from scipy import signal
import soundfile as sf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FREEVC_SUPPORT_FILES = (
    "speaker_encoder/ckpt/pretrained_bak_5805000.pt",
    "speaker_encoder/voice_encoder.py",
    "commons.py",
    "models.py",
    "modules.py",
    "utils.py",
    "mel_processing.py",
)
_FREEVC_MODEL_CANDIDATES = (
    ("freevc-24-one-shot", "configs/freevc-24.json", "checkpoints/freevc-24.pth", 24000),
    ("freevc-one-shot", "configs/freevc.json", "checkpoints/freevc.pth", 16000),
    ("freevc-s-one-shot", "configs/freevc-s.json", "checkpoints/freevc-s.pth", 16000),
)


@dataclass(frozen=True)
class FreeVCModelConfig:
    model_id: str
    assets_dir: Path
    checkpoint_path: Path
    config_path: Path
    speaker_encoder_path: Path
    wavlm_model: str
    output_sample_rate: int = 24000


@dataclass(frozen=True)
class FreeVCConversionResult:
    audio: np.ndarray
    config: FreeVCModelConfig


def _resolve_project_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def resolve_freevc_assets_dir(assets_dir: str | os.PathLike[str] | None = None) -> Path:
    raw_dir = assets_dir or os.getenv("OMNISPEECH_FREEVC_ASSETS_DIR", "models/hf/freevc-24")
    return _resolve_project_path(raw_dir)


def resolve_wavlm_model(wavlm_model: str | os.PathLike[str] | None = None) -> str:
    raw_model = str(wavlm_model or os.getenv("OMNISPEECH_WAVLM_MODEL", "")).strip()
    if raw_model:
        path = Path(raw_model).expanduser()
        path_like = (
            path.exists()
            or "\\" in raw_model
            or raw_model.startswith(("/", ".", "~"))
            or raw_model.startswith(("models/", "models\\", ".tmp/", ".tmp\\"))
        )
        if path_like:
            return str(_resolve_project_path(raw_model))
        return raw_model

    local_wavlm = _PROJECT_ROOT / "models" / "hf" / "wavlm-large"
    if (local_wavlm / "config.json").exists():
        return str(local_wavlm.resolve())
    return "microsoft/wavlm-large"


def get_freevc_config(
    assets_dir: str | os.PathLike[str] | None = None,
    *,
    wavlm_model: str | os.PathLike[str] | None = None,
) -> FreeVCModelConfig | None:
    root = resolve_freevc_assets_dir(assets_dir)
    if not root.exists():
        return None

    missing = [relative for relative in _FREEVC_SUPPORT_FILES if not (root / relative).exists()]
    if missing:
        logger.debug("FreeVC assets are incomplete under %s; missing: %s", root, ", ".join(missing))
        return None

    for model_id, config_relative, checkpoint_relative, output_sample_rate in _FREEVC_MODEL_CANDIDATES:
        config_path = root / config_relative
        checkpoint_path = root / checkpoint_relative
        if config_path.exists() and checkpoint_path.exists():
            return FreeVCModelConfig(
                model_id=model_id,
                assets_dir=root,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                speaker_encoder_path=root / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt",
                wavlm_model=resolve_wavlm_model(wavlm_model),
                output_sample_rate=output_sample_rate,
            )

    logger.debug(
        "FreeVC support files exist under %s, but no supported config/checkpoint pair was found.",
        root,
    )
    return None


def _resolve_temp_dir() -> Path:
    raw_dir = os.getenv("OMNISPEECH_FREEVC_TEMP_DIR", ".tmp/freevc")
    path = _resolve_project_path(raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_temp_run_dir() -> Path:
    temp_root = _resolve_temp_dir()
    for _ in range(20):
        run_dir = temp_root / f"omnispeech-freevc-{uuid4().hex}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a FreeVC temp directory under: {temp_root}")


def _cleanup_temp_run_dir(run_dir: Path) -> None:
    try:
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
        run_dir.rmdir()
    except OSError:
        logger.debug("Could not clean up FreeVC temp directory: %s", run_dir, exc_info=True)


def _load_output_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    if source_rate != sample_rate and x.size > 0:
        gcd = math.gcd(int(source_rate), int(sample_rate))
        up = int(sample_rate // gcd)
        down = int(source_rate // gcd)
        x = signal.resample_poly(x, up, down).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _estimate_source_samples(path: Path, sample_rate: int) -> int:
    try:
        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return int(round((info.frames / info.samplerate) * sample_rate))
    except Exception:
        pass
    return 0


def _fit_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0 or target_samples <= 0:
        return x

    length_delta = abs(x.size - target_samples)
    if length_delta <= max(128, int(target_samples * 0.015)):
        if x.size > target_samples:
            return x[:target_samples].astype(np.float32)
        return np.pad(x, (0, target_samples - x.size)).astype(np.float32)

    try:
        import librosa

        rate = float(x.size / target_samples)
        stretched = librosa.effects.time_stretch(x.astype(np.float32), rate=rate)
        x = np.asarray(stretched, dtype=np.float32)
    except Exception:
        x = signal.resample(x, target_samples).astype(np.float32)

    if x.size > target_samples:
        x = x[:target_samples]
    elif x.size < target_samples:
        x = np.pad(x, (0, target_samples - x.size))
    return np.asarray(x, dtype=np.float32)


def _format_runner_error(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    tail = stderr[-1800:] or stdout[-1800:] or "no output"
    return f"FreeVC inference failed with exit code {completed.returncode}: {tail}"


def convert_file_with_freevc(
    input_path: str,
    reference_path: str,
    sample_rate: int,
    *,
    assets_dir: str | os.PathLike[str] | None = None,
    device: str | None = None,
    wavlm_model: str | os.PathLike[str] | None = None,
) -> FreeVCConversionResult | None:
    config = get_freevc_config(assets_dir=assets_dir, wavlm_model=wavlm_model)
    if config is None:
        return None

    source_path = Path(input_path).expanduser().resolve()
    target_path = Path(reference_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"FreeVC source audio not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"FreeVC reference audio not found: {target_path}")

    freevc_device = device or os.getenv("OMNISPEECH_FREEVC_DEVICE", os.getenv("OMNISPEECH_RVC_DEVICE", "cpu"))
    temp_dir = _create_temp_run_dir()
    try:
        output_path = temp_dir / "freevc_output.wav"
        command = [
            sys.executable,
            "-m",
            "backend.tools.run_freevc_inference",
            "--assets-dir",
            str(config.assets_dir),
            "--config",
            str(config.config_path),
            "--checkpoint",
            str(config.checkpoint_path),
            "--speaker-encoder",
            str(config.speaker_encoder_path),
            "--wavlm-model",
            config.wavlm_model,
            "--input",
            str(source_path),
            "--reference",
            str(target_path),
            "--output",
            str(output_path),
            "--device",
            freevc_device,
        ]
        completed = subprocess.run(
            command,
            cwd=str(_PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(_format_runner_error(completed))
        if not output_path.exists():
            raise RuntimeError(f"FreeVC inference did not create an output file: {output_path}")

        audio = _load_output_audio(output_path, sample_rate)
        audio = _fit_duration(audio, _estimate_source_samples(source_path, sample_rate))
    finally:
        _cleanup_temp_run_dir(temp_dir)

    return FreeVCConversionResult(audio=np.asarray(audio, dtype=np.float32), config=config)
