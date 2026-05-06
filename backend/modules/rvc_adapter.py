from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import signal
import soundfile as sf
import torch

# Patch torch.load for PyTorch >= 2.6 to prevent weights_only=True issues with fairseq
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_FILENAME = "registry.json"
_RVC_INFERENCE_CLASS: Any | None = None
_RVC_INSTANCE_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class RVCModelConfig:
    mode: str
    model_id: str
    model_path: Path
    index_path: Path | None
    pitch: int
    index_rate: float


@dataclass(frozen=True)
class RVCConversionResult:
    audio: np.ndarray
    config: RVCModelConfig


def resolve_models_dir(models_dir: str | os.PathLike[str] | None = None) -> Path:
    raw_dir = models_dir or os.getenv("OMNISPEECH_RVC_MODELS_DIR", "models/rvc")
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _resolve_temp_dir() -> Path:
    raw_dir = os.getenv("OMNISPEECH_RVC_TEMP_DIR", ".tmp/rvc")
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _create_temp_run_dir() -> Path:
    temp_root = _resolve_temp_dir()
    for _ in range(20):
        run_dir = temp_root / f"omnispeech-rvc-{uuid4().hex}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create an RVC temp directory under: {temp_root}")


def _cleanup_temp_run_dir(run_dir: Path) -> None:
    try:
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
        run_dir.rmdir()
    except OSError:
        logger.debug("Could not clean up RVC temp directory: %s", run_dir, exc_info=True)


def load_rvc_registry(models_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    registry_path = resolve_models_dir(models_dir) / _REGISTRY_FILENAME
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid RVC registry JSON: {registry_path}") from exc

    if not isinstance(registry, dict):
        raise ValueError(f"RVC registry must be a JSON object: {registry_path}")
    return registry


def get_gender_age_rvc_config(mode: str, models_dir: str | os.PathLike[str] | None = None) -> RVCModelConfig | None:
    models_root = resolve_models_dir(models_dir)
    registry = load_rvc_registry(models_root)
    gender_age = registry.get("gender_age", {})
    if not isinstance(gender_age, dict):
        raise ValueError("RVC registry field 'gender_age' must be an object.")

    raw_entry = gender_age.get(mode)
    if raw_entry is None:
        return None
    if not isinstance(raw_entry, dict):
        raise ValueError(f"RVC registry entry for gender_age mode '{mode}' must be an object.")

    model_id = raw_entry.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"RVC registry entry for gender_age mode '{mode}' needs a non-empty model_id.")
    model_id = model_id.strip()
    if Path(model_id).name != model_id:
        raise ValueError(f"RVC model_id must be a local directory name, got: {model_id}")

    model_dir = models_root / model_id
    model_path = model_dir / f"{model_id}.pth"
    if not model_path.exists():
        logger.warning(f"RVC model configured for gender_age mode '{mode}' but file not found: {model_path}. Falling back to DSP.")
        return None

    index_path = model_dir / f"{model_id}.index"
    if not index_path.exists():
        index_path = None

    return RVCModelConfig(
        mode=mode,
        model_id=model_id,
        model_path=model_path,
        index_path=index_path,
        pitch=int(raw_entry.get("pitch", 0)),
        index_rate=float(raw_entry.get("index_rate", 0.5)),
    )


def get_celebrity_rvc_config(celebrity: str, models_dir: str | os.PathLike[str] | None = None) -> RVCModelConfig | None:
    models_root = resolve_models_dir(models_dir)
    registry = load_rvc_registry(models_root)
    celeb_reg = registry.get("celebrity", {})
    if not isinstance(celeb_reg, dict):
        return None

    raw_entry = celeb_reg.get(celebrity)
    if raw_entry is None or not isinstance(raw_entry, dict):
        return None

    model_id = raw_entry.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"RVC registry entry for celebrity '{celebrity}' needs a non-empty model_id.")
    model_id = model_id.strip()

    model_dir = models_root / model_id
    model_path = model_dir / f"{model_id}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"RVC model configured for celebrity '{celebrity}' but file not found: {model_path}")

    index_path = model_dir / f"{model_id}.index"
    if not index_path.exists():
        index_path = None

    return RVCModelConfig(
        mode=celebrity,
        model_id=model_id,
        model_path=model_path,
        index_path=index_path,
        pitch=int(raw_entry.get("pitch", 0)),
        index_rate=float(raw_entry.get("index_rate", 0.5)),
    )



def _load_rvc_inference_class() -> Any:
    global _RVC_INFERENCE_CLASS
    if _RVC_INFERENCE_CLASS is None:
        try:
            from rvc_python.infer import RVCInference
        except ImportError as exc:
            raise RuntimeError(
                "RVC model is configured, but rvc-python is not installed. "
                "Install optional AI dependencies with: pip install -r requirements-ai.txt"
            ) from exc
        _RVC_INFERENCE_CLASS = RVCInference
    return _RVC_INFERENCE_CLASS


def _load_rvc_engine(config: RVCModelConfig, device: str) -> Any:
    cache_key = (str(config.model_path), device)
    cached = _RVC_INSTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    inference_class = _load_rvc_inference_class()
    try:
        engine = inference_class(device=device)
    except TypeError:
        engine = inference_class()

    if not hasattr(engine, "load_model"):
        raise RuntimeError("RVCInference object does not expose load_model().")

    try:
        engine.load_model(
            str(config.model_path),
            index_path=str(config.index_path) if config.index_path is not None else "",
        )
    except TypeError:
        engine.load_model(str(config.model_path))
    _RVC_INSTANCE_CACHE[cache_key] = engine
    return engine


def _apply_inference_params(engine: Any, config: RVCModelConfig) -> bool:
    if not hasattr(engine, "set_params"):
        return False

    try:
        engine.set_params(f0up_key=config.pitch, index_rate=config.index_rate)
    except TypeError:
        return False
    return True


def _infer_file(engine: Any, input_path: Path, output_path: Path, config: RVCModelConfig) -> None:
    _apply_inference_params(engine, config)

    if not engine.current_model:
        raise ValueError("RVC model not loaded properly.")

    model_info = engine.models[engine.current_model]
    file_index = model_info.get("index", "")

    # Bypass the buggy engine.infer_file that crashes on wavfile.write
    wav_opt = engine.vc.vc_single(
        sid=0,
        input_audio_path=str(input_path),
        f0_up_key=engine.f0up_key,
        f0_method=engine.f0method,
        file_index=str(config.index_path) if config.index_path else file_index,
        index_rate=engine.index_rate,
        filter_radius=engine.filter_radius,
        resample_sr=engine.resample_sr,
        rms_mix_rate=engine.rms_mix_rate,
        protect=engine.protect,
        f0_file="",
        file_index2=""
    )

    if isinstance(wav_opt, tuple) and len(wav_opt) == 2 and isinstance(wav_opt[1], tuple):
        tgt_sr = wav_opt[1][0]
        audio_data = wav_opt[1][1]
    elif isinstance(wav_opt, np.ndarray):
        tgt_sr = engine.vc.tgt_sr
        audio_data = wav_opt
    else:
        # If vc_single failed, it usually returns an error string
        raise RuntimeError(f"RVC inference failed. vc_single returned: {wav_opt}")

    if audio_data is None:
        raise RuntimeError("RVC inference failed to produce audio data.")

    import soundfile as sf
    sf.write(str(output_path), audio_data, tgt_sr)


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


def convert_gender_age_with_rvc(
    input_path: str,
    mode: str,
    sample_rate: int,
    *,
    models_dir: str | os.PathLike[str] | None = None,
    device: str | None = None,
) -> RVCConversionResult | None:
    config = get_gender_age_rvc_config(mode, models_dir=models_dir)
    if config is None:
        return None

    rvc_device = device or os.getenv("OMNISPEECH_RVC_DEVICE", "cpu")
    source_path = Path(input_path).expanduser().resolve()

    temp_dir = _create_temp_run_dir()
    try:
        output_path = temp_dir / "rvc_output.wav"
        engine = _load_rvc_engine(config, rvc_device)
        _infer_file(engine, source_path, output_path, config)

        if not output_path.exists():
            raise RuntimeError(f"RVC inference did not create an output file: {output_path}")

        audio = _load_output_audio(output_path, sample_rate)
    finally:
        _cleanup_temp_run_dir(temp_dir)

    return RVCConversionResult(audio=np.asarray(audio, dtype=np.float32), config=config)


def convert_celebrity_with_rvc(
    input_path: str,
    celebrity: str,
    sample_rate: int,
    *,
    models_dir: str | os.PathLike[str] | None = None,
    device: str | None = None,
) -> RVCConversionResult | None:
    config = get_celebrity_rvc_config(celebrity, models_dir=models_dir)
    if config is None:
        return None

    rvc_device = device or os.getenv("OMNISPEECH_RVC_DEVICE", "cpu")
    source_path = Path(input_path).expanduser().resolve()

    temp_dir = _create_temp_run_dir()
    try:
        output_path = temp_dir / "rvc_output.wav"
        engine = _load_rvc_engine(config, rvc_device)
        _infer_file(engine, source_path, output_path, config)

        if not output_path.exists():
            raise RuntimeError(f"RVC inference did not create an output file: {output_path}")

        audio = _load_output_audio(output_path, sample_rate)
    finally:
        _cleanup_temp_run_dir(temp_dir)

    return RVCConversionResult(audio=np.asarray(audio, dtype=np.float32), config=config)



def clear_rvc_engine_cache() -> None:
    _RVC_INSTANCE_CACHE.clear()
