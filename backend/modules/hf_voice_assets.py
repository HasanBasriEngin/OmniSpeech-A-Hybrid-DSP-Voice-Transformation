from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOCAL_ROOT = "models/hf"
_MANIFEST_FILENAME = "voice_assets.manifest.json"

RepoType = Literal["model", "space", "dataset"]
SnapshotDownload = Callable[..., str]


@dataclass(frozen=True)
class HFVoiceAssetBundle:
    name: str
    repo_id: str
    repo_type: RepoType
    local_subdir: str
    allow_patterns: tuple[str, ...]
    selected_for: str
    license: str
    size_hint: str
    revision: str = "main"
    notes: str = ""


@dataclass(frozen=True)
class HFVoiceAssetImport:
    name: str
    repo_id: str
    repo_type: RepoType
    revision: str
    local_dir: str
    allow_patterns: tuple[str, ...]
    selected_for: str
    license: str
    size_hint: str
    notes: str


CURATED_VOICE_ASSETS: dict[str, HFVoiceAssetBundle] = {
    "rvc": HFVoiceAssetBundle(
        name="rvc-core-v2-48k",
        repo_id="AEmotionStudio/rvc-models",
        repo_type="model",
        local_subdir="rvc-core-v2",
        allow_patterns=(
            "README.md",
            "hubert_base.pt",
            "rmvpe.safetensors",
            "pretrained_v2/f0G48k.safetensors",
        ),
        selected_for=(
            "RVC v2 core assets with the 48 kHz F0/RMVPE path, chosen as the "
            "highest-quality RVC foundation path for later licensed target voices."
        ),
        license="MIT",
        size_hint="about 450 MB for the recommended inference subset",
        notes=(
            "This is not a target-speaker voice. OmniSpeech still needs a licensed "
            "or consent-based RVC .pth/.index pair in models/rvc/<model_id>/."
        ),
    ),
    "freevc": HFVoiceAssetBundle(
        name="freevc-24-one-shot",
        repo_id="OlaWod/FreeVC",
        repo_type="space",
        local_subdir="freevc-24",
        allow_patterns=(
            "README.md",
            "app.py",
            "commons.py",
            "mel_processing.py",
            "models.py",
            "modules.py",
            "utils.py",
            "configs/**",
            "speaker_encoder/**",
            "hifigan/**",
            "checkpoints/freevc-24.pth",
            "requirements.txt",
            "p225_001.wav",
            "p226_002.wav",
        ),
        selected_for=(
            "FreeVC 24 kHz one-shot voice conversion, selected for speaker-clone "
            "workflows that have a reference recording instead of a trained RVC voice."
        ),
        license="MIT",
        size_hint="about 500 MB for the FreeVC 24 kHz checkpoint and support files",
        notes=(
            "FreeVC uses the user's reference audio as the target style; use only "
            "recordings you have rights or consent to transform."
        ),
    ),
    "wavlm": HFVoiceAssetBundle(
        name="wavlm-large-content",
        repo_id="microsoft/wavlm-large",
        repo_type="model",
        local_subdir="wavlm-large",
        allow_patterns=(
            "README.md",
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ),
        selected_for=(
            "WavLM Large content encoder used by the Hugging Face FreeVC Space."
        ),
        license="MIT",
        size_hint="about 1.3 GB",
        notes=(
            "If this bundle is not imported, Transformers may try to fetch "
            "microsoft/wavlm-large during first FreeVC inference."
        ),
    ),
}


def resolve_hf_asset_root(local_root: str | Path | None = None) -> Path:
    root = Path(local_root or _DEFAULT_LOCAL_ROOT).expanduser()
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    return root.resolve()


def resolve_bundle_names(bundle_names: Iterable[str] | None = None) -> list[str]:
    raw_names = list(bundle_names or ("all",))
    if any(name == "all" for name in raw_names):
        return list(CURATED_VOICE_ASSETS)

    unknown = sorted(set(raw_names) - set(CURATED_VOICE_ASSETS))
    if unknown:
        allowed = ", ".join(["all", *sorted(CURATED_VOICE_ASSETS)])
        raise ValueError(f"Unknown HF voice asset bundle(s): {', '.join(unknown)}. Allowed: {allowed}")
    return raw_names


def plan_hf_voice_asset_imports(
    bundle_names: Iterable[str] | None = None,
    *,
    local_root: str | Path | None = None,
) -> list[HFVoiceAssetImport]:
    root = resolve_hf_asset_root(local_root)
    imports: list[HFVoiceAssetImport] = []
    for name in resolve_bundle_names(bundle_names):
        bundle = CURATED_VOICE_ASSETS[name]
        imports.append(
            HFVoiceAssetImport(
                name=bundle.name,
                repo_id=bundle.repo_id,
                repo_type=bundle.repo_type,
                revision=bundle.revision,
                local_dir=str(root / bundle.local_subdir),
                allow_patterns=bundle.allow_patterns,
                selected_for=bundle.selected_for,
                license=bundle.license,
                size_hint=bundle.size_hint,
                notes=bundle.notes,
            )
        )
    return imports


def _load_snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF asset import. "
            "Install optional AI dependencies with: pip install -r requirements-ai.txt"
        ) from exc
    return snapshot_download


def import_hf_voice_assets(
    bundle_names: Iterable[str] | None = None,
    *,
    local_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> dict[str, Any]:
    root = resolve_hf_asset_root(local_root)
    root.mkdir(parents=True, exist_ok=True)
    downloader = snapshot_download_fn or _load_snapshot_download()
    planned = plan_hf_voice_asset_imports(bundle_names, local_root=root)

    assets: list[dict[str, Any]] = []
    for item in planned:
        local_dir = Path(item.local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = downloader(
            repo_id=item.repo_id,
            repo_type=item.repo_type,
            revision=item.revision,
            allow_patterns=list(item.allow_patterns),
            local_dir=str(local_dir),
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
            force_download=force_download,
        )
        asset = asdict(item)
        asset["downloaded_path"] = str(Path(downloaded_path).resolve())
        assets.append(asset)

    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "assets": assets,
    }
    manifest_path = root / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
