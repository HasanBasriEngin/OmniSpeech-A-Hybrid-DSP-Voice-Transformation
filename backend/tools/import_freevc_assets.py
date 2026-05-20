from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_FILES = (
    "commons.py",
    "models.py",
    "modules.py",
    "utils.py",
    "mel_processing.py",
)
_VARIANTS: dict[str, dict[str, str]] = {
    "freevc-24": {
        "config": "configs/freevc-24.json",
        "checkpoint": "checkpoints/freevc-24.pth",
        "model_id": "freevc-24-one-shot",
    },
    "freevc": {
        "config": "configs/freevc.json",
        "checkpoint": "checkpoints/freevc.pth",
        "model_id": "freevc-one-shot",
    },
    "freevc-s": {
        "config": "configs/freevc-s.json",
        "checkpoint": "checkpoints/freevc-s.pth",
        "model_id": "freevc-s-one-shot",
    },
}
_SPEAKER_ENCODER_CHECKPOINT = "speaker_encoder/ckpt/pretrained_bak_5805000.pt"


def _resolve_path(raw_path: str | Path, *, base: Path = _PROJECT_ROOT) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _copy_file(source: Path, destination: Path, *, dry_run: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"FreeVC asset not found: {source}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path, *, dry_run: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"FreeVC asset directory not found: {source}")
    if not dry_run:
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _source_file(source_root: Path, relative_path: str, override: str | None) -> Path:
    if override:
        return _resolve_path(override)
    return (source_root / relative_path).resolve()


def _write_manifest(target_dir: Path, manifest: dict[str, Any], *, dry_run: bool) -> Path:
    manifest_path = target_dir / "freevc_assets.manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def import_freevc_assets(
    *,
    source_root: str | Path,
    target_dir: str | Path = "models/hf/freevc-24",
    variant: str = "freevc-24",
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    speaker_encoder_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if variant not in _VARIANTS:
        allowed = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unsupported FreeVC variant: {variant}. Allowed: {allowed}")

    source = _resolve_path(source_root)
    target = _resolve_path(target_dir)
    layout = _VARIANTS[variant]

    copied: list[dict[str, str]] = []

    for relative in _CODE_FILES:
        source_file = source / relative
        target_file = target / relative
        _copy_file(source_file, target_file, dry_run=dry_run)
        copied.append({"source": str(source_file), "target": str(target_file)})

    speaker_encoder_dir = source / "speaker_encoder"
    target_speaker_encoder_dir = target / "speaker_encoder"
    _copy_tree(speaker_encoder_dir, target_speaker_encoder_dir, dry_run=dry_run)
    copied.append({"source": str(speaker_encoder_dir), "target": str(target_speaker_encoder_dir)})

    config_source = _source_file(source, layout["config"], config_path)
    config_target = target / layout["config"]
    _copy_file(config_source, config_target, dry_run=dry_run)
    copied.append({"source": str(config_source), "target": str(config_target)})

    checkpoint_source = _source_file(source, layout["checkpoint"], checkpoint_path)
    checkpoint_target = target / layout["checkpoint"]
    _copy_file(checkpoint_source, checkpoint_target, dry_run=dry_run)
    copied.append({"source": str(checkpoint_source), "target": str(checkpoint_target)})

    speaker_checkpoint_source = _source_file(source, _SPEAKER_ENCODER_CHECKPOINT, speaker_encoder_path)
    speaker_checkpoint_target = target / _SPEAKER_ENCODER_CHECKPOINT
    _copy_file(speaker_checkpoint_source, speaker_checkpoint_target, dry_run=dry_run)
    copied.append({"source": str(speaker_checkpoint_source), "target": str(speaker_checkpoint_target)})

    manifest = {
        "version": 1,
        "source_root": str(source),
        "target_dir": str(target),
        "variant": variant,
        "model_id": layout["model_id"],
        "copied": copied,
        "notes": (
            "FreeVC is a one-shot conversion engine. Use only local reference "
            "audio that you have rights or consent to transform."
        ),
    }
    manifest_path = _write_manifest(target, manifest, dry_run=dry_run)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy locally downloaded OlaWod/FreeVC assets into OmniSpeech's FreeVC layout.",
    )
    parser.add_argument("--source", required=True, help="Path to a local FreeVC repo or extracted asset directory.")
    parser.add_argument("--target", default="models/hf/freevc-24", help="Destination FreeVC asset directory.")
    parser.add_argument("--variant", default="freevc-24", choices=sorted(_VARIANTS))
    parser.add_argument("--config", help="Override path to the FreeVC config JSON.")
    parser.add_argument("--checkpoint", help="Override path to the FreeVC checkpoint .pth.")
    parser.add_argument("--speaker-encoder", help="Override path to pretrained_bak_5805000.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print planned copies.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = import_freevc_assets(
        source_root=args.source,
        target_dir=args.target,
        variant=args.variant,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        speaker_encoder_path=args.speaker_encoder,
        dry_run=args.dry_run,
    )
    for item in manifest["copied"]:
        action = "Would copy" if args.dry_run else "Copied"
        print(f"{action}: {item['source']} -> {item['target']}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
