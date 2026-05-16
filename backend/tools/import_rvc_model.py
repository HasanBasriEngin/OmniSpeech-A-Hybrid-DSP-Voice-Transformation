from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from backend.modules.rvc_adapter import load_rvc_registry, resolve_models_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a trained local RVC model into OmniSpeech registry layout.",
    )
    parser.add_argument("--model-id", required=True, help="Local model directory name under models/rvc.")
    parser.add_argument("--pth", required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--index", help="Optional path to the trained .index file.")
    parser.add_argument("--category", required=True, choices=["gender_age", "celebrity"])
    parser.add_argument("--key", required=True, help="Registry key such as male_to_female or michael_jackson.")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--consent-required", default="true", choices=["true", "false"])
    parser.add_argument("--consent-owner", default="")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--allow-any-source", default="false", choices=["true", "false"])
    parser.add_argument("--models-dir", help="Override models/rvc root directory.")
    return parser


def _parse_bool_flag(value: str) -> bool:
    return value.strip().lower() == "true"


def _validate_model_id(model_id: str) -> str:
    cleaned = model_id.strip()
    if not cleaned:
        raise ValueError("model_id must be a non-empty directory name.")
    if Path(cleaned).name != cleaned:
        raise ValueError(f"model_id must be a local directory name, got: {model_id}")
    return cleaned


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _registry_entry_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "pitch": int(args.pitch),
        "index_rate": float(args.index_rate),
        "consent_required": _parse_bool_flag(args.consent_required),
        "consent_owner": args.consent_owner,
        "license": args.license,
        "allow_any_source": _parse_bool_flag(args.allow_any_source),
    }


def import_rvc_model(args: argparse.Namespace) -> Path:
    args.model_id = _validate_model_id(args.model_id)

    pth_path = Path(args.pth).expanduser().resolve()
    if not pth_path.exists():
        raise FileNotFoundError(f"RVC .pth file not found: {pth_path}")

    index_path = None
    if args.index:
        index_path = Path(args.index).expanduser().resolve()
        if not index_path.exists():
            raise FileNotFoundError(f"RVC .index file not found: {index_path}")

    models_root = resolve_models_dir(args.models_dir)
    models_root.mkdir(parents=True, exist_ok=True)

    model_dir = models_root / args.model_id
    target_pth = model_dir / f"{args.model_id}.pth"
    _copy_artifact(pth_path, target_pth)

    if index_path is not None:
        target_index = model_dir / f"{args.model_id}.index"
        _copy_artifact(index_path, target_index)

    registry_path = models_root / "registry.json"
    registry = load_rvc_registry(models_root)
    category_entries = registry.get(args.category, {})
    if not isinstance(category_entries, dict):
        raise ValueError(f"RVC registry field '{args.category}' must be an object.")
    registry[args.category] = category_entries
    category_entries[args.key] = _registry_entry_from_args(args)

    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    return registry_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    registry_path = import_rvc_model(args)
    print(f"Imported model '{args.model_id}' into {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
