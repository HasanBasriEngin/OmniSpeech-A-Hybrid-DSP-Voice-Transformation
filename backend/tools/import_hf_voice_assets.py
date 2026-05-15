from __future__ import annotations

import argparse
from pathlib import Path

from backend.modules.hf_voice_assets import (
    CURATED_VOICE_ASSETS,
    import_hf_voice_assets,
    plan_hf_voice_asset_imports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import curated RVC/FreeVC assets from Hugging Face.")
    parser.add_argument(
        "--bundle",
        action="append",
        choices=["all", *CURATED_VOICE_ASSETS.keys()],
        default=None,
        help="Asset bundle to import. Repeat for multiple bundles. Default: all.",
    )
    parser.add_argument(
        "--local-root",
        default="models/hf",
        help="Local directory for imported Hugging Face assets.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading files even if they already exist in the HF cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected imports without downloading.",
    )
    return parser


def _bundle_names(raw_bundles: list[str] | None) -> list[str]:
    return raw_bundles or ["all"]


def _print_plan(bundle_names: list[str], local_root: str) -> None:
    for item in plan_hf_voice_asset_imports(bundle_names, local_root=local_root):
        print(f"- {item.name}")
        print(f"  source: {item.repo_type}:{item.repo_id}@{item.revision}")
        print(f"  local: {item.local_dir}")
        print(f"  size: {item.size_hint}")
        print(f"  files: {', '.join(item.allow_patterns)}")
        print(f"  why: {item.selected_for}")
        if item.notes:
            print(f"  note: {item.notes}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    bundle_names = _bundle_names(args.bundle)

    if args.dry_run:
        _print_plan(bundle_names, args.local_root)
        return 0

    manifest = import_hf_voice_assets(
        bundle_names,
        local_root=args.local_root,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
    for asset in manifest["assets"]:
        print(f"Imported {asset['name']} -> {asset['local_dir']}")
    print(f"Manifest: {Path(manifest['manifest_path']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
