from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_FILENAME = "registry.json"


@dataclass(frozen=True)
class FreeVCReferenceProfile:
    category: str
    key: str
    profile_id: str
    reference_path: Path
    consent_required: bool
    consent_owner: str
    license: str
    allow_any_source: bool


def resolve_freevc_profiles_dir(profiles_dir: str | os.PathLike[str] | None = None) -> Path:
    raw_dir = profiles_dir or os.getenv("OMNISPEECH_FREEVC_PROFILES_DIR", "models/freevc_profiles")
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def load_freevc_profile_registry(profiles_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    registry_path = resolve_freevc_profiles_dir(profiles_dir) / _REGISTRY_FILENAME
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid FreeVC profile registry JSON: {registry_path}") from exc

    if not isinstance(registry, dict):
        raise ValueError(f"FreeVC profile registry must be a JSON object: {registry_path}")
    return registry


def _coerce_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"FreeVC profile registry field '{field_name}' must be a boolean.")


def _coerce_string(value: Any, *, field_name: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise ValueError(f"FreeVC profile registry field '{field_name}' must be a string.")


def _resolve_reference_path(profiles_root: Path, raw_path: str, *, category: str, key: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = profiles_root / candidate
    reference_path = candidate.resolve()
    if not reference_path.exists():
        raise FileNotFoundError(
            f"FreeVC reference profile configured for {category} key '{key}' "
            f"but file not found: {reference_path}"
        )
    return reference_path


def get_freevc_reference_profile(
    category: str,
    key: str,
    profiles_dir: str | os.PathLike[str] | None = None,
) -> FreeVCReferenceProfile | None:
    profiles_root = resolve_freevc_profiles_dir(profiles_dir)
    registry = load_freevc_profile_registry(profiles_root)
    category_entries = registry.get(category, {})
    if not isinstance(category_entries, dict):
        raise ValueError(f"FreeVC profile registry field '{category}' must be an object.")

    raw_entry = category_entries.get(key)
    if raw_entry is None:
        return None
    if not isinstance(raw_entry, dict):
        raise ValueError(f"FreeVC profile entry for {category} key '{key}' must be an object.")

    profile_id = _coerce_string(raw_entry.get("profile_id"), field_name=f"{category}.{key}.profile_id", default=key)
    if Path(profile_id).name != profile_id or not profile_id.strip():
        raise ValueError(f"FreeVC profile_id must be a local name, got: {profile_id}")

    reference = raw_entry.get("reference_path")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"FreeVC profile entry for {category} key '{key}' needs reference_path.")

    return FreeVCReferenceProfile(
        category=category,
        key=key,
        profile_id=profile_id.strip(),
        reference_path=_resolve_reference_path(profiles_root, reference, category=category, key=key),
        consent_required=_coerce_bool(
            raw_entry.get("consent_required"),
            field_name=f"{category}.{key}.consent_required",
            default=True,
        ),
        consent_owner=_coerce_string(
            raw_entry.get("consent_owner"),
            field_name=f"{category}.{key}.consent_owner",
            default="",
        ),
        license=_coerce_string(
            raw_entry.get("license"),
            field_name=f"{category}.{key}.license",
            default="private-consent",
        ),
        allow_any_source=_coerce_bool(
            raw_entry.get("allow_any_source"),
            field_name=f"{category}.{key}.allow_any_source",
            default=False,
        ),
    )
