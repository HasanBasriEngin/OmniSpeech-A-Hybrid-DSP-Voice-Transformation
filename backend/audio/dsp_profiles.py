from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_FILENAME = "registry.json"


@dataclass(frozen=True)
class DSPProfileSettings:
    post_gain_db: float = 0.0
    ceiling: float = 0.96
    knee_db: float = 6.0
    deess_reduction_db: float = 4.5
    use_noisereduce: bool = True
    use_pedalboard: bool = True

    def as_filter_settings(self) -> dict[str, object]:
        return asdict(self)


def resolve_dsp_profiles_dir(profiles_dir: str | os.PathLike[str] | None = None) -> Path:
    raw_dir = profiles_dir or os.getenv("OMNISPEECH_DSP_PROFILES_DIR", "models/dsp_profiles")
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _registry_path(profiles_dir: str | os.PathLike[str] | None = None) -> Path:
    return resolve_dsp_profiles_dir(profiles_dir) / _REGISTRY_FILENAME


def load_dsp_profile_registry(profiles_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    registry_path = _registry_path(profiles_dir)
    if not registry_path.exists():
        return {"version": 1, "profiles": {}}

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid DSP profile registry JSON: {registry_path}") from exc

    if not isinstance(registry, dict):
        raise ValueError(f"DSP profile registry must be a JSON object: {registry_path}")
    profiles = registry.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"DSP profile registry field 'profiles' must be an object: {registry_path}")
    registry.setdefault("version", 1)
    return registry


def _float_setting(raw: Any, default: float, *, lower: float, upper: float) -> float:
    if isinstance(raw, (int, float)):
        return float(min(max(float(raw), lower), upper))
    return default


def _bool_setting(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    return default


def get_dsp_profile_settings(
    profile_name: str,
    profiles_dir: str | os.PathLike[str] | None = None,
) -> DSPProfileSettings:
    registry = load_dsp_profile_registry(profiles_dir)
    profile = registry.get("profiles", {}).get(profile_name, {})
    settings = profile.get("settings", {}) if isinstance(profile, dict) else {}
    if not isinstance(settings, dict):
        settings = {}

    defaults = DSPProfileSettings()
    return DSPProfileSettings(
        post_gain_db=_float_setting(settings.get("post_gain_db"), defaults.post_gain_db, lower=-8.0, upper=6.0),
        ceiling=_float_setting(settings.get("ceiling"), defaults.ceiling, lower=0.82, upper=0.99),
        knee_db=_float_setting(settings.get("knee_db"), defaults.knee_db, lower=0.0, upper=12.0),
        deess_reduction_db=_float_setting(
            settings.get("deess_reduction_db"),
            defaults.deess_reduction_db,
            lower=0.0,
            upper=12.0,
        ),
        use_noisereduce=_bool_setting(settings.get("use_noisereduce"), defaults.use_noisereduce),
        use_pedalboard=_bool_setting(settings.get("use_pedalboard"), defaults.use_pedalboard),
    )


def _ema(previous: float | None, current: float, alpha: float = 0.2) -> float:
    if previous is None:
        return current
    return previous * (1.0 - alpha) + current * alpha


def _next_settings(settings: DSPProfileSettings, post_metrics: dict[str, float]) -> DSPProfileSettings:
    peak = float(post_metrics.get("post_peak", 0.0))
    rms = float(post_metrics.get("post_rms", 0.0))
    clipping = float(post_metrics.get("post_clipping_ratio", 0.0))
    finite_ratio = float(post_metrics.get("post_finite_ratio", 1.0))
    sibilance = float(post_metrics.get("post_sibilance_ratio", 0.0))

    post_gain_db = settings.post_gain_db
    ceiling = settings.ceiling
    deess_reduction_db = settings.deess_reduction_db

    if finite_ratio < 1.0 or clipping > 0.0005 or peak >= ceiling * 0.995:
        post_gain_db -= 0.5
        ceiling -= 0.004
    elif rms < 0.055 and peak < 0.82 and clipping == 0.0:
        post_gain_db += 0.25
    elif rms > 0.18:
        post_gain_db -= 0.25

    if sibilance > 0.32:
        deess_reduction_db += 0.5
    elif sibilance < 0.14 and deess_reduction_db > 3.0:
        deess_reduction_db -= 0.25

    return DSPProfileSettings(
        post_gain_db=round(float(min(max(post_gain_db, -8.0), 6.0)), 3),
        ceiling=round(float(min(max(ceiling, 0.82), 0.99)), 3),
        knee_db=settings.knee_db,
        deess_reduction_db=round(float(min(max(deess_reduction_db, 0.0), 12.0)), 3),
        use_noisereduce=settings.use_noisereduce,
        use_pedalboard=settings.use_pedalboard,
    )


def update_dsp_profile(
    profile_name: str,
    *,
    pre_metrics: dict[str, float],
    post_metrics: dict[str, float],
    engine: str,
    profiles_dir: str | os.PathLike[str] | None = None,
) -> DSPProfileSettings:
    registry = load_dsp_profile_registry(profiles_dir)
    profiles = registry.setdefault("profiles", {})
    profile = profiles.setdefault(profile_name, {})
    if not isinstance(profile, dict):
        profile = {}
        profiles[profile_name] = profile

    current = get_dsp_profile_settings(profile_name, profiles_dir)
    updated = _next_settings(current, post_metrics)

    runs = int(profile.get("updated_from_runs", 0)) + 1
    metrics_ema = profile.get("metrics_ema", {})
    if not isinstance(metrics_ema, dict):
        metrics_ema = {}
    for key, value in {**pre_metrics, **post_metrics}.items():
        if isinstance(value, (int, float)):
            previous = metrics_ema.get(key)
            metrics_ema[key] = round(_ema(previous if isinstance(previous, (int, float)) else None, float(value)), 6)

    profile.update(
        {
            "settings": updated.as_filter_settings(),
            "updated_from_runs": runs,
            "metrics_ema": metrics_ema,
            "last_engine": engine,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    registry_path = _registry_path(profiles_dir)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return updated
