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
    noise_gate_floor: float = 0.0
    presence_db: float = 0.0
    spectral_tilt_db: float = 0.0
    transform_intensity: float = 1.0
    formant_smoothing: float = 1.0
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
        noise_gate_floor=_float_setting(settings.get("noise_gate_floor"), defaults.noise_gate_floor, lower=0.0, upper=0.08),
        presence_db=_float_setting(settings.get("presence_db"), defaults.presence_db, lower=-6.0, upper=6.0),
        spectral_tilt_db=_float_setting(settings.get("spectral_tilt_db"), defaults.spectral_tilt_db, lower=-6.0, upper=6.0),
        transform_intensity=_float_setting(
            settings.get("transform_intensity"),
            defaults.transform_intensity,
            lower=0.35,
            upper=1.18,
        ),
        formant_smoothing=_float_setting(settings.get("formant_smoothing"), defaults.formant_smoothing, lower=0.35, upper=2.5),
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
    noise_floor = float(post_metrics.get("post_noise_floor_ratio", 0.0))
    harshness = float(post_metrics.get("post_harshness_ratio", 0.0))
    flatness = float(post_metrics.get("post_spectral_flatness", 0.0))
    pitch_stability = float(post_metrics.get("post_pitch_stability", 1.0))

    post_gain_db = settings.post_gain_db
    ceiling = settings.ceiling
    deess_reduction_db = settings.deess_reduction_db
    noise_gate_floor = settings.noise_gate_floor
    presence_db = settings.presence_db
    spectral_tilt_db = settings.spectral_tilt_db
    transform_intensity = settings.transform_intensity
    formant_smoothing = settings.formant_smoothing

    if finite_ratio < 1.0 or clipping > 0.0005 or peak >= ceiling * 0.995:
        post_gain_db -= 0.5
        ceiling -= 0.004
        noise_gate_floor += 0.001
    elif rms < 0.055 and peak < 0.82 and clipping == 0.0:
        post_gain_db += 0.25
    elif rms > 0.18:
        post_gain_db -= 0.25

    if sibilance > 0.32:
        deess_reduction_db += 0.5
        presence_db -= 0.15
    elif sibilance < 0.14 and deess_reduction_db > 3.0:
        deess_reduction_db -= 0.25

    if noise_floor > 0.16:
        noise_gate_floor += 0.002
    elif noise_floor < 0.045:
        noise_gate_floor -= 0.001

    if harshness > 0.48 or flatness > 0.42:
        presence_db -= 0.2
        spectral_tilt_db -= 0.12

    if pitch_stability < 0.68:
        transform_intensity -= 0.025
        formant_smoothing += 0.04

    return DSPProfileSettings(
        post_gain_db=round(float(min(max(post_gain_db, -8.0), 6.0)), 3),
        ceiling=round(float(min(max(ceiling, 0.82), 0.99)), 3),
        knee_db=settings.knee_db,
        deess_reduction_db=round(float(min(max(deess_reduction_db, 0.0), 12.0)), 3),
        noise_gate_floor=round(float(min(max(noise_gate_floor, 0.0), 0.08)), 4),
        presence_db=round(float(min(max(presence_db, -6.0), 6.0)), 3),
        spectral_tilt_db=round(float(min(max(spectral_tilt_db, -6.0), 6.0)), 3),
        transform_intensity=round(float(min(max(transform_intensity, 0.35), 1.18)), 3),
        formant_smoothing=round(float(min(max(formant_smoothing, 0.35), 2.5)), 3),
        use_noisereduce=settings.use_noisereduce,
        use_pedalboard=settings.use_pedalboard,
    )


_FEEDBACK_DELTAS: dict[str, dict[str, float | bool]] = {
    "clean": {"transform_intensity": 0.015, "presence_db": 0.05},
    "muffled": {"presence_db": 0.8, "spectral_tilt_db": 0.35, "post_gain_db": 0.1},
    "harsh": {"deess_reduction_db": 1.0, "presence_db": -0.7, "ceiling": -0.006},
    "robotic": {"transform_intensity": -0.1, "formant_smoothing": 0.22, "deess_reduction_db": 0.35},
    "too_thin": {"transform_intensity": -0.055, "spectral_tilt_db": -0.55, "presence_db": -0.25},
    "too_thick": {"transform_intensity": -0.035, "spectral_tilt_db": 0.5, "presence_db": 0.45},
    "noisy": {"noise_gate_floor": 0.006, "deess_reduction_db": 0.5, "ceiling": -0.004, "use_noisereduce": True},
    "unnatural": {"transform_intensity": -0.075, "formant_smoothing": 0.18, "presence_db": -0.2},
}


def _apply_delta(settings: DSPProfileSettings, delta: dict[str, float | bool]) -> DSPProfileSettings:
    current = asdict(settings)
    for key, value in delta.items():
        if isinstance(value, bool):
            current[key] = value
        elif isinstance(value, (int, float)) and isinstance(current.get(key), (int, float)):
            current[key] = float(current[key]) + float(value)

    return DSPProfileSettings(
        post_gain_db=round(float(min(max(current["post_gain_db"], -8.0), 6.0)), 3),
        ceiling=round(float(min(max(current["ceiling"], 0.82), 0.99)), 3),
        knee_db=round(float(min(max(current["knee_db"], 0.0), 12.0)), 3),
        deess_reduction_db=round(float(min(max(current["deess_reduction_db"], 0.0), 12.0)), 3),
        noise_gate_floor=round(float(min(max(current["noise_gate_floor"], 0.0), 0.08)), 4),
        presence_db=round(float(min(max(current["presence_db"], -6.0), 6.0)), 3),
        spectral_tilt_db=round(float(min(max(current["spectral_tilt_db"], -6.0), 6.0)), 3),
        transform_intensity=round(float(min(max(current["transform_intensity"], 0.35), 1.18)), 3),
        formant_smoothing=round(float(min(max(current["formant_smoothing"], 0.35), 2.5)), 3),
        use_noisereduce=bool(current["use_noisereduce"]),
        use_pedalboard=bool(current["use_pedalboard"]),
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


def apply_user_feedback(
    profile_name: str,
    feedback: str,
    *,
    profiles_dir: str | os.PathLike[str] | None = None,
) -> DSPProfileSettings:
    feedback_key = feedback.strip().lower()
    if feedback_key not in _FEEDBACK_DELTAS:
        allowed = ", ".join(sorted(_FEEDBACK_DELTAS))
        raise ValueError(f"Unsupported DSP feedback '{feedback}'. Allowed: {allowed}")

    registry = load_dsp_profile_registry(profiles_dir)
    profiles = registry.setdefault("profiles", {})
    profile = profiles.setdefault(profile_name, {})
    if not isinstance(profile, dict):
        profile = {}
        profiles[profile_name] = profile

    current = get_dsp_profile_settings(profile_name, profiles_dir)
    updated = _apply_delta(current, _FEEDBACK_DELTAS[feedback_key])

    counts = profile.get("feedback_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    counts[feedback_key] = int(counts.get(feedback_key, 0)) + 1

    profile.update(
        {
            "settings": updated.as_filter_settings(),
            "feedback_counts": counts,
            "last_feedback": feedback_key,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    registry_path = _registry_path(profiles_dir)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return updated
