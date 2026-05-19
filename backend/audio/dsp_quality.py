from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from backend.audio.dsp_metrics import measure_audio_health
from backend.audio.dsp_profiles import DSPProfileSettings
from backend.audio.filtering import post_filter_voice


@dataclass(frozen=True)
class DSPQualityCandidate:
    index: int
    settings: DSPProfileSettings
    audio: np.ndarray
    metrics: dict[str, float]
    score: float


@dataclass(frozen=True)
class DSPQualityOptimizationResult:
    audio: np.ndarray
    metrics: dict[str, float]
    selected_settings: DSPProfileSettings
    selected_index: int
    selected_score: float
    baseline_score: float
    candidate_count: int


def _clamp(value: object, lower: float, upper: float) -> float:
    if isinstance(value, (int, float)):
        return float(min(max(float(value), lower), upper))
    return lower


def _settings_from_dict(values: dict[str, object]) -> DSPProfileSettings:
    defaults = DSPProfileSettings()
    return DSPProfileSettings(
        post_gain_db=round(_clamp(values.get("post_gain_db", defaults.post_gain_db), -8.0, 6.0), 3),
        ceiling=round(_clamp(values.get("ceiling", defaults.ceiling), 0.82, 0.99), 3),
        knee_db=round(_clamp(values.get("knee_db", defaults.knee_db), 0.0, 12.0), 3),
        deess_reduction_db=round(_clamp(values.get("deess_reduction_db", defaults.deess_reduction_db), 0.0, 12.0), 3),
        noise_gate_floor=round(_clamp(values.get("noise_gate_floor", defaults.noise_gate_floor), 0.0, 0.08), 4),
        presence_db=round(_clamp(values.get("presence_db", defaults.presence_db), -6.0, 6.0), 3),
        spectral_tilt_db=round(_clamp(values.get("spectral_tilt_db", defaults.spectral_tilt_db), -6.0, 6.0), 3),
        transform_intensity=round(_clamp(values.get("transform_intensity", defaults.transform_intensity), 0.35, 1.18), 3),
        formant_smoothing=round(_clamp(values.get("formant_smoothing", defaults.formant_smoothing), 0.35, 2.5), 3),
        use_noisereduce=bool(values.get("use_noisereduce", defaults.use_noisereduce)),
        use_pedalboard=bool(values.get("use_pedalboard", defaults.use_pedalboard)),
    )


def _with_delta(settings: DSPProfileSettings, **deltas: float | bool) -> DSPProfileSettings:
    values = asdict(settings)
    for key, delta in deltas.items():
        if isinstance(delta, bool):
            values[key] = delta
        elif isinstance(values.get(key), (int, float)):
            values[key] = float(values[key]) + float(delta)
    return _settings_from_dict(values)


def build_quality_candidate_settings(settings: DSPProfileSettings) -> list[DSPProfileSettings]:
    return [
        settings,
        _with_delta(
            settings,
            post_gain_db=-0.35,
            ceiling=-0.006,
            knee_db=1.0,
            deess_reduction_db=1.2,
            noise_gate_floor=0.0035,
            presence_db=-0.35,
            spectral_tilt_db=-0.22,
        ),
        _with_delta(
            settings,
            post_gain_db=-0.12,
            ceiling=-0.003,
            knee_db=0.5,
            deess_reduction_db=0.55,
            noise_gate_floor=0.0012,
            presence_db=0.25,
            spectral_tilt_db=0.12,
        ),
    ]


def _filter_settings_for_engine(settings: DSPProfileSettings, *, neural_safe_filter: bool) -> dict[str, object]:
    values = settings.as_filter_settings()
    if neural_safe_filter:
        values.update(
            {
                "use_noisereduce": False,
                "use_pedalboard": False,
                "post_gain_db": min(float(settings.post_gain_db), 0.0),
                "deess_reduction_db": min(float(settings.deess_reduction_db), 2.0),
            }
        )
    return values


def score_quality_candidate(pre_metrics: dict[str, float], post_metrics: dict[str, float]) -> float:
    artifact = float(post_metrics.get("post_artifact_score", 0.0))
    spectrogram_artifact = float(post_metrics.get("post_spectrogram_artifact_score", 0.0))
    finite_ratio = float(post_metrics.get("post_finite_ratio", 1.0))
    peak = float(post_metrics.get("post_peak", 0.0))
    rms = max(float(post_metrics.get("post_rms", 0.0)), 1e-7)
    pre_rms = max(float(pre_metrics.get("pre_post_rms", rms)), 1e-7)
    dc_offset = abs(float(post_metrics.get("post_dc_offset", 0.0)))
    pitch_stability = float(post_metrics.get("post_pitch_stability", 1.0))
    spectral_shift = 0.0
    for band in ("body", "vocal", "presence", "harsh", "sibilance", "air"):
        before = float(pre_metrics.get(f"pre_post_spectrogram_{band}_ratio", 0.0))
        after = float(post_metrics.get(f"post_spectrogram_{band}_ratio", 0.0))
        spectral_shift += abs(after - before)

    target_rms = float(np.clip(pre_rms, 0.045, 0.16))
    loudness_ratio = rms / target_rms
    rms_penalty = min(abs(np.log2(max(loudness_ratio, 1e-4))) * 0.11, 0.32)
    quiet_penalty = 0.18 if rms < 0.025 else 0.0
    hot_penalty = 0.18 if peak > 0.975 else 0.0
    finite_penalty = min(max(1.0 - finite_ratio, 0.0), 1.0)
    dc_penalty = min(dc_offset / 0.03, 1.0) * 0.08
    pitch_penalty = max(0.0, 0.72 - pitch_stability) * 0.12
    spectral_shift_penalty = min(spectral_shift * 0.16, 0.22)
    spectrogram_penalty = spectrogram_artifact * 0.20

    score = (
        artifact
        + rms_penalty
        + quiet_penalty
        + hot_penalty
        + finite_penalty
        + dc_penalty
        + pitch_penalty
        + spectral_shift_penalty
        + spectrogram_penalty
    )
    return float(round(max(score, 0.0), 6))


def optimize_post_filter_quality(
    audio: np.ndarray,
    sample_rate: int,
    settings: DSPProfileSettings,
    *,
    engine: str,
    pre_metrics: dict[str, float] | None = None,
    filter_func: Callable[..., np.ndarray] | None = None,
) -> DSPQualityOptimizationResult:
    neural_safe_filter = engine in {"freevc", "rvc"}
    base_pre_metrics = pre_metrics or measure_audio_health(audio, sample_rate, prefix="pre_post_")
    apply_filter = filter_func or post_filter_voice
    candidates: list[DSPQualityCandidate] = []
    for index, candidate_settings in enumerate(build_quality_candidate_settings(settings)):
        filtered = apply_filter(
            audio,
            sample_rate,
            settings=_filter_settings_for_engine(candidate_settings, neural_safe_filter=neural_safe_filter),
        )
        metrics = measure_audio_health(filtered, sample_rate, prefix="post_")
        score = score_quality_candidate(base_pre_metrics, metrics)
        candidates.append(
            DSPQualityCandidate(
                index=index,
                settings=candidate_settings,
                audio=filtered,
                metrics=metrics,
                score=score,
            )
        )

    selected = min(candidates, key=lambda candidate: candidate.score)
    baseline_score = candidates[0].score
    metrics = {
        **selected.metrics,
        "dsp_quality_optimizer_applied": 1.0,
        "dsp_quality_candidates": float(len(candidates)),
        "dsp_quality_selected_candidate": float(selected.index),
        "dsp_quality_selected_score": float(selected.score),
        "dsp_quality_baseline_score": float(baseline_score),
        "dsp_quality_score_improvement": float(round(max(0.0, baseline_score - selected.score), 6)),
    }
    for candidate in candidates:
        metrics[f"dsp_quality_candidate_{candidate.index}_score"] = float(candidate.score)
        metrics[f"dsp_quality_candidate_{candidate.index}_artifact"] = float(candidate.metrics.get("post_artifact_score", 0.0))
        metrics[f"dsp_quality_candidate_{candidate.index}_spectrogram"] = float(
            candidate.metrics.get("post_spectrogram_artifact_score", 0.0)
        )

    return DSPQualityOptimizationResult(
        audio=np.asarray(selected.audio, dtype=np.float32),
        metrics=metrics,
        selected_settings=selected.settings,
        selected_index=selected.index,
        selected_score=selected.score,
        baseline_score=baseline_score,
        candidate_count=len(candidates),
    )
