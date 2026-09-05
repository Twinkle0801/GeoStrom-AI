"""Reads the already-committed Phase 2/5/6/7/8 benchmark report JSON files
from `ml/reports/` and reshapes them into `ModelPerformanceResponse`.

This is a pure READ of static, already-computed, already-committed files --
no model is retrained, no metric is recomputed, no ML code is imported
(matches `app/main.py`'s existing invariant: this backend never imports
`ml.geostrom_ml`, `torch`, or `lightgbm`). If a report file is missing, the
corresponding task's comparison is simply omitted from the response rather
than fabricated -- see `_load_json`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from app.schemas.analytics import ModelMetricEntry, ModelPerformanceResponse, TaskComparison

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
ML_REPORTS_DIR = REPO_ROOT / "ml" / "reports"

# docs/PHASE_7_INTENSITY_PREDICTION.md / PHASE_8_TRACK_PREDICTION.md's own
# established, honest recommendation -- restated here, never re-derived.
INTENSITY_RECOMMENDED = "intensity_lightgbm_v1"
TRACK_RECOMMENDED = "track_cliper_v1"
CLASSIFICATION_RECOMMENDED = "logistic_regression"

_DISPLAY_NAMES = {
    "intensity_persistence_v1": "Persistence", "intensity_ridge_v1": "Ridge",
    "intensity_lightgbm_v1": "LightGBM", "intensity_gru_v1": "GRU (absolute)",
    "intensity_gru_delta_v1": "GRU (Δwind)",
    "track_persistence_v1": "Persistence", "track_cliper_v1": "CLIPER-style Ridge",
    "track_lightgbm_v1": "LightGBM", "track_gru_v1": "GRU",
    "majority_class": "Majority Class", "logistic_regression": "Logistic Regression",
    "lightgbm": "LightGBM", "resnet18": "ResNet-18", "small_cnn": "Small CNN",
}

_TIERS: dict[str, str] = {
    "intensity_persistence_v1": "baseline", "intensity_ridge_v1": "baseline",
    "intensity_lightgbm_v1": "baseline", "intensity_gru_v1": "exploratory",
    "intensity_gru_delta_v1": "exploratory",
    "track_persistence_v1": "baseline", "track_cliper_v1": "baseline",
    "track_lightgbm_v1": "baseline", "track_gru_v1": "exploratory",
    "majority_class": "floor", "logistic_regression": "baseline", "lightgbm": "baseline",
    "resnet18": "exploratory", "small_cnn": "exploratory",
}


def _load_json(name: str) -> dict | None:
    path = ML_REPORTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(name: str, version: str, recommended: str,
           metrics_by_horizon: dict | None = None, metrics: dict | None = None) -> ModelMetricEntry:
    return ModelMetricEntry(
        model_name=name, display_name=_DISPLAY_NAMES.get(name, name),
        model_version=version, tier=_TIERS.get(name, "exploratory"),
        is_recommended=(name == recommended),
        metrics_by_horizon=metrics_by_horizon, metrics=metrics,
    )


def _sequence_comparison(report: dict, task: str, recommended: str, metric_key: str) -> TaskComparison:
    """Shared shape for intensity (phase7) and track (phase8) reports --
    both already contain a full `comparison_vs_phase2[horizon][model] ->
    {metric...}` table, built in their own training scripts (never
    recomputed here)."""
    comparison = report["comparison_vs_phase2"]
    horizons = sorted(int(h) for h in comparison.keys())
    model_names = sorted({name for row in comparison.values() for name in row})

    by_model: dict[str, dict[str, dict]] = {name: {} for name in model_names}
    for h, row in comparison.items():
        for name, metrics in row.items():
            by_model[name][str(int(h))] = metrics

    models = [
        _entry(name, "v1", recommended, metrics_by_horizon=by_model[name])
        for name in model_names
    ]
    models.sort(key=lambda m: (m.tier != "baseline", m.model_name))

    return TaskComparison(
        task=task, horizons_h=horizons, models=models,
        recommended_model=_DISPLAY_NAMES.get(recommended, recommended),
        methodology_note=(
            f"{metric_key} per forecast horizon, evaluated once on the frozen, storm-disjoint "
            f"test split (docs/PHASE_2_FORECASTING_BASELINES.md). GRU rows are exploratory "
            f"research results (docs/PHASE_7_INTENSITY_PREDICTION.md / "
            f"docs/PHASE_8_TRACK_PREDICTION.md) and are not the recommended production model."
        ),
    )


def _classification_comparison(baseline: dict | None, resnet18: dict | None,
                               small_cnn: dict | None) -> TaskComparison:
    models: list[ModelMetricEntry] = []
    if baseline is not None:
        for name, result in baseline["models"].items():
            test = result.get("test", {})
            metrics = {k: v for k, v in test.items() if isinstance(v, (int, float))}
            models.append(_entry(name, "v1", CLASSIFICATION_RECOMMENDED, metrics=metrics))
    for report, name in ((resnet18, "resnet18"), (small_cnn, "small_cnn")):
        if report is None:
            continue
        test = report.get("test_metrics", {})
        metrics = {k: v for k, v in test.items() if isinstance(v, (int, float))}
        models.append(_entry(name, "v1", CLASSIFICATION_RECOMMENDED, metrics=metrics))
    models.sort(key=lambda m: {"floor": 0, "baseline": 1, "exploratory": 2}[m.tier])

    return TaskComparison(
        task="classification", horizons_h=None, models=models,
        recommended_model=_DISPLAY_NAMES.get(CLASSIFICATION_RECOMMENDED, CLASSIFICATION_RECOMMENDED),
        methodology_note=(
            "Macro-F1 and accuracy on the frozen, storm-disjoint test split "
            "(docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md). CNN/ResNet-18 rows are exploratory "
            "deep-learning results (docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md) that did not "
            "beat the Logistic Regression baseline at the current dataset scale."
        ),
    )


def get_model_performance() -> ModelPerformanceResponse:
    phase7 = _load_json("phase7_intensity_gru_results.json")
    phase8 = _load_json("phase8_track_gru_results.json")
    phase5 = _load_json("phase5_baseline_results.json")
    phase6_resnet18 = _load_json("phase6_resnet18_results.json")
    phase6_small_cnn = _load_json("phase6_small_cnn_results.json")

    intensity = (
        _sequence_comparison(phase7, "intensity", INTENSITY_RECOMMENDED, "MAE (kt)")
        if phase7 is not None
        else TaskComparison(task="intensity", horizons_h=None, models=[],
                            recommended_model=_DISPLAY_NAMES[INTENSITY_RECOMMENDED],
                            methodology_note="Benchmark report unavailable on this deployment.")
    )
    track = (
        _sequence_comparison(phase8, "track", TRACK_RECOMMENDED, "Mean track error (km)")
        if phase8 is not None
        else TaskComparison(task="track", horizons_h=None, models=[],
                            recommended_model=_DISPLAY_NAMES[TRACK_RECOMMENDED],
                            methodology_note="Benchmark report unavailable on this deployment.")
    )
    classification = _classification_comparison(phase5, phase6_resnet18, phase6_small_cnn)

    dataset_version = phase7.get("dataset_version", "v1") if phase7 else "v1"
    split_version = phase7.get("split_version", "v1") if phase7 else "v1"

    return ModelPerformanceResponse(
        generated_at=dt.datetime.now(dt.timezone.utc),
        dataset_version=dataset_version, split_version=split_version,
        intensity=intensity, track=track, classification=classification,
    )
