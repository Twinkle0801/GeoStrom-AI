"""Benchmark harness: model -> predictions -> metrics -> BenchmarkResult.

Implements the "compare models" principle from ML_ARCHITECTURE.md §1.1/§8:
one frozen split, one metric set per task, N models, one comparison table.
Adding a new baseline means registering it here -- no per-model evaluation
code is written elsewhere.

Rule enforced: the test set is touched exactly once per model (a single
`predict(test_df)` call per model), consistent with ML_ARCHITECTURE.md §8
("Test set is touched once per model, at the end").

Every result dict contains, at minimum, the fields the Phase 2 task brief
requires: model name, model version, dataset version, split version, feature
version, forecast horizon, sample count, metrics, timestamp, and the model's
configuration.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.evaluation.metrics import intensity_metrics, track_point_metrics  # noqa: E402
from ml.geostrom_ml.models.base import BaselineModel  # noqa: E402
from ml.geostrom_ml.models.intensity_baselines import target_col as intensity_target_col  # noqa: E402
from ml.geostrom_ml.models.track_baselines import dlat_col, dlon_col  # noqa: E402

MODEL_VERSION = "v1"


def _model_config(model: BaselineModel) -> dict:
    """A JSON-safe snapshot of the model's constructor state."""
    cfg = {}
    for k, v in vars(model).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            cfg[k] = v
        elif isinstance(v, (list, tuple)):
            cfg[k] = list(v)
    return cfg


def evaluate_intensity_model(
    model: BaselineModel, test_df: pd.DataFrame, horizons_h,
    dataset_version: str, split_version: str, feature_version: str,
) -> list[dict]:
    preds = model.predict(test_df)  # single call -- test set touched once
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    for h in horizons_h:
        col = intensity_target_col(h)
        y_true = test_df[col].to_numpy(dtype=np.float64)
        y_pred = np.asarray(preds[col], dtype=np.float64)
        results.append({
            "model_name": model.name, "model_version": MODEL_VERSION,
            "task": "intensity", "dataset_version": dataset_version,
            "split_version": split_version, "feature_version": feature_version,
            "forecast_horizon_h": h, "sample_count": int(len(test_df)),
            "metrics": intensity_metrics(y_true, y_pred),
            "timestamp_utc": now, "config": _model_config(model),
        })
    return results


def evaluate_track_model(
    model: BaselineModel, test_df: pd.DataFrame, horizons_h,
    dataset_version: str, split_version: str, feature_version: str,
) -> list[dict]:
    preds = model.predict(test_df)  # single call -- test set touched once
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ref_lat = test_df["ref_lat"].to_numpy(dtype=np.float64)
    ref_lon = test_df["ref_lon"].to_numpy(dtype=np.float64)
    results = []
    for h in horizons_h:
        true_lat = test_df[f"y_lat_future_{h}h"].to_numpy(dtype=np.float64)
        true_lon = test_df[f"y_lon_future_{h}h"].to_numpy(dtype=np.float64)
        pred_dlat = np.asarray(preds[dlat_col(h)], dtype=np.float64)
        pred_dlon = np.asarray(preds[dlon_col(h)], dtype=np.float64)
        results.append({
            "model_name": model.name, "model_version": MODEL_VERSION,
            "task": "track", "dataset_version": dataset_version,
            "split_version": split_version, "feature_version": feature_version,
            "forecast_horizon_h": h, "sample_count": int(len(test_df)),
            "metrics": track_point_metrics(ref_lat, ref_lon, true_lat, true_lon,
                                           pred_dlat, pred_dlon),
            "timestamp_utc": now, "config": _model_config(model),
        })
    return results


def run_benchmark(
    intensity_models: list[BaselineModel],
    track_models: list[BaselineModel],
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    horizons_h, dataset_version: str, split_version: str, feature_version: str,
) -> list[dict]:
    """Fit every model on train, evaluate once on test. Val is unused by
    these baselines (no hyperparameter selection is performed in Phase 2 --
    fixed, documented hyperparameters are used throughout), but is threaded
    through for future models that do need it, and its predictions are
    also computed and returned for diagnostic plotting.
    """
    all_results = []
    for model in intensity_models:
        model.fit(train_df)
        all_results.extend(evaluate_intensity_model(
            model, test_df, horizons_h, dataset_version, split_version, feature_version))
    for model in track_models:
        model.fit(train_df)
        all_results.extend(evaluate_track_model(
            model, test_df, horizons_h, dataset_version, split_version, feature_version))
    return all_results


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


def comparison_table(results: list[dict], task: str, metric_key: str,
                      horizon_h: int | None = None) -> pd.DataFrame:
    """Pivot benchmark results into a model x horizon comparison table."""
    rows = [r for r in results if r["task"] == task
            and (horizon_h is None or r["forecast_horizon_h"] == horizon_h)]
    df = pd.DataFrame([{
        "model": r["model_name"], "horizon_h": r["forecast_horizon_h"],
        "n": r["sample_count"], metric_key: r["metrics"][metric_key],
    } for r in rows])
    if df.empty:
        return df
    return df.pivot(index="model", columns="horizon_h", values=metric_key)
