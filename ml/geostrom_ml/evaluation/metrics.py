"""Regression metrics for intensity and track evaluation.

MAE, RMSE, and bias are used for intensity, per ML_ARCHITECTURE.md §6.2 --
MAPE is deliberately never computed here (rejected explicitly in
ML_ARCHITECTURE.md §6.2: it penalises errors on weak storms far more
harshly than identical errors on intense ones, exactly inverting real-world
importance).

Track error is always geodesic (Haversine), never raw lat/lon differences
treated as a flat-plane distance, per ML_ARCHITECTURE.md §7.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.features.geo import along_cross_track_km, displace, haversine_km  # noqa: E402


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (pred - true). Positive = systematic over-forecast."""
    return float(np.mean(y_pred - y_true))


def skill_vs_baseline(model_mae: float, baseline_mae: float) -> float:
    """Percent MAE improvement over a reference baseline. Positive = better.

    skill = 100 * (baseline_mae - model_mae) / baseline_mae
    A model tied with the baseline scores 0. A model worse than the
    baseline scores negative -- reported as such, never clipped to zero.
    """
    if baseline_mae == 0:
        return float("nan")
    return float(100.0 * (baseline_mae - model_mae) / baseline_mae)


def intensity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "mae_kt": mae(y_true, y_pred),
        "rmse_kt": rmse(y_true, y_pred),
        "bias_kt": bias(y_true, y_pred),
    }


def track_point_metrics(
    ref_lat: np.ndarray, ref_lon: np.ndarray,
    true_lat: np.ndarray, true_lon: np.ndarray,
    pred_dlat: np.ndarray, pred_dlon: np.ndarray,
) -> dict:
    """Full track-error metric set for one horizon.

    `pred_dlat`/`pred_dlon` are the model's predicted displacement from the
    reference point; absolute predicted position is reconstructed via
    `displace` (wrap-safe) before computing geodesic error, exactly the way
    the served forecast will be reconstructed (ML_ARCHITECTURE.md §7.2).
    """
    pred_lat, pred_lon = displace(ref_lat, ref_lon, pred_dlat, pred_dlon)

    dist_err_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    along_km, cross_km = along_cross_track_km(
        ref_lat, ref_lon, true_lat, true_lon, pred_lat, pred_lon)

    lat_err_deg = pred_lat - true_lat
    # longitude error must go through the wrap-safe difference, never a raw
    # subtraction (docs/DATA_STRATEGY.md pitfall #5).
    from ml.geostrom_ml.features.geo import wrap_lon_diff
    lon_err_deg = wrap_lon_diff(true_lon, pred_lon)

    return {
        "n": int(len(true_lat)),
        "mean_lat_error_deg": float(np.mean(lat_err_deg)),
        "mean_lon_error_deg": float(np.mean(lon_err_deg)),
        "mean_track_error_km": float(np.mean(dist_err_km)),
        "median_track_error_km": float(np.median(dist_err_km)),
        "p90_track_error_km": float(np.percentile(dist_err_km, 90)),
        "max_track_error_km": float(np.max(dist_err_km)),
        "rmse_track_error_km": float(np.sqrt(np.mean(dist_err_km ** 2))),
        "mean_along_track_km": float(np.mean(along_km)),
        "mean_cross_track_km": float(np.mean(cross_km)),
        "mean_abs_along_track_km": float(np.mean(np.abs(along_km))),
        "mean_abs_cross_track_km": float(np.mean(np.abs(cross_km))),
    }
