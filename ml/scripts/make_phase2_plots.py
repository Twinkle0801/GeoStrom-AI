"""Phase 2 offline evaluation plots.

Reads the artifacts already written by run_phase2_benchmark.py (never
re-fits a model) and produces six plots into ml/reports/figures/:

  1. intensity_actual_vs_predicted.png
  2. intensity_error_distribution.png
  3. track_examples.png             (observed vs predicted tracks, sample storms)
  4. track_error_vs_horizon.png
  5. model_comparison.png           (headline 24h bar chart, both tasks)
  6. error_by_storm.png             (per-storm intensity/track error spread)

Every plotted quantity is either a value present in the IBTrACS dataset or a
model prediction computed by run_phase2_benchmark.py -- nothing here
fabricates a meteorological field (e.g. no wind field, no rainfall).

Usage:
    python ml/scripts/make_phase2_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR  # noqa: E402
from ml.geostrom_ml.evaluation.metrics import intensity_metrics, track_point_metrics  # noqa: E402
from ml.geostrom_ml.features.engineering import HORIZONS_H  # noqa: E402

FIG_DIR = REPORT_DIR / "figures"
HEADLINE_H = 24

INTENSITY_MODELS = ["intensity_persistence_v1", "intensity_ridge_v1", "intensity_lightgbm_v1"]
TRACK_MODELS = ["track_persistence_v1", "track_cliper_v1", "track_lightgbm_v1"]
COLORS = {
    "intensity_persistence_v1": "#9BA6B8", "intensity_ridge_v1": "#4C8DFF",
    "intensity_lightgbm_v1": "#F72585",
    "track_persistence_v1": "#9BA6B8", "track_cliper_v1": "#4C8DFF",
    "track_lightgbm_v1": "#F72585",
}


def load_artifacts():
    pred = pd.read_parquet(REPORT_DIR / "phase2_test_predictions.parquet")
    results = json.loads((REPORT_DIR / "phase2_benchmark_results.json").read_text())
    return pred, results


def plot_intensity_actual_vs_predicted(pred: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    true = pred[f"y_wind_abs_{HEADLINE_H}h_true"]
    lims = (0, max(true.max(), pred[[f"{m}__wind_{HEADLINE_H}h" for m in INTENSITY_MODELS]].max().max()) + 10)
    for ax, model in zip(axes, INTENSITY_MODELS):
        predicted = pred[f"{model}__wind_{HEADLINE_H}h"]
        ax.scatter(true, predicted, s=8, alpha=0.4, color=COLORS[model])
        ax.plot(lims, lims, "--", color="#5E6979", linewidth=1, label="perfect")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Observed wind (kt) [OBSERVED DATA]")
        ax.set_title(model.replace("intensity_", "").replace("_v1", ""))
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted wind (kt) [MODEL PREDICTION]")
    fig.suptitle(f"Intensity: actual vs. predicted wind @ +{HEADLINE_H}h (test set, n={len(pred)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intensity_actual_vs_predicted.png", dpi=130)
    plt.close(fig)


def plot_intensity_error_distribution(pred: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    true = pred[f"y_wind_abs_{HEADLINE_H}h_true"]
    for model in INTENSITY_MODELS:
        err = pred[f"{model}__wind_{HEADLINE_H}h"] - true
        ax.hist(err, bins=40, alpha=0.5, label=model.replace("intensity_", "").replace("_v1", ""),
                color=COLORS[model])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Prediction error, pred - observed (kt)  [DERIVED FEATURE: error]")
    ax.set_ylabel("Count of test windows")
    ax.set_title(f"Intensity error distribution @ +{HEADLINE_H}h (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intensity_error_distribution.png", dpi=130)
    plt.close(fig)


def plot_track_examples(pred: pd.DataFrame, n_examples: int = 4):
    rng = np.random.default_rng(42)
    sids = pred["sid"].unique()
    chosen = rng.choice(sids, size=min(n_examples, len(sids)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, sid in zip(axes.ravel(), chosen):
        storm = pred[pred["sid"] == sid].sort_values("t_ref")
        row = storm.iloc[len(storm) // 2]  # a representative origin time

        ax.plot(storm["ref_lon"], storm["ref_lat"], "o-", color="#22D3A7",
                markersize=3, linewidth=1, label="Observed track [OBSERVED DATA]")
        ax.plot(row["ref_lon"], row["ref_lat"], "s", color="#22D3A7", markersize=10,
                label="Forecast origin")

        for model in TRACK_MODELS:
            lats, lons = [row["ref_lat"]], [row["ref_lon"]]
            for h in HORIZONS_H:
                lats.append(row["ref_lat"] + row[f"{model}__dlat_{h}h"])
                lon_wrapped = ((row["ref_lon"] + row[f"{model}__dlon_{h}h"] + 180) % 360) - 180
                lons.append(lon_wrapped)
            ax.plot(lons, lats, "x--", color=COLORS[model], markersize=6, linewidth=1,
                    label=f"{model.replace('track_', '').replace('_v1', '')} [MODEL PREDICTION]")

        future_lats = [row[f"y_lat_future_{h}h_true"] for h in HORIZONS_H]
        future_lons = [row[f"y_lon_future_{h}h_true"] for h in HORIZONS_H]
        ax.plot(future_lons, future_lats, "o", color="#FFB020", markersize=6,
                label="Actual future [OBSERVED DATA]")

        ax.set_title(f"{sid}")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Track examples: observed history, forecast origin, "
                 "model predictions (+6/+12/+18/+24h), and actual future")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_examples.png", dpi=130)
    plt.close(fig)


def plot_track_error_vs_horizon(results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model in TRACK_MODELS:
        rows = sorted([r for r in results if r["model_name"] == model],
                      key=lambda r: r["forecast_horizon_h"])
        h = [r["forecast_horizon_h"] for r in rows]
        mean_e = [r["metrics"]["mean_track_error_km"] for r in rows]
        med_e = [r["metrics"]["median_track_error_km"] for r in rows]
        label = model.replace("track_", "").replace("_v1", "")
        axes[0].plot(h, mean_e, "o-", label=label, color=COLORS[model])
        axes[1].plot(h, med_e, "o-", label=label, color=COLORS[model])
    for ax, title in zip(axes, ["Mean great-circle error", "Median great-circle error"]):
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel("Track error (km)  [DERIVED: Haversine(pred, observed)]")
        ax.set_title(title)
        ax.set_xticks(list(HORIZONS_H))
        ax.legend()
    fig.suptitle("Track error growth with forecast horizon (test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_error_vs_horizon.png", dpi=130)
    plt.close(fig)


def plot_model_comparison(results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    int_rows = [r for r in results if r["task"] == "intensity" and r["forecast_horizon_h"] == HEADLINE_H]
    names = [r["model_name"].replace("intensity_", "").replace("_v1", "") for r in int_rows]
    vals = [r["metrics"]["mae_kt"] for r in int_rows]
    axes[0].bar(names, vals, color=[COLORS[r["model_name"]] for r in int_rows])
    axes[0].set_ylabel("MAE (kt)  [DERIVED: mean(|pred-observed|)]")
    axes[0].set_title(f"Intensity MAE @ +{HEADLINE_H}h")

    trk_rows = [r for r in results if r["task"] == "track" and r["forecast_horizon_h"] == HEADLINE_H]
    names_t = [r["model_name"].replace("track_", "").replace("_v1", "") for r in trk_rows]
    vals_t = [r["metrics"]["mean_track_error_km"] for r in trk_rows]
    axes[1].bar(names_t, vals_t, color=[COLORS[r["model_name"]] for r in trk_rows])
    axes[1].set_ylabel("Mean great-circle error (km)  [DERIVED: Haversine]")
    axes[1].set_title(f"Track error @ +{HEADLINE_H}h")

    for ax in axes:
        for i, v in enumerate(ax.containers[0]):
            h = v.get_height()
            ax.text(v.get_x() + v.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Phase 2 baseline comparison, headline {HEADLINE_H}h horizon (test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)


def plot_error_by_storm(pred: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    best_int = "intensity_lightgbm_v1"
    per_storm_int = (
        (pred[f"{best_int}__wind_{HEADLINE_H}h"] - pred[f"y_wind_abs_{HEADLINE_H}h_true"]).abs()
        .groupby(pred["sid"]).mean().sort_values()
    )
    axes[0].bar(range(len(per_storm_int)), per_storm_int.to_numpy(), color="#F72585")
    axes[0].set_xlabel("Test storms (sorted)")
    axes[0].set_ylabel("Mean |error| (kt)  [DERIVED: per-storm intensity MAE]")
    axes[0].set_title(f"Per-storm intensity error, {best_int.replace('_v1','')} @ +{HEADLINE_H}h")

    best_trk = "track_cliper_v1"
    lats_pred = pred["ref_lat"] + pred[f"{best_trk}__dlat_{HEADLINE_H}h"]
    lons_pred = ((pred["ref_lon"] + pred[f"{best_trk}__dlon_{HEADLINE_H}h"] + 180) % 360) - 180
    m = track_point_metrics(
        pred["ref_lat"].to_numpy(), pred["ref_lon"].to_numpy(),
        pred[f"y_lat_future_{HEADLINE_H}h_true"].to_numpy(),
        pred[f"y_lon_future_{HEADLINE_H}h_true"].to_numpy(),
        pred[f"{best_trk}__dlat_{HEADLINE_H}h"].to_numpy(),
        pred[f"{best_trk}__dlon_{HEADLINE_H}h"].to_numpy(),
    )
    from ml.geostrom_ml.features.geo import haversine_km
    err_km = haversine_km(
        pred[f"y_lat_future_{HEADLINE_H}h_true"].to_numpy(),
        pred[f"y_lon_future_{HEADLINE_H}h_true"].to_numpy(),
        lats_pred.to_numpy(), lons_pred.to_numpy(),
    )
    per_storm_trk = pd.Series(err_km, index=pred["sid"]).groupby(level=0).mean().sort_values()
    axes[1].bar(range(len(per_storm_trk)), per_storm_trk.to_numpy(), color="#4C8DFF")
    axes[1].set_xlabel("Test storms (sorted)")
    axes[1].set_ylabel("Mean track error (km)  [DERIVED: per-storm mean Haversine error]")
    axes[1].set_title(f"Per-storm track error, {best_trk.replace('_v1','')} @ +{HEADLINE_H}h")

    fig.suptitle("Error spread across individual test storms (best-performing model per task)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_storm.png", dpi=130)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred, results = load_artifacts()

    plot_intensity_actual_vs_predicted(pred)
    plot_intensity_error_distribution(pred)
    plot_track_examples(pred)
    plot_track_error_vs_horizon(results)
    plot_model_comparison(results)
    plot_error_by_storm(pred)

    print(f"6 plots written to {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
