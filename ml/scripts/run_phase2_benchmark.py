"""Run the Phase 2 baseline benchmark: fit all baselines, evaluate on test.

Loads the frozen split + dataset (never recomputes them), fits Persistence /
Ridge / LightGBM for intensity and Persistence / CLIPER-style / LightGBM for
track, evaluates each once on the held-out test set, and writes:

  ml/reports/phase2_benchmark_results.json   -- every BenchmarkResult
  ml/reports/phase2_comparison_intensity.md  -- MAE-by-horizon comparison table
  ml/reports/phase2_comparison_track.md      -- track-error-by-horizon table

Also saves raw test-set predictions (ml/reports/phase2_test_predictions.parquet
mirrored to $DATA_ROOT) for the plotting script, so plots never re-fit models.

Usage:
    python ml/scripts/run_phase2_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402
from ml.geostrom_ml.evaluation.benchmark import (  # noqa: E402
    comparison_table, run_benchmark, write_results,
)
from ml.geostrom_ml.features.engineering import HORIZONS_H  # noqa: E402
from ml.geostrom_ml.models.intensity_baselines import (  # noqa: E402
    LightGBMIntensity, PersistenceIntensity, RidgeIntensity, target_col as intensity_target_col,
)
from ml.geostrom_ml.models.track_baselines import (  # noqa: E402
    CliperTrack, LightGBMTrack, PersistenceTrack, dlat_col, dlon_col,
)
from ml.geostrom_ml.splits.split import load_split_manifest  # noqa: E402

DATASET_VERSION = "v1"


def load_dataset_splits():
    d = zone("datasets", DATASET_VERSION)
    train = pd.read_parquet(d / "train.parquet")
    val = pd.read_parquet(d / "val.parquet")
    test = pd.read_parquet(d / "test.parquet")
    return train, val, test


def save_test_predictions(models, test_df, out_path):
    """Persist every model's raw test-set predictions for offline plotting."""
    frame = test_df[["sid", "t_ref", "season", "ref_lat", "ref_lon",
                     "ref_wind", "ref_pres"]].copy()
    for h in HORIZONS_H:
        frame[f"y_wind_abs_{h}h_true"] = test_df[intensity_target_col(h)]
        frame[f"y_lat_future_{h}h_true"] = test_df[f"y_lat_future_{h}h"]
        frame[f"y_lon_future_{h}h_true"] = test_df[f"y_lon_future_{h}h"]

    for model in models:
        preds = model.predict(test_df)
        if model.task == "intensity":
            for h in HORIZONS_H:
                frame[f"{model.name}__wind_{h}h"] = preds[intensity_target_col(h)]
        else:
            for h in HORIZONS_H:
                frame[f"{model.name}__dlat_{h}h"] = preds[dlat_col(h)]
                frame[f"{model.name}__dlon_{h}h"] = preds[dlon_col(h)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return frame


def main() -> int:
    split_manifest = load_split_manifest()
    split_version = split_manifest["split_version"]
    feature_version = split_manifest["feature_version"]

    train_df, val_df, test_df = load_dataset_splits()
    print(f"Loaded dataset  : train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    intensity_models = [
        PersistenceIntensity(), RidgeIntensity(), LightGBMIntensity(),
    ]
    track_models = [
        PersistenceTrack(), CliperTrack(), LightGBMTrack(),
    ]

    print("Fitting and evaluating (test set touched once per model)...")
    results = run_benchmark(
        intensity_models, track_models, train_df, val_df, test_df,
        HORIZONS_H, DATASET_VERSION, split_version, feature_version,
    )

    results_path = REPORT_DIR / "phase2_benchmark_results.json"
    write_results(results, results_path)
    print(f"Results written : {results_path}  ({len(results)} rows)")

    # --- comparison tables --------------------------------------------
    print("\n=== INTENSITY: MAE (kt) by horizon ===")
    t = comparison_table(results, "intensity", "mae_kt")
    print(t.round(2).to_string())
    t.round(2).to_markdown(REPORT_DIR / "phase2_comparison_intensity_mae.md")

    print("\n=== INTENSITY: bias (kt) by horizon ===")
    tb = comparison_table(results, "intensity", "bias_kt")
    print(tb.round(2).to_string())

    print("\n=== TRACK: mean great-circle error (km) by horizon ===")
    tt = comparison_table(results, "track", "mean_track_error_km")
    print(tt.round(1).to_string())
    tt.round(1).to_markdown(REPORT_DIR / "phase2_comparison_track_mean_km.md")

    print("\n=== TRACK: median great-circle error (km) by horizon ===")
    ttm = comparison_table(results, "track", "median_track_error_km")
    print(ttm.round(1).to_string())

    # --- skill vs. persistence, headline horizon (24h) ------------------
    headline = 24
    print(f"\n=== SKILL vs. persistence @ {headline}h ===")
    pers_int_mae = t.loc["intensity_persistence_v1", headline]
    for name in t.index:
        skill = 100 * (pers_int_mae - t.loc[name, headline]) / pers_int_mae
        print(f"  intensity  {name:<28} MAE={t.loc[name, headline]:.2f} kt  "
              f"skill_vs_persistence={skill:+.1f}%")
    pers_trk_km = tt.loc["track_persistence_v1", headline]
    for name in tt.index:
        skill = 100 * (pers_trk_km - tt.loc[name, headline]) / pers_trk_km
        print(f"  track      {name:<28} mean_err={tt.loc[name, headline]:.1f} km  "
              f"skill_vs_persistence={skill:+.1f}%")

    # --- save raw test predictions for plotting --------------------------
    pred_path_local = REPORT_DIR / "phase2_test_predictions.parquet"
    save_test_predictions(intensity_models + track_models, test_df, pred_path_local)
    print(f"\nTest predictions saved: {pred_path_local}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
