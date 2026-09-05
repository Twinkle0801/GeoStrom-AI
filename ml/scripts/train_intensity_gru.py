"""Phase 7: train + evaluate the GRU intensity model(s) against the frozen
Phase 2 dataset/split, and compare honestly against the Phase 2 baselines.

Reuses, unmodified: `ml/manifests/splits_v1.json` (frozen split),
`$DATA_ROOT/datasets/v1/{train,val,test}.parquet` (Phase 2's materialised
windows -- NOT rebuilt), `ml/geostrom_ml/evaluation/benchmark.py::
evaluate_intensity_model` (Phase 2's evaluation harness),
`ml/reports/phase2_benchmark_results.json` (Phase 2's own committed
results -- read for comparison, never recomputed).

Usage:
    python ml/scripts/train_intensity_gru.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, REPORT_DIR, zone  # noqa: E402
from ml.geostrom_ml.evaluation.benchmark import evaluate_intensity_model  # noqa: E402
from ml.geostrom_ml.evaluation.metrics import ri_recall  # noqa: E402
from ml.geostrom_ml.features.engineering import HORIZONS_H  # noqa: E402
from ml.geostrom_ml.models.intensity_gru import (  # noqa: E402
    GRUIntensityConfig, IntensityGRU, delta_target_col,
)
from ml.geostrom_ml.splits.split import (  # noqa: E402
    FEATURE_VERSION, SPLIT_VERSION, load_split_manifest, storm_to_split_map,
    validate_split_integrity,
)

DATASET_VERSION = "v1"


def main() -> int:
    import pandas as pd

    split_manifest = load_split_manifest()
    validate_split_integrity(split_manifest)  # defense-in-depth re-check, not a rebuild
    sid_to_split = storm_to_split_map(split_manifest)

    data_dir = zone("datasets", DATASET_VERSION)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Re-verify (not re-derive) that the materialised parquet split matches
    # the frozen manifest -- catches silent drift between the two artifacts.
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        actual_splits = set(df["sid"].map(sid_to_split).unique())
        if actual_splits != {name}:
            raise ValueError(f"{name}.parquet contains storms mapped to split(s) "
                             f"{actual_splits}, expected only {{{name!r}}}")
    print(f"train={len(train_df)} windows/{train_df['sid'].nunique()} storms  "
          f"val={len(val_df)} windows/{val_df['sid'].nunique()} storms  "
          f"test={len(test_df)} windows/{test_df['sid'].nunique()} storms")

    config = GRUIntensityConfig()
    print(f"Config: {config}")

    results: dict = {
        "phase": 7, "model_family": "intensity_gru", "config": vars(config),
        "dataset_version": DATASET_VERSION, "split_version": SPLIT_VERSION,
        "feature_version": FEATURE_VERSION, "horizons_h": list(HORIZONS_H),
        "models": {},
    }

    all_benchmark_rows = []
    for target_mode in ("absolute", "delta"):
        print(f"\n=== Training IntensityGRU(target_mode={target_mode!r}) ===")
        model = IntensityGRU(target_mode=target_mode, config=config)
        model.fit(train_df, val_df=val_df)
        print(f"  best_epoch={model.best_epoch}  "
              f"n_epochs_ran={len(model.history)}  "
              f"best_val_mae_kt={model.history[model.best_epoch]['val_mae_kt']:.3f}")

        test_results = evaluate_intensity_model(
            model, test_df, HORIZONS_H, DATASET_VERSION, SPLIT_VERSION, FEATURE_VERSION)
        all_benchmark_rows.extend(test_results)

        # RI-recall diagnostic (delta scale), per docs/ML_ARCHITECTURE.md §6.5
        delta_preds = model.predict_delta(test_df) if target_mode == "delta" else None
        ri_by_horizon = {}
        if delta_preds is not None:
            for h in HORIZONS_H:
                y_true_delta = test_df[delta_target_col(h)].to_numpy()
                y_pred_delta = delta_preds[delta_target_col(h)]
                ri_by_horizon[h] = ri_recall(y_true_delta, y_pred_delta)

        results["models"][model.name] = {
            "target_mode": target_mode,
            "best_epoch": model.best_epoch,
            "n_epochs_ran": len(model.history),
            "training_history": model.history,
            "test_metrics_by_horizon": {
                r["forecast_horizon_h"]: r["metrics"] for r in test_results
            },
            "ri_recall_by_horizon": ri_by_horizon,
        }
        for h in HORIZONS_H:
            m = results["models"][model.name]["test_metrics_by_horizon"][h]
            print(f"  +{h:>2}h  MAE={m['mae_kt']:.3f}kt  RMSE={m['rmse_kt']:.3f}kt  "
                  f"bias={m['bias_kt']:+.3f}kt")

    # ---- comparison against the Phase 2 baselines (read, never recomputed) ----
    phase2_path = REPORT_DIR / "phase2_benchmark_results.json"
    phase2_results = json.loads(phase2_path.read_text(encoding="utf-8"))
    phase2_intensity = [r for r in phase2_results if r["task"] == "intensity"]

    comparison = {}
    for h in HORIZONS_H:
        row = {}
        for r in phase2_intensity:
            if r["forecast_horizon_h"] == h:
                row[r["model_name"]] = {
                    "mae_kt": r["metrics"]["mae_kt"], "rmse_kt": r["metrics"]["rmse_kt"],
                }
        for r in all_benchmark_rows:
            if r["forecast_horizon_h"] == h:
                row[r["model_name"]] = {
                    "mae_kt": r["metrics"]["mae_kt"], "rmse_kt": r["metrics"]["rmse_kt"],
                }
        comparison[h] = row
    results["comparison_vs_phase2"] = comparison

    # ---- headline: 24h GRU (absolute) vs Phase 2 LightGBM ----
    headline_h = 24
    lgbm_mae = comparison[headline_h].get("intensity_lightgbm_v1", {}).get("mae_kt")
    persistence_mae = comparison[headline_h].get("intensity_persistence_v1", {}).get("mae_kt")
    gru_abs_mae = comparison[headline_h].get("intensity_gru_v1", {}).get("mae_kt")
    gru_delta_mae = comparison[headline_h].get("intensity_gru_delta_v1", {}).get("mae_kt")

    print(f"\n=== 24h headline comparison ===")
    print(f"  Persistence         MAE={persistence_mae:.3f}kt")
    print(f"  Phase 2 LightGBM    MAE={lgbm_mae:.3f}kt  (existing benchmark)")
    print(f"  Phase 7 GRU (abs)   MAE={gru_abs_mae:.3f}kt")
    print(f"  Phase 7 GRU (delta) MAE={gru_delta_mae:.3f}kt  (diagnostic; reconstructed to absolute)")
    if lgbm_mae is not None and gru_abs_mae is not None:
        delta_vs_lgbm = 100.0 * (lgbm_mae - gru_abs_mae) / lgbm_mae
        verdict = "BEATS" if delta_vs_lgbm > 0 else "DOES NOT BEAT"
        print(f"  GRU (abs) {verdict} the Phase 2 LightGBM baseline "
              f"({delta_vs_lgbm:+.1f}% change in MAE)")
        results["headline_24h"] = {
            "persistence_mae_kt": persistence_mae, "phase2_lightgbm_mae_kt": lgbm_mae,
            "phase7_gru_absolute_mae_kt": gru_abs_mae, "phase7_gru_delta_mae_kt": gru_delta_mae,
            "pct_mae_change_vs_lightgbm": delta_vs_lgbm, "beats_lightgbm_baseline": delta_vs_lgbm > 0,
        }

    out_path = REPORT_DIR / "phase7_intensity_gru_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
