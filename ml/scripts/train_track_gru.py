"""Phase 8: train + evaluate the GRU track model against the frozen Phase 2
dataset/split, and compare honestly against the Phase 2 track baselines.

Reuses, unmodified: `ml/manifests/splits_v1.json` (frozen split),
`$DATA_ROOT/datasets/v1/{train,val,test}.parquet` (Phase 2's materialised
windows -- NOT rebuilt), `ml/geostrom_ml/evaluation/benchmark.py::
evaluate_track_model` (Phase 2's evaluation harness, unmodified),
`ml/reports/phase2_benchmark_results.json` (Phase 2's own committed
results -- read for comparison, never recomputed).

Usage:
    python ml/scripts/train_track_gru.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402
from ml.geostrom_ml.evaluation.benchmark import evaluate_track_model  # noqa: E402
from ml.geostrom_ml.features.engineering import HORIZONS_H  # noqa: E402
from ml.geostrom_ml.models.track_gru import TrackGRU, TrackGRUConfig  # noqa: E402
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

    config = TrackGRUConfig()
    print(f"Config: {config}")

    results: dict = {
        "phase": 8, "model_family": "track_gru", "config": vars(config),
        "dataset_version": DATASET_VERSION, "split_version": SPLIT_VERSION,
        "feature_version": FEATURE_VERSION, "horizons_h": list(HORIZONS_H),
        "models": {},
    }

    print(f"\n=== Training TrackGRU ===")
    model = TrackGRU(config=config)
    model.fit(train_df, val_df=val_df)
    print(f"  best_epoch={model.best_epoch}  n_epochs_ran={len(model.history)}  "
          f"best_val_mean_track_error_km="
          f"{model.history[model.best_epoch]['val_mean_track_error_km']:.3f}")

    test_results = evaluate_track_model(
        model, test_df, HORIZONS_H, DATASET_VERSION, SPLIT_VERSION, FEATURE_VERSION)

    results["models"][model.name] = {
        "best_epoch": model.best_epoch,
        "n_epochs_ran": len(model.history),
        "training_history": model.history,
        "test_metrics_by_horizon": {r["forecast_horizon_h"]: r["metrics"] for r in test_results},
    }
    for h in HORIZONS_H:
        m = results["models"][model.name]["test_metrics_by_horizon"][h]
        print(f"  +{h:>2}h  mean_track_error={m['mean_track_error_km']:.3f}km  "
              f"median={m['median_track_error_km']:.3f}km  rmse={m['rmse_track_error_km']:.3f}km")

    # ---- comparison against the Phase 2 baselines (read, never recomputed) ----
    phase2_path = REPORT_DIR / "phase2_benchmark_results.json"
    phase2_results = json.loads(phase2_path.read_text(encoding="utf-8"))
    phase2_track = [r for r in phase2_results if r["task"] == "track"]

    comparison = {}
    for h in HORIZONS_H:
        row = {}
        for r in phase2_track:
            if r["forecast_horizon_h"] == h:
                row[r["model_name"]] = {
                    "mean_track_error_km": r["metrics"]["mean_track_error_km"],
                    "median_track_error_km": r["metrics"]["median_track_error_km"],
                    "rmse_track_error_km": r["metrics"]["rmse_track_error_km"],
                }
        for r in test_results:
            if r["forecast_horizon_h"] == h:
                row[r["model_name"]] = {
                    "mean_track_error_km": r["metrics"]["mean_track_error_km"],
                    "median_track_error_km": r["metrics"]["median_track_error_km"],
                    "rmse_track_error_km": r["metrics"]["rmse_track_error_km"],
                }
        comparison[h] = row
    results["comparison_vs_phase2"] = comparison

    # ---- headline: 24h GRU vs Phase 2's best track baseline (CLIPER) ----
    headline_h = 24
    persistence_km = comparison[headline_h].get("track_persistence_v1", {}).get("mean_track_error_km")
    cliper_km = comparison[headline_h].get("track_cliper_v1", {}).get("mean_track_error_km")
    lightgbm_km = comparison[headline_h].get("track_lightgbm_v1", {}).get("mean_track_error_km")
    gru_km = comparison[headline_h].get("track_gru_v1", {}).get("mean_track_error_km")

    best_phase2_name, best_phase2_km = min(
        [("track_cliper_v1", cliper_km), ("track_lightgbm_v1", lightgbm_km)],
        key=lambda kv: kv[1],
    )

    print(f"\n=== 24h headline comparison ===")
    print(f"  Persistence          mean_track_error={persistence_km:.3f}km")
    print(f"  Phase 2 CLIPER       mean_track_error={cliper_km:.3f}km")
    print(f"  Phase 2 LightGBM     mean_track_error={lightgbm_km:.3f}km")
    print(f"  Phase 8 GRU          mean_track_error={gru_km:.3f}km")
    if best_phase2_km is not None and gru_km is not None:
        pct_change = 100.0 * (best_phase2_km - gru_km) / best_phase2_km
        verdict = "BEATS" if pct_change > 0 else "DOES NOT BEAT"
        print(f"  GRU {verdict} the best Phase 2 baseline ({best_phase2_name}, "
              f"{best_phase2_km:.3f}km) -- {pct_change:+.1f}% change in mean track error")
        results["headline_24h"] = {
            "persistence_mean_track_error_km": persistence_km,
            "phase2_cliper_mean_track_error_km": cliper_km,
            "phase2_lightgbm_mean_track_error_km": lightgbm_km,
            "phase8_gru_mean_track_error_km": gru_km,
            "best_phase2_baseline": best_phase2_name,
            "best_phase2_mean_track_error_km": best_phase2_km,
            "pct_change_vs_best_phase2_baseline": pct_change,
            "beats_best_phase2_baseline": pct_change > 0,
        }

    out_path = REPORT_DIR / "phase8_track_gru_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
