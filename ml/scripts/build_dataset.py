"""Build the Phase 2 windowed feature dataset and assign the frozen split.

Reads the frozen splits_v1.json (never regenerates it). Loads IBTrACS,
builds causal per-timestep features, builds L=8/H={6,12,18,24} sequence
windows, tags each window with its storm's split, and writes one Parquet
file per split under $DATA_ROOT/datasets/<version>/ -- per the DATA_STRATEGY
zone architecture ("DATASET ZONE $DATA_ROOT/datasets/<build_version>/").

A small manifest.json (row counts, feature/split versions, column list) is
written alongside AND mirrored into ml/manifests/ (git-tracked; the manifest
is metadata, not data).

Usage:
    python ml/scripts/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, zone  # noqa: E402
from ml.geostrom_ml.data.ibtracs import load_usable_basin  # noqa: E402
from ml.geostrom_ml.features.engineering import (  # noqa: E402
    build_per_timestep_features, build_sequence_windows, flattened_feature_columns,
    HORIZONS_H, L_STEPS,
)
from ml.geostrom_ml.splits.split import (  # noqa: E402
    BASIN, SEASON_START, SEASON_END, load_split_manifest, storm_to_split_map,
)

DATASET_VERSION = "v1"


def main() -> int:
    split_manifest = load_split_manifest()
    sid_to_split = storm_to_split_map(split_manifest)

    df, overlap = load_usable_basin(BASIN, SEASON_START, SEASON_END)
    feat = build_per_timestep_features(df)
    windows = build_sequence_windows(feat, L=L_STEPS, horizons_h=HORIZONS_H)
    windows["split"] = windows["sid"].map(sid_to_split)

    unassigned = windows["split"].isna().sum()
    if unassigned:
        raise ValueError(
            f"{unassigned} windows belong to storms absent from the frozen "
            f"split manifest -- the split and the dataset build have drifted "
            f"out of sync. Re-run build_splits.py or investigate."
        )

    out_dir = zone("datasets", DATASET_VERSION, create=True)
    counts = {}
    for split_name in ("train", "val", "test"):
        sub = windows[windows["split"] == split_name].reset_index(drop=True)
        sub.to_parquet(out_dir / f"{split_name}.parquet", index=False)
        counts[split_name] = {
            "n_windows": int(len(sub)),
            "n_storms": int(sub["sid"].nunique()),
        }

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_version": split_manifest["split_version"],
        "feature_version": split_manifest["feature_version"],
        "basin": BASIN, "season_range": [SEASON_START, SEASON_END],
        "L_steps": L_STEPS, "horizons_h": list(HORIZONS_H),
        "n_flattened_feature_columns": len(flattened_feature_columns()),
        "flattened_feature_columns": flattened_feature_columns(),
        "target_columns": sorted(c for c in windows.columns if c.startswith("y_")),
        "usable_row_overlap_report": overlap,
        "counts": counts,
        "total_windows": int(len(windows)),
        "storage_path": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                            encoding="utf-8")
    mirror_path = MANIFEST_DIR / f"dataset_{DATASET_VERSION}_manifest.json"
    mirror_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Dataset version : {DATASET_VERSION}   feature version: {manifest['feature_version']}")
    print(f"Total windows   : {manifest['total_windows']}")
    for name, c in counts.items():
        print(f"  {name:<6} windows={c['n_windows']:<6} storms={c['n_storms']}")
    print(f"Flattened feature columns: {manifest['n_flattened_feature_columns']}")
    print(f"Written to      : {out_dir}")
    print(f"Manifest mirror : {mirror_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
