"""Storm-level, season-block train/val/test split (frozen, versioned).

Locked rules enforced here (docs/PROJECT_REQUIREMENTS.md §4.1, docs/
DATA_STRATEGY.md §9 decision #8):
  1. Split by storm ID, never by row/window.
  2. Season-block temporal split: train <= year X, val in (X, Y], test > Y.
     This tests generalisation to *future* storms -- the deployment
     condition -- rather than a random storm-level split, which would still
     let the model see storms from every era during training.
  3. The split is written to disk as JSON and is treated as frozen: once
     written, `build_dataset.py` and all benchmark scripts read it back
     rather than ever recomputing it ad hoc.

Season boundaries reuse the exact split proposed in Phase 1
(docs/PHASE_1_DATASET_VERIFICATION.md §11): train 1980-2004 (25 seasons),
val 2005-2009 (5 seasons), test 2010-2015 (6 seasons). Reusing Phase 1's own
proposal rather than inventing a new boundary keeps Phase 1's storage/sample
estimates consistent with what Phase 2 actually builds.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR  # noqa: E402
from ml.geostrom_ml.data.ibtracs import (  # noqa: E402
    WIND_COLUMN, PRESSURE_COLUMN, load_usable_basin,
)
from ml.geostrom_ml.features.engineering import (  # noqa: E402
    HEADLINE_HORIZON_H, HORIZONS_H, L_STEPS, STEP_HOURS,
)

SPLIT_VERSION = "v1"
BASIN = "NA"
SEASON_START, SEASON_END = 1980, 2015
TRAIN_SEASONS = (1980, 2004)
VAL_SEASONS = (2005, 2009)
TEST_SEASONS = (2010, 2015)
RANDOM_SEED = 42   # unused by the season-block rule itself, but fixed for any
                    # downstream stochastic step (model init, LightGBM, etc.)
                    # so the whole Phase 2 pipeline is reproducible end to end.
FEATURE_VERSION = "v1"  # bump if build_per_timestep_features/build_sequence_windows changes


def _season_of(sid: str) -> int:
    return int(sid[:4])


def build_split_manifest() -> dict:
    """Compute and return the split manifest (does not write to disk)."""
    df, overlap = load_usable_basin(BASIN, SEASON_START, SEASON_END)
    storm_season = df.groupby("SID")["SEASON"].first()

    train_sids = sorted(storm_season[(storm_season >= TRAIN_SEASONS[0])
                                      & (storm_season <= TRAIN_SEASONS[1])].index)
    val_sids = sorted(storm_season[(storm_season >= VAL_SEASONS[0])
                                    & (storm_season <= VAL_SEASONS[1])].index)
    test_sids = sorted(storm_season[(storm_season >= TEST_SEASONS[0])
                                     & (storm_season <= TEST_SEASONS[1])].index)

    def obs_count(sids):
        return int(df["SID"].isin(sids).sum())

    manifest = {
        "split_version": SPLIT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "basin": BASIN,
        "season_range": [SEASON_START, SEASON_END],
        "wind_column": WIND_COLUMN,
        "pressure_column": PRESSURE_COLUMN,
        "feature_version": FEATURE_VERSION,
        "feature_config": {
            "step_hours": STEP_HOURS,
            "L_steps": L_STEPS,
            "horizons_h": list(HORIZONS_H),
            "headline_horizon_h": HEADLINE_HORIZON_H,
        },
        "filtering_rules": [
            "synoptic times only (00/06/12/18 UTC, on the hour)",
            "IFLAG char1 == 'O' (USA-agency original report, not interpolated)",
            "TRACK_TYPE == 'main' (excludes spur/PROVISIONAL tracks)",
            f"{WIND_COLUMN} present (single agency, no cross-agency fallback)",
            f"season in [{SEASON_START}, {SEASON_END}]",
        ],
        "split_method": (
            "season-block temporal split by storm ID: a storm's season "
            "(from its SID, cross-checked against IBTrACS SEASON) determines "
            "its split. No storm's observations are ever divided across "
            "splits, and no random row/window-level shuffling is used."
        ),
        "ibtracs_overlap_report": overlap,
        "train": {
            "seasons": list(TRAIN_SEASONS), "storm_ids": train_sids,
            "n_storms": len(train_sids), "n_observations": obs_count(train_sids),
        },
        "val": {
            "seasons": list(VAL_SEASONS), "storm_ids": val_sids,
            "n_storms": len(val_sids), "n_observations": obs_count(val_sids),
        },
        "test": {
            "seasons": list(TEST_SEASONS), "storm_ids": test_sids,
            "n_storms": len(test_sids), "n_observations": obs_count(test_sids),
        },
    }

    # SID-derived season must agree with the IBTrACS SEASON field for every
    # storm in the split -- a cheap, independent cross-check that the split
    # boundary is exactly where it's claimed to be.
    mismatches = [sid for sid in train_sids + val_sids + test_sids
                  if _season_of(sid) != int(storm_season[sid])]
    manifest["sid_season_cross_check"] = {
        "n_checked": len(train_sids) + len(val_sids) + len(test_sids),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:10],
    }
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} storm(s) have SID-derived season != IBTrACS "
            f"SEASON field. Refusing to freeze an inconsistent split."
        )

    validate_split_integrity(manifest)
    return manifest


def validate_split_integrity(manifest: dict) -> None:
    """Raise if the three splits are not pairwise disjoint at the storm level."""
    train = set(manifest["train"]["storm_ids"])
    val = set(manifest["val"]["storm_ids"])
    test = set(manifest["test"]["storm_ids"])

    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    manifest["integrity_check"] = {
        "intersection_train_val": overlaps["train_val"],
        "intersection_train_test": overlaps["train_test"],
        "intersection_val_test": overlaps["val_test"],
        "all_disjoint": not any(overlaps.values()),
    }
    if any(overlaps.values()):
        raise ValueError(f"Split integrity violated -- storms shared across splits: {overlaps}")

    all_ids = manifest["train"]["storm_ids"] + manifest["val"]["storm_ids"] + manifest["test"]["storm_ids"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate storm IDs found across the concatenated split lists.")


def split_manifest_path(version: str = SPLIT_VERSION) -> Path:
    return MANIFEST_DIR / f"splits_{version}.json"


def write_split_manifest(manifest: dict | None = None) -> Path:
    manifest = manifest or build_split_manifest()
    path = split_manifest_path(manifest["split_version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def load_split_manifest(version: str = SPLIT_VERSION) -> dict:
    path = split_manifest_path(version)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run ml/scripts/build_splits.py first; "
            "the split is frozen and must not be silently regenerated."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_split_integrity(manifest)
    return manifest


def storm_to_split_map(manifest: dict) -> dict[str, str]:
    m = {}
    for split_name in ("train", "val", "test"):
        for sid in manifest[split_name]["storm_ids"]:
            m[sid] = split_name
    return m
