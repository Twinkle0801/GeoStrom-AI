"""Build and freeze the Phase 2 train/val/test split.

Writes ml/manifests/splits_v1.json. This file is committed to the repository
(it is small, storm-ID metadata only -- no raw data) and, once written, is
treated as FROZEN: re-running this script produces byte-for-byte the same
output (the split rule is deterministic, not seeded/random), but the correct
workflow is to bump SPLIT_VERSION in split.py if the methodology ever needs
to change, not to overwrite v1 silently.

Usage:
    python ml/scripts/build_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.splits.split import build_split_manifest, write_split_manifest  # noqa: E402


def main() -> int:
    manifest = build_split_manifest()
    path = write_split_manifest(manifest)

    print(f"Split version   : {manifest['split_version']}")
    print(f"Basin           : {manifest['basin']}   seasons {manifest['season_range']}")
    print(f"Wind column     : {manifest['wind_column']}")
    print()
    for name in ("train", "val", "test"):
        s = manifest[name]
        print(f"{name.upper():<6} seasons {s['seasons'][0]}-{s['seasons'][1]:<6} "
              f"storms={s['n_storms']:<5} observations={s['n_observations']:<6}")
    print()
    ic = manifest["integrity_check"]
    print(f"Integrity check : all_disjoint={ic['all_disjoint']}")
    print(f"  train n val   = {ic['intersection_train_val']}")
    print(f"  train n test  = {ic['intersection_train_test']}")
    print(f"  val n test    = {ic['intersection_val_test']}")
    sc = manifest["sid_season_cross_check"]
    print(f"SID/season cross-check: {sc['n_checked']} storms checked, "
          f"{sc['n_mismatches']} mismatches")
    print()
    print(f"WRITTEN: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
