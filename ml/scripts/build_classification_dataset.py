"""Phase 5 Task 6: build the canonical classification dataset index.

Reads Phase 4's sample index, applies `scene_taxonomy_v1`
(`ml/geostrom_ml/classification/taxonomy.py`), and writes:

  DATA_ROOT/processed/classification/<label_version>/classification_index.parquet
  ml/manifests/classification_dataset_v1_manifest.json   (committed, small)

Never modifies Phase 4's own Zarr/Parquet outputs or `splits_v1.json`.

Usage:
    python ml/scripts/build_classification_dataset.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, get_data_root  # noqa: E402
from ml.geostrom_ml.classification.dataset import build_classification_index, split_summary  # noqa: E402
from ml.geostrom_ml.classification.taxonomy import (  # noqa: E402
    EXCLUSION_REASONS, FINAL_CLASSES_V1, LABEL_VERSION, SCENE_TAXONOMY_V1,
)
from ml.geostrom_ml.satellite.schema import DATASET_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    ap.add_argument("--out-manifest", type=Path,
                    default=MANIFEST_DIR / "classification_dataset_v1_manifest.json")
    args = ap.parse_args()

    import pandas as pd

    root = get_data_root()
    parquet_path = root / "processed" / "satellite" / args.dataset_version / "sample_index.parquet"
    if not parquet_path.exists():
        print(f"No Phase 4 sample index at {parquet_path}.", file=sys.stderr)
        return 1

    sample_index = pd.read_parquet(parquet_path)
    classification_index = build_classification_index(sample_index)

    out_dir = root / "processed" / "classification" / LABEL_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "classification_index.parquet"
    classification_index.to_parquet(out_path, index=False)
    print(f"Classification index written: {out_path} ({len(classification_index)} rows)")

    summary = split_summary(classification_index)
    print(f"Included: {summary['total_included']}  Excluded: {summary['total_excluded']}")
    print(f"Excluded by reason: {summary['excluded_by_reason']}")
    print(f"Samples by split (included only): {summary['samples_by_split']}")
    print(f"Storms by split (included only): {summary['storms_by_split']}")

    manifest = {
        "label_version": LABEL_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_dataset_version": args.dataset_version,
        "source": "ml/manifests/satellite_dataset_v1_manifest.json (Phase 4, unmodified)",
        "taxonomy_mapping": SCENE_TAXONOMY_V1,
        "exclusion_reasons": EXCLUSION_REASONS,
        "final_classes": FINAL_CLASSES_V1,
        "split_summary": summary,
        "index_path_relative_to_data_root": "processed/classification/" + LABEL_VERSION
                                            + "/classification_index.parquet",
        "total_rows": int(len(classification_index)),
        "note": "original_scene is preserved verbatim on every row; no Phase 4 artifact was modified.",
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Manifest written: {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
