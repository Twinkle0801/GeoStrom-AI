"""Phase 4 satellite dataset manifest.

Same spirit as `ml/scripts/make_manifest.py` (Phase 1) and
`ml/manifests/dataset_v1_manifest.json` (Phase 2): a small, committed,
machine-readable provenance record. Never contains absolute,
machine-specific paths -- only DATA_ROOT-relative ones, per the Phase 4
task's explicit instruction.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd

from ml.geostrom_ml.satellite.schema import (
    DATASET_VERSION,
    DEFAULT_TEMPORAL_TOLERANCE_MIN,
    IRWIN_VALID_RANGE_K,
    PREPROCESSING_VERSION,
    SPATIAL_QC_KM,
)


def build_manifest(
    *,
    data_root: Path,
    final_index: pd.DataFrame,
    qc_report: dict,
    basin: str,
    seasons_covered: list[int],
    storms_requested: list[str],
    zarr_path: Path,
    parquet_path: Path,
    splits_path_repo_relative: str,
    code_version: str | None = None,
) -> dict:
    counts = qc_report["counts"]
    n_scene_labeled = int(final_index["scene_label"].notna().sum()) if len(final_index) else 0

    def rel(p: Path) -> str:
        try:
            return os.path.relpath(Path(p), data_root).replace("\\", "/")
        except ValueError:
            return str(p)

    return {
        "manifest_version": 1,
        "project": "GeoStrom AI",
        "phase": "Phase 4 - Satellite Data Pipeline",
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_version": code_version,
        "source_datasets": [
            {"name": "IBTrACS", "version": "v04r01",
             "role": "best-track spine; wind=USA_WIND (1-min convention)"},
            {"name": "HURSAT-B1", "version": "v06",
             "role": "storm-centric IR imagery, IRWIN channel"},
            {"name": "ADT-HURSAT", "version": "ADT v9.0 over HURSAT V07b (NCEI Accession 0307249, v1.1)",
             "role": "Scene classification label only -- NEVER intensity ground truth"},
        ],
        "selection": {
            "basin": basin,
            "seasons_covered": seasons_covered,
            "storms_requested": len(storms_requested),
        },
        "counts": counts,
        "qc_thresholds": {
            "irwin_valid_range_k": list(IRWIN_VALID_RANGE_K),
            "spatial_qc_km": SPATIAL_QC_KM,
            "temporal_tolerance_min": DEFAULT_TEMPORAL_TOLERANCE_MIN,
        },
        "scene_labeled_samples": n_scene_labeled,
        "gate_status": qc_report["summary"]["gate_status"],
        "storage": {
            "zarr_path_relative_to_data_root": rel(zarr_path),
            "sample_index_path_relative_to_data_root": rel(parquet_path),
        },
        "split_source": {
            "manifest": splits_path_repo_relative,
            "note": "Storm-level split reused verbatim from Phase 2; no new split was created.",
        },
        "data_root_note": ("Data lives outside the Git repository and outside OneDrive. "
                           "Only this manifest, the QC report, and the small figures are "
                           "version-controlled."),
    }
