"""End-to-end satellite dataset build orchestration.

RAW SATELLITE DATA -> HURSAT ingestion -> metadata extraction -> IRWIN QC ->
IBTrACS join -> ADT-HURSAT Scene join -> duplicate resolution -> QC gate ->
dataset manifest -> Zarr + sample index.

Deterministic end to end: given the same files on disk, the same
`splits_v1.json`, and the same configuration, this produces byte-identical
`final_index` rows (only `source_checksum`/wall-clock-free fields) and an
identical Zarr store every time -- no randomness anywhere in this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("geostrom_ml.satellite.pipeline")

from ml.geostrom_ml.satellite import adt as adt_mod
from ml.geostrom_ml.satellite import dedup as dedup_mod
from ml.geostrom_ml.satellite import hursat as hursat_mod
from ml.geostrom_ml.satellite import qc as qc_mod
from ml.geostrom_ml.satellite.alignment import join_frames_to_ibtracs
from ml.geostrom_ml.satellite.download import sha256_of
from ml.geostrom_ml.satellite.imagery import SatelliteZarrStore
from ml.geostrom_ml.satellite.schema import (
    DATASET_VERSION,
    DEFAULT_TEMPORAL_TOLERANCE_MIN,
    PREPROCESSING_VERSION,
    SAMPLE_COLUMNS,
    SAMPLE_DTYPES,
    SPATIAL_QC_KM,
    make_sample_id,
)


def load_split_map(splits_path: Path) -> dict[str, str]:
    """storm_id -> 'train'|'val'|'test', from the frozen Phase 2 manifest."""
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for sid in data.get(split_name, {}).get("storm_ids", []):
            mapping[sid] = split_name
    return mapping


def run_pipeline(
    *,
    interim_hursat_dir: Path | list[Path],
    adt_dir: Path,
    splits_path: Path,
    ibtracs_full_track: pd.DataFrame,
    zarr_out_path: Path,
    tolerance_min: int = DEFAULT_TEMPORAL_TOLERANCE_MIN,
    spatial_qc_km: float = SPATIAL_QC_KM,
    dataset_version: str = DATASET_VERSION,
    preprocessing_version: str = PREPROCESSING_VERSION,
    write_zarr: bool = True,
) -> dict:
    frame_paths = hursat_mod.discover_frame_files(interim_hursat_dir)
    logger.info("discovered %d candidate HURSAT frame files under %s", len(frame_paths), interim_hursat_dir)
    inventory = hursat_mod.inventory_frames(frame_paths)
    logger.info("parsed metadata for %d frames", len(inventory))

    ok_mask = inventory["error"].isna() if "error" in inventory else pd.Series([True] * len(inventory))
    parsed_ok = inventory[ok_mask].copy()
    dup_summary = dedup_mod.duplicate_summary(parsed_ok)

    if parsed_ok.empty:
        canonical = parsed_ok
        rejected_dup = parsed_ok
    else:
        canonical, rejected_dup = dedup_mod.select_canonical_frames(parsed_ok)
    logger.info("deduplicated to %d canonical frames (%d rejected duplicates)",
                len(canonical), len(rejected_dup))

    split_map = load_split_map(splits_path)
    known_split_storm_ids = set(split_map.keys())

    if canonical.empty:
        joined = canonical
    else:
        joined = join_frames_to_ibtracs(
            canonical, ibtracs_full_track,
            tolerance_min=tolerance_min, spatial_qc_km=spatial_qc_km,
        )
    logger.info("IBTrACS join complete: %d ok, %d rejected",
                int((joined.get("qc_status") == "ok").sum()) if len(joined) else 0,
                int((joined.get("qc_status") == "rejected").sum()) if len(joined) else 0)

    ok_rows = joined[joined.get("qc_status") == "ok"].copy() if len(joined) else joined

    if ok_rows is not None and len(ok_rows):
        fused = adt_mod.join_adt_scene(ok_rows, adt_dir, tolerance_min=tolerance_min)
    else:
        fused = ok_rows
    logger.info("ADT join complete")

    split_rejected = pd.DataFrame()
    if fused is not None and len(fused):
        split_series = fused["storm_id"].map(split_map)
        unknown_mask = split_series.isna()
        if unknown_mask.any():
            split_rejected = fused[unknown_mask].copy()
            split_rejected["qc_reason"] = "storm_not_in_frozen_split_manifest"
        fused = fused[~unknown_mask].copy()
        fused["split"] = split_series[~unknown_mask]

    if fused is not None and len(fused):
        fused["season"] = fused["storm_id"].str[:4].astype(int)
        fused = fused.sort_values(["storm_id", "satellite_timestamp"], kind="mergesort").reset_index(drop=True)
        fused["zarr_index"] = fused.index
        fused["sample_id"] = [make_sample_id(sid, ts) for sid, ts in
                               zip(fused["storm_id"], fused["satellite_timestamp"])]
        fused["dataset_version"] = dataset_version
        fused["preprocessing_version"] = preprocessing_version
        fused["channel"] = "IRWIN"
        fused["image_height"] = hursat_mod.EXPECTED_GRID[0]
        fused["image_width"] = hursat_mod.EXPECTED_GRID[1]
        fused["source_checksum"] = [sha256_of(Path(p)) for p in fused["source_file"]]

    final_index = _select_and_order_columns(fused)
    logger.info("final fused sample index: %d rows", len(final_index))

    zarr_store = None
    if write_zarr and len(final_index):
        store = SatelliteZarrStore(zarr_out_path).create(len(final_index), overwrite=True)
        n = len(final_index)
        for i, (_, row) in enumerate(final_index.sort_values("zarr_index").iterrows()):
            kelvin, valid_mask = hursat_mod.read_irwin(Path(row["source_file"]))
            store.write_frame(int(row["zarr_index"]), kelvin, valid_mask)
            if (i + 1) % 50 == 0 or (i + 1) == n:
                logger.info("Zarr write progress: %d/%d frames", i + 1, n)
        zarr_store = store

    qc_report = qc_mod.build_qc_report(
        n_files_discovered=len(frame_paths),
        inventory=inventory,
        duplicate_summary=dup_summary,
        canonical=canonical,
        rejected_duplicates=rejected_dup,
        joined=joined,
        final_index=final_index,
        known_split_storm_ids=known_split_storm_ids,
    )

    return {
        "inventory": inventory,
        "canonical": canonical,
        "rejected_duplicates": rejected_dup,
        "joined": joined,
        "split_rejected": split_rejected,
        "final_index": final_index,
        "qc_report": qc_report,
        "zarr_store_path": str(zarr_out_path) if zarr_store else None,
    }


def _select_and_order_columns(fused: pd.DataFrame | None) -> pd.DataFrame:
    if fused is None or fused.empty:
        out = pd.DataFrame(columns=SAMPLE_COLUMNS)
    else:
        missing = [c for c in SAMPLE_COLUMNS if c not in fused.columns]
        for c in missing:
            fused[c] = None
        out = fused[SAMPLE_COLUMNS].reset_index(drop=True)
    # Enforce the committed schema's dtypes (schema.SAMPLE_DTYPES) so the
    # written Parquet has a predictable, documented column schema rather
    # than whatever pandas happened to infer along the way.
    for col, dtype in SAMPLE_DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    return out
