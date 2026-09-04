"""Deterministic duplicate-satellite-frame resolution.

Phase 1 discovered simultaneous satellite views: the same (storm_id,
satellite_timestamp) can have frames from more than one geostationary
satellite (e.g. GOES-East and GOES-West both viewing the same storm). Phase
1's verified rule (docs/DATA_STRATEGY.md, ml/scripts/verify_hursat_join.py
`duplicate_frames` section): keep the frame with the SMALLEST viewing zenith
angle (VZA) -- the most direct, least oblique view of the storm.

This module implements that rule with a fully deterministic, documented
tie-break chain so re-running the pipeline on unchanged inputs always
reaches the same selection (never `pandas.sample`, `set` iteration order, or
any other non-deterministic mechanism).
"""

from __future__ import annotations

import pandas as pd


def select_canonical_frames(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve (storm_id, satellite_timestamp) duplicates to one canonical frame.

    Selection order (all deterministic):
      1. lowest VZA (most direct view) wins;
      2. if VZA is tied or missing on all candidates, lowest `source_file`
         path string wins (a stable, arbitrary-but-reproducible tie-break).

    Returns (canonical_df, rejected_log_df). `rejected_log_df` has one row
    per NON-selected duplicate candidate with a machine-readable reason.
    """
    required = {"storm_id", "satellite_timestamp", "vza", "source_file"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory is missing required columns: {sorted(missing)}")

    df = inventory.copy()
    df["_vza_sort"] = pd.to_numeric(df["vza"], errors="coerce").fillna(float("inf"))
    df = df.sort_values(
        ["storm_id", "satellite_timestamp", "_vza_sort", "source_file"],
        kind="mergesort",  # stable sort -> deterministic given the columns above
    )

    group_cols = ["storm_id", "satellite_timestamp"]
    df["_rank"] = df.groupby(group_cols, dropna=False).cumcount()
    df["_group_size"] = df.groupby(group_cols, dropna=False)["source_file"].transform("size")

    canonical = df[df["_rank"] == 0].drop(columns=["_vza_sort", "_rank", "_group_size"])

    rejected_mask = df["_rank"] > 0
    rejected = df[rejected_mask].copy()
    if len(rejected):
        vza_known = rejected["vza"].notna()
        rejected["rejection_reason"] = "duplicate_frame_higher_vza"
        rejected.loc[~vza_known, "rejection_reason"] = "duplicate_frame_no_vza_tiebreak_filename"
    rejected = rejected.drop(columns=["_vza_sort", "_rank", "_group_size"], errors="ignore")

    return canonical.reset_index(drop=True), rejected.reset_index(drop=True)


def duplicate_summary(inventory: pd.DataFrame) -> dict:
    """Counts for the QC gate: candidates, duplicated groups, unique frames."""
    if inventory.empty:
        return {"n_candidates": 0, "n_groups": 0, "n_groups_with_duplicates": 0,
                "n_unique_frames": 0, "max_frames_at_one_time": 0}
    sizes = inventory.groupby(["storm_id", "satellite_timestamp"], dropna=False).size()
    return {
        "n_candidates": int(len(inventory)),
        "n_groups": int(len(sizes)),
        "n_groups_with_duplicates": int((sizes > 1).sum()),
        "n_unique_frames": int(len(sizes)),
        "max_frames_at_one_time": int(sizes.max()),
    }
