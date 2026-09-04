"""Satellite dataset QC gate: the Phase 4 task's explicit 18-point report.

Mirrors the Check/report pattern established in `ml/scripts/qc_gate.py`
(Phase 1) rather than inventing a new reporting style. Blocking checks fail
the gate (non-zero exit from the calling script); informational checks are
always reported but never fail the gate on their own.

"The QC gate should fail loudly when critical assumptions are violated. Do
not hide dropped samples." -- every count below is either a real total or
explicitly computed from the intermediate DataFrames the pipeline produced;
nothing here is fabricated or estimated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class QCCheck:
    id: str
    name: str
    blocking: bool
    threshold: str
    passed: bool | None = None
    value: object = None
    detail: dict = field(default_factory=dict)


def build_qc_report(
    *,
    n_files_discovered: int,
    inventory: pd.DataFrame,
    duplicate_summary: dict,
    canonical: pd.DataFrame,
    rejected_duplicates: pd.DataFrame,
    joined: pd.DataFrame,
    final_index: pd.DataFrame,
    known_split_storm_ids: set[str],
) -> dict:
    checks: list[QCCheck] = []

    parse_errors = (inventory["error"].notna() if "error" in inventory
                    else pd.Series([False] * len(inventory)))
    n_parsed_ok = int((~parse_errors).sum())
    n_parse_failed = int(parse_errors.sum())

    irwin_valid_pct = inventory.get("irwin_valid_pct", pd.Series(dtype=float))
    n_valid_irwin = int((irwin_valid_pct > 0).sum())

    n_ok = int((joined.get("qc_status") == "ok").sum()) if len(joined) else 0
    n_rejected = int((joined.get("qc_status") == "rejected").sum()) if len(joined) else 0
    spatial_fail = int(joined["qc_reason"].astype(str).str.startswith("spatial_separation").sum()) \
        if "qc_reason" in joined else 0
    temporal_fail = int(joined["qc_reason"].astype(str).str.startswith("no_ibtracs_row").sum()) \
        if "qc_reason" in joined else 0

    # `adt_qc_status` (set by adt.join_adt_scene) is not part of the committed
    # SAMPLE_COLUMNS schema, so it does not survive final-index column
    # selection; `scene_label` presence is an exact proxy for a matched ADT
    # join (join_adt_scene sets both together, see satellite/adt.py).
    adt_matched = int(final_index["scene_label"].notna().sum()) if "scene_label" in final_index else 0
    adt_unmatched = int(final_index["scene_label"].isna().sum()) if "scene_label" in final_index else 0

    scene_dist = (final_index["scene_label"].dropna().value_counts().to_dict()
                  if "scene_label" in final_index and len(final_index) else {})
    per_season = (final_index["season"].value_counts().sort_index().to_dict()
                  if "season" in final_index and len(final_index) else {})
    per_storm = (final_index["storm_id"].value_counts().to_dict()
                 if "storm_id" in final_index and len(final_index) else {})

    missing_stats = {
        col: int(final_index[col].isna().sum())
        for col in ("scene_label", "usa_wind", "pressure_if_valid", "vza")
        if col in final_index
    }

    # ---- blocking checks --------------------------------------------------
    dup_ids = (final_index.groupby(["storm_id", "satellite_timestamp"]).size()
               if len(final_index) else pd.Series(dtype=int))
    n_dup_in_final = int((dup_ids > 1).sum()) if len(dup_ids) else 0
    checks.append(QCCheck("Q1", "No duplicate (storm_id, satellite_timestamp) in final index",
                          True, "== 0", n_dup_in_final == 0, n_dup_in_final))

    unknown_storms = (set(final_index["storm_id"].unique()) - known_split_storm_ids
                      if len(final_index) else set())
    checks.append(QCCheck("Q2", "Every fused sample's storm belongs to the frozen split manifest",
                          True, "== 0 unknown storms", len(unknown_storms) == 0,
                          len(unknown_storms), {"unknown_storms": sorted(unknown_storms)[:10]}))

    bad_spatial = (int((final_index["spatial_distance_km"] >= 50.0).sum())
                   if "spatial_distance_km" in final_index and len(final_index) else 0)
    checks.append(QCCheck("Q3", "No fused sample exceeds the 50 km spatial QC gate",
                          True, "== 0", bad_spatial == 0, bad_spatial))

    bad_status = (int((final_index["qc_status"] != "ok").sum())
                  if "qc_status" in final_index and len(final_index) else 0)
    checks.append(QCCheck("Q4", "Every row in the final index has qc_status == 'ok'",
                          True, "== 0 non-ok rows", bad_status == 0, bad_status))

    checks.append(QCCheck("Q5", "No open/parse errors among discovered frames",
                          False, "== 0", n_parse_failed == 0, n_parse_failed))

    blocking_failed = [c.id for c in checks if c.blocking and c.passed is False]

    return {
        "gate": "phase4_satellite",
        "counts": {
            "1_total_files_discovered": int(n_files_discovered),
            "2_files_successfully_parsed": n_parsed_ok,
            "3_files_rejected": n_parse_failed,
            "4_frames_discovered": int(len(inventory)),
            "5_frames_with_valid_irwin": n_valid_irwin,
            "6_duplicate_frames": duplicate_summary.get("n_groups_with_duplicates", 0),
            "7_unique_frames": duplicate_summary.get("n_unique_frames", 0),
            "8_successful_ibtracs_joins": n_ok,
            "9_failed_ibtracs_joins": n_rejected,
            "10_spatial_qc_failures": spatial_fail,
            "11_temporal_qc_failures": temporal_fail,
            "12_adt_matches": adt_matched,
            "13_adt_unmatched_samples": adt_unmatched,
            "14_final_fused_samples": int(len(final_index)),
            "15_scene_class_distribution": scene_dist,
            "16_missing_value_statistics": missing_stats,
            "17_per_season_counts": {str(k): int(v) for k, v in per_season.items()},
            "18_per_storm_counts": {str(k): int(v) for k, v in per_storm.items()},
        },
        "checks": [asdict(c) for c in checks],
        "summary": {
            "total_checks": len(checks),
            "passed": sum(1 for c in checks if c.passed),
            "failed": sum(1 for c in checks if c.passed is False),
            "blocking_failures": blocking_failed,
            "gate_status": "PASS" if not blocking_failed else "FAIL",
        },
    }
