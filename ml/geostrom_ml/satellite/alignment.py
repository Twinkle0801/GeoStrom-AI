"""HURSAT <-> IBTrACS temporal alignment and spatial QC.

Implements the Phase 1 verified join strategy (docs/DATA_STRATEGY.md check
#1, ml/scripts/verify_hursat_join.py `temporal_join`/`spatial_agreement`):
storm identity + nearest valid IBTrACS observation in time, within an
explicit configurable tolerance, gated by great-circle spatial agreement.

Deliberately joins against the FULL IBTrACS track for each storm (every
IFLAG/TRACK_TYPE value), not the Phase 2 "usable rows only" subset --
because a satellite frame can legitimately land nearest an interpolated or
non-main-track best-track row, and the Phase 4 task requires preserving the
observed / interpolated / missing distinction rather than silently
pre-filtering it away. A frame that matches only a non-'main'/interpolated
row is still recorded (with `is_observed=False`), never discarded for that
reason alone -- it is discarded only by the spatial/temporal QC gates below.

Geodesic math is imported directly from `ml.geostrom_ml.features.geo`
(Phase 2's validated Haversine implementation) -- this module lives inside
`ml/`, so, unlike `backend/`, there is no "never import from ml/" rule to
respect here; re-implementing Haversine a third time would be pure
duplication.
"""

from __future__ import annotations

import pandas as pd

from ml.geostrom_ml.features.geo import haversine_km
from ml.geostrom_ml.satellite.schema import DEFAULT_TEMPORAL_TOLERANCE_MIN, SPATIAL_QC_KM


def join_frames_to_ibtracs(
    canonical: pd.DataFrame,
    ibtracs_full_track: pd.DataFrame,
    *,
    tolerance_min: int = DEFAULT_TEMPORAL_TOLERANCE_MIN,
    spatial_qc_km: float = SPATIAL_QC_KM,
) -> pd.DataFrame:
    """Join canonical HURSAT frames to the nearest IBTrACS row per storm.

    `ibtracs_full_track` must have columns: SID, ISO_TIME, LAT, LON, IFLAG,
    TRACK_TYPE, USA_WIND, USA_PRES (the full per-storm track, not
    pre-filtered to synoptic/observed-only rows).

    Adds: ibtracs_timestamp, ibtracs_lat, ibtracs_lon, spatial_distance_km,
    temporal_offset_minutes, is_observed, is_interpolated, usa_wind,
    pressure_if_valid, qc_status ('ok' | 'rejected'), qc_reason (machine
    readable; None when qc_status == 'ok').
    """
    out_rows = []
    track_by_sid = {sid: g.sort_values("ISO_TIME") for sid, g in ibtracs_full_track.groupby("SID")}

    for storm_id, grp in canonical.groupby("storm_id", dropna=False):
        track = track_by_sid.get(storm_id)
        left = grp.sort_values("satellite_timestamp")
        if track is None or track.empty:
            for _, fr in left.iterrows():
                r = fr.to_dict()
                r.update(_empty_join_fields())
                r["qc_status"] = "rejected"
                r["qc_reason"] = "storm_id_not_in_ibtracs"
                out_rows.append(r)
            continue

        merged = pd.merge_asof(
            left, track[["SID", "ISO_TIME", "LAT", "LON", "IFLAG", "TRACK_TYPE",
                          "USA_WIND", "USA_PRES"]].rename(columns={"SID": "_sid_track"}),
            left_on="satellite_timestamp", right_on="ISO_TIME",
            direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_min),
        )

        for _, r in merged.iterrows():
            row = r.to_dict()
            row.pop("_sid_track", None)
            matched = pd.notna(row.get("ISO_TIME"))
            if not matched:
                row.update(_empty_join_fields())
                row["qc_status"] = "rejected"
                row["qc_reason"] = f"no_ibtracs_row_within_{tolerance_min}min"
                out_rows.append(row)
                continue

            dt_min = abs((row["satellite_timestamp"] - row["ISO_TIME"]).total_seconds()) / 60.0
            sep_km = None
            if pd.notna(row.get("satellite_lat")) and pd.notna(row.get("LAT")):
                sep_km = float(haversine_km(row["satellite_lat"], row["satellite_lon"],
                                             row["LAT"], row["LON"]))

            iflag0 = str(row["IFLAG"])[0] if pd.notna(row.get("IFLAG")) else None
            row["ibtracs_timestamp"] = row["ISO_TIME"]
            row["ibtracs_lat"] = row["LAT"]
            row["ibtracs_lon"] = row["LON"]
            row["spatial_distance_km"] = sep_km
            row["temporal_offset_minutes"] = dt_min
            row["is_observed"] = bool(iflag0 == "O")
            row["is_interpolated"] = bool(iflag0 is not None and iflag0 != "O")
            row["usa_wind"] = row.get("USA_WIND")
            row["pressure_if_valid"] = row.get("USA_PRES")

            if sep_km is None:
                row["qc_status"] = "rejected"
                row["qc_reason"] = "missing_position_for_spatial_qc"
            elif sep_km >= spatial_qc_km:
                row["qc_status"] = "rejected"
                row["qc_reason"] = f"spatial_separation_{sep_km:.1f}km_exceeds_{spatial_qc_km:.0f}km_gate"
            else:
                row["qc_status"] = "ok"
                row["qc_reason"] = None
            out_rows.append(row)

    result = pd.DataFrame(out_rows)
    drop_cols = [c for c in ("ISO_TIME", "LAT", "LON", "IFLAG", "TRACK_TYPE", "USA_WIND", "USA_PRES")
                 if c in result.columns]
    return result.drop(columns=drop_cols)


def _empty_join_fields() -> dict:
    return {
        "ibtracs_timestamp": pd.NaT, "ibtracs_lat": None, "ibtracs_lon": None,
        "spatial_distance_km": None, "temporal_offset_minutes": None,
        "is_observed": None, "is_interpolated": None,
        "usa_wind": None, "pressure_if_valid": None,
    }
