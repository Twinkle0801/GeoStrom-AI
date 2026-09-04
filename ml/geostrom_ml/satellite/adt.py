"""ADT-HURSAT Scene-label loading and joining.

Consolidates the validated Phase 1 ADT parsing logic (originally inline in
`ml/scripts/verify_adt.py`) into a reusable library component, per the same
reuse rule already applied elsewhere in this package. `verify_adt.py` is
left unmodified as the Phase 1 historical artifact.

ADT-HURSAT `Scene` is the recommended primary classification target (Phase 1
finding, docs/DATA_STRATEGY.md decision #14). It is joined on TRUE ADT scan
time, not the nominal HURSAT `htime`, because ADT records genuine scan times
(docs/DATA_STRATEGY.md check #13/#17) rather than synchronised slots.

CRITICAL, per the Phase 4 task: ADT-HURSAT must NEVER replace IBTrACS
intensity ground truth. This module only ever writes a `scene_label` (+
provenance) column; it never writes `usa_wind` or any field that Phase 2/3
treat as the intensity target. A fused sample lacking a matched ADT record
still keeps its (IBTrACS-sourced) intensity fields -- the ADT join is
optional and non-blocking, exactly as `docs/DATA_STRATEGY.md` §4.3 specifies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

ADT_FIELDS = ("Date", "Time", "Scene", "EyeScene", "CloudScene", "CI", "Lat", "Lon", "VZA")


def parse_adt_time(date_s: str, time_s: str) -> pd.Timestamp:
    """ADT Date='2005AUG26', Time='174513' (HHMMSS)."""
    t = str(time_s).zfill(6)
    return pd.to_datetime(f"{date_s} {t[:2]}:{t[2:4]}:{t[4:6]}",
                          format="%Y%b%d %H:%M:%S", errors="coerce")


def load_adt_storm(sid: str, adt_dir: Path) -> pd.DataFrame | None:
    """Load one storm's ADT-HURSAT record file, if present.

    Returns None (not an error) when the storm has no ADT file -- ADT
    coverage is expected to be incomplete; see docs/DATA_STRATEGY.md §4.3.
    """
    f = adt_dir / f"{sid}.nc"
    if not f.exists():
        return None
    with xr.open_dataset(f) as ds:
        data = {k: ds[k].values for k in ADT_FIELDS if k in ds.variables}
    df = pd.DataFrame(data)
    if df.empty:
        return None
    df["adt_time"] = [parse_adt_time(d, t) for d, t in zip(df["Date"], df["Time"])]
    df = df.dropna(subset=["adt_time"]).sort_values("adt_time").reset_index(drop=True)
    df["sid"] = sid
    return df


def join_adt_scene(
    joined: pd.DataFrame,
    adt_dir: Path,
    *,
    tolerance_min: int = 90,
) -> pd.DataFrame:
    """Attach `scene_label`, `adt_timestamp`, and ADT match status.

    Only operates on rows already `qc_status == 'ok'` from the IBTrACS join;
    the ADT match itself never rejects a sample (adt_qc_status is reported
    separately in the QC gate, not folded into `qc_status`).
    """
    out = joined.copy()
    out["scene_label"] = pd.Series([None] * len(out), dtype="object")
    out["adt_timestamp"] = pd.NaT
    out["adt_qc_status"] = "unmatched"

    cache: dict[str, pd.DataFrame | None] = {}
    for storm_id, idx in out.groupby("storm_id", dropna=False).groups.items():
        if storm_id not in cache:
            cache[storm_id] = load_adt_storm(storm_id, adt_dir)
        adt = cache[storm_id]
        if adt is None or adt.empty:
            continue

        sub = out.loc[idx].sort_values("satellite_timestamp")
        merged = pd.merge_asof(
            sub[["satellite_timestamp"]], adt[["adt_time", "Scene"]],
            left_on="satellite_timestamp", right_on="adt_time",
            direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_min),
        )
        merged.index = sub.index
        matched = merged["adt_time"].notna()
        out.loc[merged.index[matched], "scene_label"] = merged.loc[matched, "Scene"].values
        out.loc[merged.index[matched], "adt_timestamp"] = merged.loc[matched, "adt_time"].values
        out.loc[merged.index[matched], "adt_qc_status"] = "matched"

    return out
