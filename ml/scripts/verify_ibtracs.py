"""IBTrACS schema verification (Phase 1, Task 4).

Loads an IBTrACS basin CSV and reports structure, field availability, missing-value
rates, temporal coverage, synoptic-hour distribution, observed-vs-interpolated
composition, duplicate keys, and cross-agency wind convention availability.

Read-only: never modifies the downloaded file.

Usage:
    python ml/scripts/verify_ibtracs.py --csv <path> [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR  # noqa: E402

# IBTrACS sentinels.
#
# CRITICAL: `keep_default_na` MUST be False. pandas' default NA list contains the
# literal string "NA", which is also the IBTrACS basin code for the North Atlantic.
# With defaults enabled, every North Atlantic row silently gets BASIN=NaN. The same
# trap applies to SUBBASIN and to any agency field whose code is "NA".
# Verified on ibtracs.NA.list.v04r01.csv: 126,586 of 127,188 rows were corrupted
# before this was fixed.
NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]

# Wind averaging period per agency, from the official v04r01 column documentation.
AGENCY_WIND = {
    "USA_WIND": ("US agencies (NHC/JTWC/HURDAT/ATCF)", "1-minute"),
    "TOKYO_WIND": ("RSMC Tokyo (JMA)", "10-minute"),
    "NEWDELHI_WIND": ("RSMC New Delhi (IMD)", "3-minute"),
    "REUNION_WIND": ("RSMC La Reunion (MFLR)", "10-minute"),
    "BOM_WIND": ("Australian TCWCs", "10-minute"),
    "NADI_WIND": ("RSMC Nadi (FMS)", "10-minute"),
    "WELLINGTON_WIND": ("TCWC Wellington (NZMS)", "10-minute"),
    "CMA_WIND": ("CMA (Shanghai)", "2-minute"),
    "HKO_WIND": ("Hong Kong Observatory", "10-minute"),
    "WMO_WIND": ("WMO-responsible agency (MIXED conventions)", "MIXED - unsafe"),
}

PHYSICAL_RANGES = {
    "LAT": (-90.0, 90.0),
    "LON": (-180.0, 360.0),
    "USA_WIND": (0.0, 300.0),
    "WMO_WIND": (0.0, 300.0),
    "USA_PRES": (850.0, 1050.0),
    "WMO_PRES": (850.0, 1050.0),
}


def load_ibtracs(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load IBTrACS CSV, skipping the units row. Returns (df, units_row)."""
    header = pd.read_csv(csv_path, nrows=1)
    units = pd.read_csv(csv_path, skiprows=1, nrows=1, header=None)
    units_row = [str(v) for v in units.iloc[0].tolist()]

    df = pd.read_csv(
        csv_path,
        skiprows=[1],                 # drop the units row
        na_values=NA_VALUES,
        keep_default_na=False,        # see NA_VALUES note: protects basin code "NA"
        low_memory=False,
    )
    assert list(df.columns) == list(header.columns), "header/data column mismatch"
    return df, units_row


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 3) if total else 0.0


def verify(csv_path: Path) -> dict:
    rep: dict = {"file": str(csv_path), "file_bytes": csv_path.stat().st_size}
    df, units_row = load_ibtracs(csv_path)

    # ---- structure -------------------------------------------------------
    rep["n_rows"], rep["n_columns"] = int(df.shape[0]), int(df.shape[1])
    rep["units_row_present"] = True
    rep["units_row_sample"] = dict(list(zip(df.columns, units_row))[:14])
    rep["columns"] = list(df.columns)

    # ---- typing ----------------------------------------------------------
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=False)
    for col in ("LAT", "LON", "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES",
                "STORM_SPEED", "STORM_DIR", "DIST2LAND", "LANDFALL", "USA_SSHS"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- candidate key fields -------------------------------------------
    rep["candidate_fields"] = {
        "storm_id": [c for c in ("SID", "USA_ATCF_ID", "NUMBER", "NAME") if c in df.columns],
        "timestamp": [c for c in ("ISO_TIME",) if c in df.columns],
        "latitude": [c for c in ("LAT", "USA_LAT") if c in df.columns],
        "longitude": [c for c in ("LON", "USA_LON") if c in df.columns],
        "wind": [c for c in df.columns if c.endswith("_WIND")],
        "pressure": [c for c in df.columns if c.endswith("_PRES")],
        "agency": [c for c in ("WMO_AGENCY", "USA_AGENCY") if c in df.columns],
        "observed_flag": [c for c in ("IFLAG", "TRACK_TYPE") if c in df.columns],
        "nature_category": [c for c in ("NATURE", "USA_SSHS", "USA_STATUS") if c in df.columns],
    }

    # ---- coverage --------------------------------------------------------
    rep["time_range"] = {
        "min": str(df["ISO_TIME"].min()),
        "max": str(df["ISO_TIME"].max()),
        "n_unmparseable": int(df["ISO_TIME"].isna().sum()),
    }
    rep["n_storms"] = int(df["SID"].nunique())
    rep["season_range"] = [int(df["SEASON"].min()), int(df["SEASON"].max())]
    rep["basins"] = df["BASIN"].value_counts(dropna=False).to_dict()
    rep["subbasins"] = df["SUBBASIN"].value_counts(dropna=False).head(10).to_dict()

    # ---- missing-value rates --------------------------------------------
    interesting = [
        "SID", "ISO_TIME", "LAT", "LON", "NATURE", "TRACK_TYPE", "IFLAG",
        "WMO_WIND", "WMO_PRES", "WMO_AGENCY", "USA_WIND", "USA_PRES", "USA_SSHS",
        "USA_ATCF_ID", "USA_AGENCY", "USA_STATUS", "USA_RMW",
        "STORM_SPEED", "STORM_DIR", "DIST2LAND", "LANDFALL",
        "TOKYO_WIND", "NEWDELHI_WIND", "REUNION_WIND", "BOM_WIND", "CMA_WIND",
    ]
    rep["missing_pct"] = {
        c: pct(int(df[c].isna().sum()), len(df)) for c in interesting if c in df.columns
    }
    rep["dtypes"] = {c: str(df[c].dtype) for c in interesting if c in df.columns}

    # ---- synoptic structure ---------------------------------------------
    hours = df["ISO_TIME"].dt.hour
    minutes = df["ISO_TIME"].dt.minute
    synoptic = hours.isin([0, 6, 12, 18]) & (minutes == 0)
    rep["synoptic"] = {
        "hour_distribution": hours.value_counts().sort_index().to_dict(),
        "n_synoptic_6h": int(synoptic.sum()),
        "pct_synoptic_6h": pct(int(synoptic.sum()), len(df)),
        "n_nonzero_minute": int((minutes != 0).sum()),
        "pct_nonzero_minute": pct(int((minutes != 0).sum()), len(df)),
    }

    # ---- observed vs interpolated ---------------------------------------
    obs: dict = {}
    if "TRACK_TYPE" in df.columns:
        obs["track_type_counts"] = df["TRACK_TYPE"].value_counts(dropna=False).to_dict()
    if "IFLAG" in df.columns:
        # IFLAG char 1 = USA agency; char 2 = Tokyo; ... (15 datasets)
        f = df["IFLAG"].astype("string")
        obs["iflag_len_sample"] = int(f.dropna().str.len().mode().iloc[0]) if f.notna().any() else None
        obs["iflag_char1_usa_counts"] = f.str[0].value_counts(dropna=False).to_dict()
        obs["iflag_full_value_top10"] = f.value_counts(dropna=False).head(10).to_dict()
        # 'O' = original report, 'P'/'I'/'V' = some interpolation, '_' = missing
        usa_original = (f.str[0] == "O")
        obs["n_usa_original"] = int(usa_original.sum())
        obs["pct_usa_original"] = pct(int(usa_original.sum()), len(df))
        obs["n_usa_original_and_synoptic"] = int((usa_original & synoptic).sum())
        obs["pct_usa_original_and_synoptic"] = pct(int((usa_original & synoptic).sum()), len(df))
    rep["observed_vs_interpolated"] = obs

    # ---- duplicate keys --------------------------------------------------
    dup_mask = df.duplicated(subset=["SID", "ISO_TIME"], keep=False)
    rep["duplicates"] = {
        "n_duplicate_sid_time_rows": int(dup_mask.sum()),
        "pct_duplicate_sid_time_rows": pct(int(dup_mask.sum()), len(df)),
        "example_sids": df.loc[dup_mask, "SID"].unique()[:5].tolist(),
    }

    # ---- agency wind conventions ----------------------------------------
    conv: dict = {}
    for col, (agency, avg) in AGENCY_WIND.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            conv[col] = {
                "agency": agency,
                "averaging_period": avg,
                "units": "knots",
                "n_present": int(s.notna().sum()),
                "pct_present": pct(int(s.notna().sum()), len(df)),
                "min": None if s.notna().sum() == 0 else float(s.min()),
                "max": None if s.notna().sum() == 0 else float(s.max()),
            }
    rep["agency_wind_conventions"] = conv
    rep["n_distinct_averaging_periods_present"] = len(
        {v["averaging_period"] for v in conv.values()
         if v["n_present"] > 0 and v["averaging_period"] != "MIXED - unsafe"}
    )
    if "WMO_AGENCY" in df.columns:
        rep["wmo_agency_counts"] = df["WMO_AGENCY"].value_counts(dropna=False).head(10).to_dict()

    # ---- physical range violations --------------------------------------
    viol = {}
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = int(((s < lo) | (s > hi)).sum())
            viol[col] = {"range": [lo, hi], "n_out_of_range": bad,
                         "observed_min": None if s.notna().sum() == 0 else float(s.min()),
                         "observed_max": None if s.notna().sum() == 0 else float(s.max())}
    rep["physical_range_check"] = viol

    # ---- longitude convention -------------------------------------------
    lon = pd.to_numeric(df["LON"], errors="coerce")
    rep["longitude_convention"] = {
        "min": float(lon.min()), "max": float(lon.max()),
        "convention": "-180..180" if lon.min() < 0 else "0..360",
        "n_beyond_180": int((lon > 180).sum()),
    }

    # ---- storm duration distribution (synoptic steps) --------------------
    syn = df[synoptic].copy()
    per_storm = syn.groupby("SID")["ISO_TIME"].agg(["count", "min", "max"])
    steps = per_storm["count"]
    rep["storm_duration_synoptic_steps"] = {
        "n_storms_with_synoptic": int(len(per_storm)),
        "mean": round(float(steps.mean()), 2),
        "median": float(steps.median()),
        "percentiles": {p: float(np.percentile(steps, int(p))) for p in ("10", "25", "50", "75", "90")},
        "n_storms_ge_12_steps_L8_H4": int((steps >= 12).sum()),
        "pct_storms_ge_12_steps": pct(int((steps >= 12).sum()), len(per_storm)),
        "n_storms_ge_8_steps_L4_H4": int((steps >= 8).sum()),
    }

    # ---- NATURE / category distribution ---------------------------------
    if "NATURE" in df.columns:
        rep["nature_counts"] = df["NATURE"].value_counts(dropna=False).to_dict()
    if "USA_SSHS" in df.columns:
        rep["usa_sshs_counts"] = (
            pd.to_numeric(df["USA_SSHS"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
        )
    return rep


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if np.isscalar(obj) and not isinstance(obj, str) else False:
        return None
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rep = jsonable(verify(args.csv))

    out = args.out or (REPORT_DIR / f"ibtracs_verification_{args.csv.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # concise stdout summary
    print(f"FILE            : {rep['file']}  ({rep['file_bytes']:,} bytes)")
    print(f"SHAPE           : {rep['n_rows']:,} rows x {rep['n_columns']} columns")
    print(f"STORMS          : {rep['n_storms']:,}   seasons {rep['season_range']}")
    print(f"TIME RANGE      : {rep['time_range']['min']} -> {rep['time_range']['max']}")
    print(f"BASINS          : {rep['basins']}")
    print(f"LON CONVENTION  : {rep['longitude_convention']['convention']} "
          f"(min {rep['longitude_convention']['min']}, max {rep['longitude_convention']['max']})")
    print(f"SYNOPTIC 6h     : {rep['synoptic']['n_synoptic_6h']:,} rows "
          f"({rep['synoptic']['pct_synoptic_6h']}%)")
    print(f"DUP (SID,TIME)  : {rep['duplicates']['n_duplicate_sid_time_rows']:,} rows")
    print(f"AVG PERIODS     : {rep['n_distinct_averaging_periods_present']} distinct present")
    print(f"REPORT          : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
