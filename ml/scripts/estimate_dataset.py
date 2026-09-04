"""Dataset size / usable-sample estimation (Phase 1, Tasks 7 and 12).

Combines verified IBTrACS structure with measured HURSAT sample statistics to
project how many usable samples and how much storage the MVP subset requires.

EVERY NUMBER PRODUCED BY THIS SCRIPT IS AN ESTIMATE. Inputs measured on a
3-storm / 195-frame HURSAT sample and a 7-season archive listing are
extrapolated; they are not a full-archive census.

Usage:
    python ml/scripts/estimate_dataset.py --basin NA --from 1995 --to 2015
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402

NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]

# ---- MEASURED inputs from Phase 1 verification (see reports/) ------------
MEASURED = {
    "hursat_storm_coverage_frac": 0.960,      # crosswalk_verification.json, NA seasons
    "frames_per_synoptic_time_before_dedup": 195 / 109,
    "hursat_frames_at_synoptic_frac": 0.5138,  # hursat_join_verification.json
    "mean_raw_frame_mb_all_channels": 0.409,
    "compression_ratio_targz": 43.0 / 24.4,
    "mean_archive_mb_per_storm_NA": 26.3,
    "adt_bytes_per_storm": 42_000,
    "adt_join_frac_at_90min": 1.00,
}
IMG = 224 * 224  # bytes per uint8 image


def load(basin: str, y0: int, y1: int) -> pd.DataFrame:
    out = []
    for p in sorted(zone("raw", "ibtracs").glob("ibtracs.*.list.v04r01.csv")):
        df = pd.read_csv(p, skiprows=[1], na_values=NA_VALUES, keep_default_na=False,
                         low_memory=False,
                         usecols=["SID", "SEASON", "BASIN", "ISO_TIME", "LAT", "LON",
                                  "USA_WIND", "USA_PRES", "USA_SSHS", "NATURE",
                                  "TRACK_TYPE", "IFLAG"])
        out.append(df)
    ib = pd.concat(out, ignore_index=True).drop_duplicates(["SID", "ISO_TIME"])
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    first = ib.sort_values("ISO_TIME").groupby("SID")["BASIN"].first()
    ib = ib[(ib["SID"].map(first) == basin) & ib["SEASON"].between(y0, y1)].copy()
    for c in ("LAT", "LON", "USA_WIND", "USA_PRES", "USA_SSHS"):
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    return ib.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def count_windows(g: pd.DataFrame, L: int, H: int) -> int:
    """Count sliding windows needing L+H contiguous 6-hourly steps."""
    t = g["ISO_TIME"].sort_values().to_numpy()
    if len(t) < L + H:
        return 0
    gaps = np.diff(t).astype("timedelta64[m]").astype(int)
    runs, cur = [], 1
    for gap in gaps:
        if gap == 360:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return int(sum(max(0, r - (L + H) + 1) for r in runs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="NA")
    ap.add_argument("--from", dest="y0", type=int, default=1995)
    ap.add_argument("--to", dest="y1", type=int, default=2015)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "dataset_estimates.json")
    args = ap.parse_args()

    ib = load(args.basin, args.y0, args.y1)
    rep: dict = {
        "DISCLAIMER": "ALL VALUES ARE ESTIMATES extrapolated from a small sample.",
        "basin": args.basin, "seasons": [args.y0, args.y1],
        "window": {"L": args.L, "H": args.H,
                   "required_contiguous_steps": args.L + args.H},
        "measured_inputs": MEASURED,
    }

    # ---- Stage 1: raw IBTrACS -------------------------------------------
    rep["s1_ibtracs_raw"] = {
        "rows": int(len(ib)), "storms": int(ib["SID"].nunique()),
        "seasons": int(ib["SEASON"].nunique()),
    }

    # ---- Stage 2: synoptic filter ---------------------------------------
    t = ib["ISO_TIME"]
    syn = ib[t.dt.hour.isin([0, 6, 12, 18]) & (t.dt.minute == 0)].copy()
    rep["s2_synoptic_6h"] = {"rows": int(len(syn)), "storms": int(syn["SID"].nunique()),
                             "pct_of_raw": round(100 * len(syn) / len(ib), 2)}

    # ---- Stage 3: observed-only (IFLAG[0] == 'O') + main track ----------
    obs = syn[(syn["IFLAG"].astype(str).str[0] == "O")
              & (syn["TRACK_TYPE"] == "main")].copy()
    rep["s3_observed_main"] = {
        "rows": int(len(obs)), "storms": int(obs["SID"].nunique()),
        "pct_of_synoptic": round(100 * len(obs) / len(syn), 2),
        "rule": "IFLAG char1 == 'O' (USA original report) AND TRACK_TYPE == 'main'",
    }

    # ---- Stage 4: usable wind label (single agency, no fallback) --------
    lab = obs[obs["USA_WIND"].notna()].copy()
    rep["s4_with_usa_wind"] = {
        "rows": int(len(lab)), "storms": int(lab["SID"].nunique()),
        "pct_of_observed": round(100 * len(lab) / len(obs), 2),
        "also_with_usa_pres": int(lab["USA_PRES"].notna().sum()),
        "pct_with_pressure": round(100 * lab["USA_PRES"].notna().mean(), 2),
    }

    # ---- Stage 5: forecasting windows (IBTrACS-only path, P2) -----------
    win = lab.groupby("SID", group_keys=False).apply(
        lambda g: count_windows(g, args.L, args.H), include_groups=False)
    rep["s5_sequence_windows_ibtracs_only"] = {
        "total_windows": int(win.sum()),
        "storms_contributing": int((win > 0).sum()),
        "storms_yielding_zero": int((win == 0).sum()),
        "median_windows_per_contributing_storm": float(win[win > 0].median()) if (win > 0).any() else 0,
    }
    for L in (4, 8, 12):
        w = lab.groupby("SID", group_keys=False).apply(
            lambda g, L=L: count_windows(g, L, args.H), include_groups=False)
        rep["s5_sequence_windows_ibtracs_only"][f"windows_if_L{L}_H{args.H}"] = int(w.sum())

    # ---- Stage 6: fused (satellite) samples ------------------------------
    cov = MEASURED["hursat_storm_coverage_frac"]
    fused_rows = len(lab) * cov
    rep["s6_fused_with_imagery"] = {
        "estimated_storms_with_imagery": int(round(lab["SID"].nunique() * cov)),
        "estimated_fused_frames_after_dedup": int(round(fused_rows)),
        "assumption": ("every observed synoptic best-track row of a covered storm has a "
                       "deduplicated HURSAT frame; measured join rate on the sample was 100%"),
        "estimated_frames_before_dedup": int(round(
            fused_rows * MEASURED["frames_per_synoptic_time_before_dedup"])),
    }
    rep["s7_windows_with_imagery"] = {
        "estimated_windows": int(round(rep["s5_sequence_windows_ibtracs_only"]["total_windows"] * cov)),
    }

    # ---- Stage 7: split -------------------------------------------------
    seasons = sorted(lab["SEASON"].unique())
    n = len(seasons)
    tr, va = seasons[: int(n * 0.7)], seasons[int(n * 0.7): int(n * 0.85)]
    te = seasons[int(n * 0.85):]
    split = {}
    for nm, ys in (("train", tr), ("val", va), ("test", te)):
        s = lab[lab["SEASON"].isin(ys)]
        w = s.groupby("SID", group_keys=False).apply(
            lambda g: count_windows(g, args.L, args.H), include_groups=False)
        split[nm] = {"seasons": [int(y) for y in ys], "storms": int(s["SID"].nunique()),
                     "frames": int(len(s)), "windows": int(w.sum())}
    rep["s8_temporal_split_by_season"] = split

    # ---- Storage ---------------------------------------------------------
    n_frames = rep["s6_fused_with_imagery"]["estimated_fused_frames_after_dedup"]
    n_all_frames_3h = n_frames / MEASURED["hursat_frames_at_synoptic_frac"]
    raw_dl_gb = (lab["SID"].nunique() * cov * MEASURED["mean_archive_mb_per_storm_NA"]) / 1000
    rep["storage_estimates_gb"] = {
        "raw_hursat_targz_download": round(raw_dl_gb, 1),
        "raw_hursat_extracted": round(raw_dl_gb * MEASURED["compression_ratio_targz"], 1),
        "zarr_irwin_uint8_224_synoptic_only": round(n_frames * IMG / 1e9, 3),
        "zarr_irwin_uint8_224_all_3hourly": round(n_all_frames_3h * IMG / 1e9, 3),
        "zarr_if_float32_instead": round(n_frames * IMG * 4 / 1e9, 3),
        "ibtracs_csv_raw": round(sum(p.stat().st_size for p in
                                     zone("raw", "ibtracs").glob("*.csv")) / 1e9, 3),
        "adt_hursat_all_storms": round(lab["SID"].nunique()
                                       * MEASURED["adt_bytes_per_storm"] / 1e9, 4),
        "thumbnails_png_256": round(n_frames * 25_000 / 1e9, 3),
    }
    rep["storage_estimates_gb"]["TOTAL_working_set"] = round(
        rep["storage_estimates_gb"]["raw_hursat_extracted"]
        + rep["storage_estimates_gb"]["zarr_irwin_uint8_224_synoptic_only"]
        + rep["storage_estimates_gb"]["ibtracs_csv_raw"]
        + rep["storage_estimates_gb"]["thumbnails_png_256"], 1)
    rep["storage_estimates_gb"]["TOTAL_if_raw_deleted_after_conversion"] = round(
        rep["storage_estimates_gb"]["zarr_irwin_uint8_224_synoptic_only"]
        + rep["storage_estimates_gb"]["thumbnails_png_256"]
        + rep["storage_estimates_gb"]["ibtracs_csv_raw"], 2)

    # ---- label distributions --------------------------------------------
    rep["label_distributions"] = {
        "usa_sshs_on_usable_rows": lab["USA_SSHS"].value_counts().sort_index().to_dict(),
        "nature_on_usable_rows": lab["NATURE"].value_counts().to_dict(),
        "n_non_tropical_negatives_pathB": int((lab["NATURE"] != "TS").sum()),
        "pct_non_tropical_negatives_pathB": round(100 * (lab["NATURE"] != "TS").mean(), 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print(f"*** ALL FIGURES ARE ESTIMATES ***  basin={args.basin} "
          f"seasons={args.y0}-{args.y1}  L={args.L} H={args.H}\n")
    print("FUNNEL (measured on IBTrACS, exact):")
    print(f"  1 raw rows                 {rep['s1_ibtracs_raw']['rows']:>8,}   "
          f"storms {rep['s1_ibtracs_raw']['storms']:>4}")
    print(f"  2 synoptic 6-hourly        {rep['s2_synoptic_6h']['rows']:>8,}   "
          f"storms {rep['s2_synoptic_6h']['storms']:>4}   ({rep['s2_synoptic_6h']['pct_of_raw']}% of raw)")
    print(f"  3 observed + main track    {rep['s3_observed_main']['rows']:>8,}   "
          f"storms {rep['s3_observed_main']['storms']:>4}   ({rep['s3_observed_main']['pct_of_synoptic']}% of synoptic)")
    print(f"  4 with USA_WIND label      {rep['s4_with_usa_wind']['rows']:>8,}   "
          f"storms {rep['s4_with_usa_wind']['storms']:>4}   "
          f"(pressure on {rep['s4_with_usa_wind']['pct_with_pressure']}%)")
    s5 = rep["s5_sequence_windows_ibtracs_only"]
    print(f"  5 sequence windows L8+H4   {s5['total_windows']:>8,}   from "
          f"{s5['storms_contributing']} storms  (L4:{s5['windows_if_L4_H4']:,} "
          f"L12:{s5['windows_if_L12_H4']:,})")
    print("\nESTIMATED (extrapolated from sample):")
    print(f"  6 fused frames w/ imagery  {rep['s6_fused_with_imagery']['estimated_fused_frames_after_dedup']:>8,}   "
          f"({rep['s6_fused_with_imagery']['estimated_frames_before_dedup']:,} before dedup)")
    print(f"  7 windows w/ imagery       {rep['s7_windows_with_imagery']['estimated_windows']:>8,}")
    print("\nSPLIT (by season):")
    for nm, d in split.items():
        print(f"  {nm:<6} seasons {d['seasons'][0]}-{d['seasons'][-1]}  "
              f"storms {d['storms']:>4}  frames {d['frames']:>7,}  windows {d['windows']:>7,}")
    print("\nSTORAGE (GB, estimates):")
    for k, v in rep["storage_estimates_gb"].items():
        print(f"  {k:<42} {v:>8}")
    print(f"\nREPORT: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
