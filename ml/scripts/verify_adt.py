"""ADT-HURSAT verification (Phase 1, Task 9).

Answers blocking TO-VERIFY #16: does ADT-HURSAT expose a genuine Dvorak
scene-type / pattern classification field?

Downloads a small per-storm sample (files are ~35-50 KB each, one per IBTrACS
SID), then reports the scene vocabulary, class distribution, ADT<->IBTrACS
temporal offsets, and coverage.

Usage:
    python ml/scripts/verify_adt.py --season 2005 --basin NA --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402

ADT_BASE = ("https://www.ncei.noaa.gov/data/oceans/archive/arc0239/0307249/"
            "1.1/data/0-data")
NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]

# From the ADT-HURSAT variable descriptions (v9.0)
EYE_SCENE = {0: "Eye", 1: "Pinhole Eye", 2: "Large Eye", 3: "No Eye"}
CLOUD_SCENE = {0: "CDO", 1: "Embedded Center", 2: "Irregular CDO",
               3: "Curved Band", 4: "Shear"}

FIELDS = ("Date", "Time", "Scene", "EyeScene", "CloudScene", "CI", "FinalT",
          "RawT", "AdjT", "WindSpeed", "MSLP", "Lat", "Lon", "SatIDChar",
          "Land", "VZA", "EyeSize", "CDOSize", "ShearDist", "RMW", "CloudSym")


def parse_adt_time(date_s: str, time_s: str) -> pd.Timestamp:
    """ADT Date='2005AUG26', Time='174513' (HHMMSS)."""
    t = str(time_s).zfill(6)
    return pd.to_datetime(f"{date_s} {t[:2]}:{t[2:4]}:{t[4:6]}",
                          format="%Y%b%d %H:%M:%S", errors="coerce")


def load_ibtracs(basin: str, season: int) -> pd.DataFrame:
    out = []
    for p in sorted(zone("raw", "ibtracs").glob("ibtracs.*.list.v04r01.csv")):
        df = pd.read_csv(p, skiprows=[1], na_values=NA_VALUES, keep_default_na=False,
                         low_memory=False,
                         usecols=["SID", "SEASON", "BASIN", "NAME", "ISO_TIME",
                                  "LAT", "LON", "USA_WIND", "USA_SSHS", "NATURE"])
        out.append(df[(df["SEASON"] == season)])
    ib = pd.concat(out, ignore_index=True).drop_duplicates(["SID", "ISO_TIME"])
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    first = ib.sort_values("ISO_TIME").groupby("SID")["BASIN"].first()
    ib = ib[ib["SID"].map(first) == basin]
    for c in ("LAT", "LON", "USA_WIND", "USA_SSHS"):
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    return ib.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2005)
    ap.add_argument("--basin", default="NA")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "adt_verification.json")
    args = ap.parse_args()

    ib = load_ibtracs(args.basin, args.season)
    sids = sorted(ib["SID"].unique())[: args.limit]
    cache = zone("samples", "adt", create=True)

    frames, missing, errors = [], [], []
    for sid in sids:
        f = cache / f"{sid}.nc"
        if not f.exists():
            r = requests.get(f"{ADT_BASE}/{sid}.nc", timeout=120)
            if r.status_code == 404:
                missing.append(sid)
                continue
            r.raise_for_status()
            f.write_bytes(r.content)
        try:
            with xr.open_dataset(f) as ds:
                d = {k: ds[k].values for k in FIELDS if k in ds.variables}
                df = pd.DataFrame(d)
                df["sid"] = sid
                df["adt_version"] = ds.attrs.get("ADT_version")
                df["hursat_version"] = ds.attrs.get("HURSAT_version")
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            errors.append({"sid": sid, "error": f"{type(exc).__name__}: {exc}"})

    if not frames:
        print("No ADT files retrieved.", file=sys.stderr)
        return 1
    a = pd.concat(frames, ignore_index=True)
    a["adt_time"] = [parse_adt_time(d, t) for d, t in zip(a["Date"], a["Time"])]

    rep: dict = {
        "question": "TO-VERIFY #16: does ADT-HURSAT expose a Dvorak scene-type field?",
        "answer": "YES",
        "source_url": ADT_BASE,
        "season": args.season, "basin": args.basin,
        "storms_requested": len(sids),
        "storms_retrieved": int(a["sid"].nunique()),
        "storms_missing_from_adt": missing,
        "open_errors": errors,
        "adt_version": sorted(a["adt_version"].dropna().unique().tolist()),
        "hursat_version_used_by_adt": sorted(a["hursat_version"].dropna().unique().tolist()),
        "n_records": int(len(a)),
        "mean_records_per_storm": round(len(a) / a["sid"].nunique(), 1),
        "scene_field": {
            "name": "Scene",
            "dtype": "string",
            "derivation": "combination of EyeScene and CloudScene integer codes",
            "eye_scene_codes": EYE_SCENE,
            "cloud_scene_codes": CLOUD_SCENE,
        },
    }

    # ---- class distribution ---------------------------------------------
    vc = a["Scene"].value_counts()
    rep["scene_distribution"] = {
        "counts": vc.to_dict(),
        "pct": (100 * vc / len(a)).round(2).to_dict(),
        "n_classes": int(vc.size),
        "imbalance_ratio_max_over_min": round(float(vc.max() / vc.min()), 1),
        "classes_under_200_samples": vc[vc < 200].index.tolist(),
        "land_records": int((a["Scene"] == "Land").sum()),
        "pct_land_records": round(100 * (a["Scene"] == "Land").mean(), 2),
    }
    rep["scene_distribution_excluding_land"] = (
        100 * a.loc[a["Scene"] != "Land", "Scene"].value_counts(normalize=True)
    ).round(2).to_dict()
    rep["eye_scene_distribution"] = {
        EYE_SCENE.get(int(k), str(k)): int(v)
        for k, v in a["EyeScene"].value_counts().sort_index().items()}
    rep["cloud_scene_distribution"] = {
        CLOUD_SCENE.get(int(k), str(k)): int(v)
        for k, v in a["CloudScene"].value_counts().sort_index().items()}

    # ---- temporal structure ---------------------------------------------
    mins = a["adt_time"].dt.minute
    rep["adt_time_structure"] = {
        "minute_distribution_top": mins.value_counts().head(8).to_dict(),
        "pct_on_the_hour": round(100 * (mins == 0).mean(), 2),
        "hour_distribution": a["adt_time"].dt.hour.value_counts().sort_index().to_dict(),
        "note": "ADT reports true satellite scan times, not nominal synoptic slots",
    }

    # ---- join to IBTrACS -------------------------------------------------
    rows = []
    for sid, g in a.groupby("sid"):
        tr = ib[ib["SID"] == sid].sort_values("ISO_TIME")
        if tr.empty:
            continue
        left = g[["sid", "adt_time", "Scene", "CI", "WindSpeed", "Lat", "Lon"]] \
            .dropna(subset=["adt_time"]).sort_values("adt_time").rename(columns={"sid": "SID"})
        for tol in (15, 30, 90, 180):
            m = pd.merge_asof(left, tr, left_on="adt_time", right_on="ISO_TIME",
                              by="SID", direction="nearest",
                              tolerance=pd.Timedelta(minutes=tol))
            rows.append({"tol": tol, "n": len(m), "matched": int(m["ISO_TIME"].notna().sum())})
        m90 = pd.merge_asof(left, tr, left_on="adt_time", right_on="ISO_TIME", by="SID",
                            direction="nearest", tolerance=pd.Timedelta(minutes=90))
        ok = m90[m90["ISO_TIME"].notna()].copy()
        if len(ok):
            ok["dt_min"] = (ok["adt_time"] - ok["ISO_TIME"]).abs().dt.total_seconds() / 60
            rep.setdefault("_dt", []).extend(ok["dt_min"].tolist())
            rep.setdefault("_pairs", []).append(
                ok[["CI", "WindSpeed", "USA_WIND", "Scene", "USA_SSHS"]])

    tol_df = pd.DataFrame(rows).groupby("tol").sum()
    rep["join_to_ibtracs_by_tolerance"] = {
        f"+/-{int(t)}min": {"attempted": int(r["n"]), "matched": int(r["matched"]),
                            "pct": round(100 * r["matched"] / r["n"], 2)}
        for t, r in tol_df.iterrows()}

    dt = np.array(rep.pop("_dt", []))
    if dt.size:
        rep["adt_ibtracs_dt_minutes"] = {
            "n": int(dt.size), "min": float(dt.min()), "max": float(dt.max()),
            "mean": round(float(dt.mean()), 2), "median": float(np.median(dt)),
            "p95": round(float(np.percentile(dt, 95)), 2),
            "pct_exact_zero": round(100 * float((dt == 0).mean()), 2),
            "pct_le_90": round(100 * float((dt <= 90).mean()), 2),
        }
    pairs = rep.pop("_pairs", [])
    if pairs:
        p = pd.concat(pairs, ignore_index=True).dropna(subset=["WindSpeed", "USA_WIND"])
        if len(p):
            diff = p["WindSpeed"] - p["USA_WIND"]
            rep["adt_vs_besttrack_wind"] = {
                "n": int(len(p)),
                "mean_bias_kt": round(float(diff.mean()), 2),
                "mae_kt": round(float(diff.abs().mean()), 2),
                "rmse_kt": round(float(np.sqrt((diff ** 2).mean())), 2),
                "corr": round(float(p["WindSpeed"].corr(p["USA_WIND"])), 4),
                "note": "ADT is an algorithm estimate; best track remains ground truth",
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print(f"ADT version     : {rep['adt_version']}  (HURSAT {rep['hursat_version_used_by_adt']})")
    print(f"STORMS          : {rep['storms_retrieved']}/{rep['storms_requested']} retrieved"
          f"  missing={len(missing)}")
    print(f"RECORDS         : {rep['n_records']:,} "
          f"({rep['mean_records_per_storm']} per storm)")
    print(f"SCENE FIELD     : PRESENT -> 'Scene' ({rep['scene_distribution']['n_classes']} values)")
    print("SCENE DIST      :")
    for k, v in rep["scene_distribution"]["counts"].items():
        print(f"   {k:<14} {v:>6}  ({rep['scene_distribution']['pct'][k]:>5}%)")
    print(f"  imbalance max/min = {rep['scene_distribution']['imbalance_ratio_max_over_min']}x"
          f"  | Land records = {rep['scene_distribution']['pct_land_records']}%")
    print(f"ADT TIMES       : {rep['adt_time_structure']['pct_on_the_hour']}% on the hour "
          f"(top minutes {list(rep['adt_time_structure']['minute_distribution_top'])[:5]})")
    print("JOIN vs TOL     :", {k: v["pct"] for k, v in rep["join_to_ibtracs_by_tolerance"].items()})
    if "adt_ibtracs_dt_minutes" in rep:
        d = rep["adt_ibtracs_dt_minutes"]
        print(f"  |dt| minutes  : median={d['median']} mean={d['mean']} max={d['max']} "
              f"exact0={d['pct_exact_zero']}%")
    if "adt_vs_besttrack_wind" in rep:
        w = rep["adt_vs_besttrack_wind"]
        print(f"ADT vs BT wind  : n={w['n']} bias={w['mean_bias_kt']}kt MAE={w['mae_kt']}kt "
              f"r={w['corr']}")
    print(f"\nREPORT: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
