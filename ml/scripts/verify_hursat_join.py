"""HURSAT-B1 frame inventory + HURSAT<->IBTrACS temporal/spatial join test.

Phase 1, Tasks 5, 6, 7, 10, 11.

For each extracted HURSAT storm directory this script:
  1. Opens every NetCDF frame and records identity, time, satellite, VZA,
     embedded storm state, structural diagnostics, and IRWIN image validity.
  2. Joins each frame to IBTrACS on SID + nearest ISO_TIME.
  3. Measures the observed |dt| distribution (tests the +/-90 min hypothesis).
  4. Measures great-circle separation between the frame centre and best track.
  5. Quantifies duplicate frames per (SID, timestamp) and the VZA dedup rule.
  6. Reports synoptic (00/06/12/18Z) coverage.
  7. Reports NATURE composition -> negative-class feasibility.

Read-only. Writes a JSON report only.

Usage:
    python ml/scripts/verify_hursat_join.py --sample-dir <dir> [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]
FILL = -1.0  # HURSAT global attr: "Missing Values are -1.0"

# <SID>.<NAME>.<YYYY>.<MM>.<DD>.<HHMM>.<vza>.<SAT>.<sss>.hursat-b1.v06.nc
FRAME_RE = re.compile(
    r"^(?P<sid>\d{7}[NS]\d{5})\.(?P<name>[^.]+)\.(?P<Y>\d{4})\.(?P<M>\d{2})\.(?P<D>\d{2})"
    r"\.(?P<hhmm>\d{4})\.(?P<f1>\d+)\.(?P<sat>[^.]+)\.(?P<f2>\d+)\.hursat-b1\.v06\.nc$"
)

R_EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def scalar(ds, name):
    if name not in ds.variables:
        return None
    v = ds[name].values
    v = v.item() if getattr(v, "size", 0) == 1 else (v.ravel()[0] if getattr(v, "size", 0) else None)
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip()
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f) or f == FILL:
        return None
    return f


def inventory_frames(sample_dir: Path) -> pd.DataFrame:
    rows = []
    for nc in sorted(sample_dir.rglob("*.hursat-b1.v06.nc")):
        m = FRAME_RE.match(nc.name)
        rec: dict = {"file": nc.name, "bytes": nc.stat().st_size,
                     "fname_parsed": bool(m)}
        if m:
            rec.update({"sid_fname": m.group("sid"), "name_fname": m.group("name"),
                        "sat_fname": m.group("sat"),
                        "field1": int(m.group("f1")), "field2": int(m.group("f2")),
                        "t_fname": pd.Timestamp(
                            f"{m.group('Y')}-{m.group('M')}-{m.group('D')} "
                            f"{m.group('hhmm')[:2]}:{m.group('hhmm')[2:]}")})
        try:
            with xr.open_dataset(nc, decode_timedelta=False) as ds:
                rec["sid_attr"] = ds.attrs.get("TC_serial_number")
                rec["name_attr"] = ds.attrs.get("TC_name")
                rec["ibtracs_version_attr"] = ds.attrs.get("IBTrACS_Version")
                rec["satellite_attr"] = ds.attrs.get("Satellite_Name")
                rec["projection"] = ds.attrs.get("Projection")
                rec["t_cov_start"] = ds.attrs.get("time_coverage_start")
                rec["t_cov_end"] = ds.attrs.get("time_coverage_end")
                sid_var = scalar(ds, "sid")
                rec["sid_var"] = sid_var if isinstance(sid_var, str) else None
                rec["htime"] = pd.Timestamp(ds["htime"].values[0]).round("s")
                rec["nlat"], rec["nlon"] = ds.sizes.get("lat"), ds.sizes.get("lon")
                rec["channels"] = ",".join(
                    c for c in ("IRWIN", "IRWVP", "IRNIR", "IRSPL", "VSCHN", "VSVAR", "IRVAR")
                    if c in ds.variables)
                for v in ("VZA", "WindSpd", "CentPrs", "CentLat", "CentLon",
                          "odt84", "eye_prob", "eye_comp", "rad_eye", "bt_eye",
                          "archer_lat", "archer_lon", "var_icen", "SubSatLat", "SubSatLon"):
                    rec[v] = scalar(ds, v)
                if "IRWIN" in ds.variables:
                    a = ds["IRWIN"].values.astype("float32").ravel()
                    valid = np.isfinite(a) & (a != FILL)
                    rec["irwin_valid_pct"] = round(100.0 * valid.sum() / a.size, 3)
                    if valid.any():
                        vv = a[valid]
                        rec["irwin_min"] = float(vv.min())
                        rec["irwin_max"] = float(vv.max())
                        rec["irwin_mean"] = float(vv.mean())
                        rec["irwin_constant"] = bool(vv.min() == vv.max())
                    else:
                        rec["irwin_constant"] = True
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
    return pd.DataFrame(rows)


def load_ibtracs_for(sids: set[str]) -> pd.DataFrame:
    frames = []
    for p in sorted(zone("raw", "ibtracs").glob("ibtracs.*.list.v04r01.csv")):
        df = pd.read_csv(p, skiprows=[1], na_values=NA_VALUES, keep_default_na=False,
                         low_memory=False,
                         usecols=["SID", "ISO_TIME", "LAT", "LON", "NATURE", "TRACK_TYPE",
                                  "IFLAG", "USA_WIND", "USA_PRES", "USA_SSHS", "NAME",
                                  "BASIN", "SEASON"])
        frames.append(df[df["SID"].isin(sids)])
    ib = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["SID", "ISO_TIME"])
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    for c in ("LAT", "LON", "USA_WIND", "USA_PRES", "USA_SSHS"):
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    return ib.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path,
                    default=zone("samples", "hursat"))
    ap.add_argument("--tolerance-min", type=int, default=90)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "hursat_join_verification.json")
    args = ap.parse_args()

    inv = inventory_frames(args.sample_dir)
    if inv.empty:
        print(f"No HURSAT NetCDF files under {args.sample_dir}", file=sys.stderr)
        return 1

    rep: dict = {"sample_dir": str(args.sample_dir), "tolerance_minutes": args.tolerance_min}

    # ---- 1. inventory ----------------------------------------------------
    ok = inv[inv.get("error").isna()] if "error" in inv else inv
    rep["inventory"] = {
        "n_frames": int(len(inv)),
        "n_open_errors": int(inv["error"].notna().sum()) if "error" in inv else 0,
        "n_filenames_parsed": int(inv["fname_parsed"].sum()),
        "n_storms": int(ok["sid_attr"].nunique()),
        "storms": sorted(ok["sid_attr"].dropna().unique().tolist()),
        "grid_sizes": ok.groupby(["nlat", "nlon"]).size().to_dict().__str__(),
        "projections": ok["projection"].value_counts().to_dict(),
        "ibtracs_version_in_files": ok["ibtracs_version_attr"].value_counts().to_dict(),
        "satellites": ok["satellite_attr"].value_counts().to_dict(),
        "channel_sets": ok["channels"].value_counts().to_dict(),
        "total_uncompressed_mb": round(inv["bytes"].sum() / 1e6, 2),
        "mean_frame_mb": round(inv["bytes"].mean() / 1e6, 3),
    }

    # ---- 2. identity agreement ------------------------------------------
    rep["identity"] = {
        "sid_filename_eq_attr": int((ok["sid_fname"] == ok["sid_attr"]).sum()),
        "sid_var_eq_attr": int((ok["sid_var"] == ok["sid_attr"]).sum()),
        "n_checked": int(len(ok)),
        "filename_field1_equals_VZA_rounded": int(
            (ok["field1"] == ok["VZA"].round()).sum()),
        "note": "field1 in the filename is tested against round(VZA)",
    }

    # ---- 3. duplicate frames per (sid, htime) ---------------------------
    grp = ok.groupby(["sid_attr", "htime"])
    sizes = grp.size()
    dup = sizes[sizes > 1]
    rep["duplicate_frames"] = {
        "n_unique_sid_time": int(len(sizes)),
        "n_sid_time_with_multiple_frames": int(len(dup)),
        "pct_sid_time_duplicated": round(100 * len(dup) / len(sizes), 2),
        "max_frames_at_one_time": int(sizes.max()),
        "vza_available_pct": round(100 * ok["VZA"].notna().mean(), 2),
        "dedup_rule": "keep min(VZA) per (sid, htime); VZA present on all frames tested",
    }
    # simulate dedup
    ded = ok.sort_values("VZA").groupby(["sid_attr", "htime"], as_index=False).first()
    rep["duplicate_frames"]["n_frames_after_dedup"] = int(len(ded))
    rep["duplicate_frames"]["n_frames_before_dedup"] = int(len(ok))

    # ---- 4. temporal join to IBTrACS ------------------------------------
    ib = load_ibtracs_for(set(ok["sid_attr"].dropna()))
    joined_rows = []
    for sid, g in ded.groupby("sid_attr"):
        track = ib[ib["SID"] == sid]
        if track.empty:
            for _, fr in g.iterrows():
                joined_rows.append({"sid": sid, "htime": fr["htime"], "matched": False,
                                    "reason": "SID absent from IBTrACS v04r01"})
            continue
        left = g[["sid_attr", "htime", "CentLat", "CentLon", "VZA", "satellite_attr"]] \
            .rename(columns={"sid_attr": "SID"}).sort_values("htime")
        merged = pd.merge_asof(
            left, track.sort_values("ISO_TIME"),
            left_on="htime", right_on="ISO_TIME", by="SID",
            direction="nearest", tolerance=pd.Timedelta(minutes=args.tolerance_min),
        )
        for _, r in merged.iterrows():
            matched = pd.notna(r["ISO_TIME"])
            row = {"sid": sid, "htime": r["htime"], "matched": bool(matched)}
            if matched:
                row["dt_min"] = abs((r["htime"] - r["ISO_TIME"]).total_seconds()) / 60.0
                row["iso_time"] = r["ISO_TIME"]
                row["nature"] = r["NATURE"]
                row["track_type"] = r["TRACK_TYPE"]
                row["iflag0"] = str(r["IFLAG"])[0] if pd.notna(r["IFLAG"]) else None
                row["usa_wind"] = r["USA_WIND"]
                row["usa_sshs"] = r["USA_SSHS"]
                if pd.notna(r["CentLat"]) and pd.notna(r["LAT"]):
                    row["sep_km"] = float(haversine_km(r["CentLat"], r["CentLon"],
                                                       r["LAT"], r["LON"]))
            else:
                row["reason"] = f"no IBTrACS row within {args.tolerance_min} min"
            joined_rows.append(row)

    j = pd.DataFrame(joined_rows)
    n_att, n_ok = len(j), int(j["matched"].sum())
    dtv = j.loc[j["matched"], "dt_min"].dropna()
    sep = j.loc[j["matched"], "sep_km"].dropna() if "sep_km" in j else pd.Series(dtype=float)

    rep["temporal_join"] = {
        "n_attempted": n_att,
        "n_successful": n_ok,
        "n_failed": n_att - n_ok,
        "pct_matched": round(100 * n_ok / n_att, 2) if n_att else None,
        "failure_reasons": j.loc[~j["matched"], "reason"].value_counts().to_dict()
        if "reason" in j else {},
        "dt_minutes": {
            "min": float(dtv.min()), "max": float(dtv.max()),
            "mean": round(float(dtv.mean()), 2), "median": float(dtv.median()),
            "p95": float(np.percentile(dtv, 95)), "p99": float(np.percentile(dtv, 99)),
            "n_exact_zero": int((dtv == 0).sum()),
            "pct_exact_zero": round(100 * (dtv == 0).mean(), 2),
            "pct_le_30": round(100 * (dtv <= 30).mean(), 2),
            "pct_le_90": round(100 * (dtv <= 90).mean(), 2),
            "histogram": dtv.round().value_counts().sort_index().head(12).to_dict(),
        } if len(dtv) else None,
    }
    rep["spatial_agreement"] = {
        "n": int(len(sep)),
        "mean_km": round(float(sep.mean()), 2), "median_km": round(float(sep.median()), 2),
        "p95_km": round(float(np.percentile(sep, 95)), 2), "max_km": round(float(sep.max()), 2),
        "pct_under_50km": round(100 * (sep < 50).mean(), 2),
        "pct_under_25km": round(100 * (sep < 25).mean(), 2),
    } if len(sep) else None

    # ---- 5. synoptic coverage -------------------------------------------
    if len(dtv):
        mt = j[j["matched"]].copy()
        mt["iso_hour"] = pd.to_datetime(mt["iso_time"]).dt.hour
        syn = mt["iso_hour"].isin([0, 6, 12, 18])
        rep["synoptic_alignment"] = {
            "matched_frames": int(len(mt)),
            "matched_at_synoptic_hours": int(syn.sum()),
            "pct_at_synoptic": round(100 * syn.mean(), 2),
            "iso_hour_distribution": mt["iso_hour"].value_counts().sort_index().to_dict(),
            "iflag_usa_char_distribution": mt["iflag0"].value_counts(dropna=False).to_dict(),
            "n_synoptic_and_usa_original": int((syn & (mt["iflag0"] == "O")).sum()),
            "pct_synoptic_and_usa_original": round(100 * (syn & (mt["iflag0"] == "O")).mean(), 2),
            "track_type": mt["track_type"].value_counts(dropna=False).to_dict(),
        }
        # ---- 6. negative-class feasibility ------------------------------
        rep["negative_class"] = {
            "nature_distribution_of_frames": mt["nature"].value_counts(dropna=False).to_dict(),
            "n_non_tropical_frames": int((mt["nature"] != "TS").sum()),
            "pct_non_tropical_frames": round(100 * (mt["nature"] != "TS").mean(), 2),
            "usa_sshs_distribution": mt["usa_sshs"].value_counts(dropna=False).sort_index().to_dict(),
            "note": "NATURE != 'TS' are Path-B negative candidates (still contain a vortex)",
        }

    # ---- 7. image quality / conversion feasibility ----------------------
    rep["image_quality"] = {
        "irwin_present_frames": int(ok["irwin_valid_pct"].notna().sum()),
        "irwin_valid_pct": {
            "mean": round(float(ok["irwin_valid_pct"].mean()), 2),
            "min": round(float(ok["irwin_valid_pct"].min()), 2),
            "pct_frames_over_99_valid": round(
                100 * (ok["irwin_valid_pct"] > 99).mean(), 2),
        },
        "n_constant_or_empty_frames": int(ok["irwin_constant"].sum()),
        "irwin_kelvin_range": {
            "min": round(float(ok["irwin_min"].min()), 2),
            "max": round(float(ok["irwin_max"].max()), 2),
            "mean_of_frame_means": round(float(ok["irwin_mean"].mean()), 2),
        },
    }
    rep["structural_fields_present"] = {
        c: {"pct_present": round(100 * ok[c].notna().mean(), 2)}
        for c in ("odt84", "eye_prob", "eye_comp", "rad_eye", "bt_eye",
                  "var_icen", "archer_lat", "WindSpd", "CentPrs", "VZA")
        if c in ok
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    # ---- stdout ----------------------------------------------------------
    i, t = rep["inventory"], rep["temporal_join"]
    print(f"FRAMES          : {i['n_frames']} across {i['n_storms']} storms "
          f"({i['total_uncompressed_mb']} MB, mean {i['mean_frame_mb']} MB/frame)")
    print(f"OPEN ERRORS     : {i['n_open_errors']}")
    print(f"GRID            : {i['grid_sizes']}   proj={i['projections']}")
    print(f"CHANNELS        : {i['channel_sets']}")
    print(f"SATELLITES      : {i['satellites']}")
    print(f"IDENTITY        : filename SID == attr SID for "
          f"{rep['identity']['sid_filename_eq_attr']}/{rep['identity']['n_checked']}; "
          f"var==attr {rep['identity']['sid_var_eq_attr']}/{rep['identity']['n_checked']}")
    d = rep["duplicate_frames"]
    print(f"DUPLICATES      : {d['n_sid_time_with_multiple_frames']}/{d['n_unique_sid_time']} "
          f"timestamps have >1 frame (max {d['max_frames_at_one_time']}); "
          f"{d['n_frames_before_dedup']} -> {d['n_frames_after_dedup']} after VZA dedup")
    print(f"JOIN            : {t['n_successful']}/{t['n_attempted']} matched "
          f"({t['pct_matched']}%)  failures={t['failure_reasons']}")
    if t["dt_minutes"]:
        m = t["dt_minutes"]
        print(f"  |dt| minutes  : median={m['median']} mean={m['mean']} max={m['max']} "
              f"p95={m['p95']} | exact0={m['pct_exact_zero']}% <=30m={m['pct_le_30']}%")
    if rep.get("spatial_agreement"):
        s = rep["spatial_agreement"]
        print(f"  separation km : median={s['median_km']} mean={s['mean_km']} "
              f"max={s['max_km']} | <50km={s['pct_under_50km']}%")
    if rep.get("synoptic_alignment"):
        s = rep["synoptic_alignment"]
        print(f"SYNOPTIC        : {s['matched_at_synoptic_hours']}/{s['matched_frames']} "
              f"({s['pct_at_synoptic']}%) at 00/06/12/18Z; "
              f"USA-original AND synoptic = {s['pct_synoptic_and_usa_original']}%")
    q = rep["image_quality"]
    print(f"IRWIN           : valid mean={q['irwin_valid_pct']['mean']}% "
          f"min={q['irwin_valid_pct']['min']}%  constant/empty frames="
          f"{q['n_constant_or_empty_frames']}  K range="
          f"[{q['irwin_kelvin_range']['min']}, {q['irwin_kelvin_range']['max']}]")
    if rep.get("negative_class"):
        print(f"NATURE          : {rep['negative_class']['nature_distribution_of_frames']}")
    print(f"\nREPORT: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
