"""Preliminary data quality gate (Phase 1, Task 13).

Runs the DATA_STRATEGY.md 4.4 assertions plus the Phase 1 additions against the
verified sample, and emits a machine-readable JSON report.

This is a PRELIMINARY gate operating on the Phase 1 verification sample. It is
not the production pipeline; the production gate (Phase 4) will run the same
assertions over the full fused dataset.

Exit code 0 = all blocking assertions pass, 1 = at least one blocking failure.

Usage:
    python ml/scripts/qc_gate.py [--sample-dir DIR] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]
FRAME_RE = re.compile(
    r"^(?P<sid>\d{7}[NS]\d{5})\.(?P<name>[^.]+)\.(?P<Y>\d{4})\.(?P<M>\d{2})\.(?P<D>\d{2})"
    r"\.(?P<hhmm>\d{4})\.(?P<f1>\d+)\.(?P<sat>[^.]+)\.(?P<f2>\d+)\.hursat-b1\.v06\.nc$")
R_EARTH_KM = 6371.0088

# Physical ranges (units verified from IBTrACS v04r01 column documentation)
RANGES = {"LAT": (-90, 90), "LON": (-180, 180),
          "USA_WIND": (0, 300), "USA_PRES": (850, 1050)}
IRWIN_PHYSICAL_K = (150.0, 350.0)   # anything outside is masked, not trusted


@dataclass
class Check:
    id: str
    name: str
    blocking: bool
    threshold: str
    passed: bool | None = None
    value: object = None
    detail: dict = field(default_factory=dict)


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def load_tracks() -> tuple[pd.DataFrame, dict]:
    """Load and concatenate IBTrACS basin CSVs.

    IBTrACS basin files OVERLAP: a storm that crosses basins appears in full in
    every basin file it touches. Verified on v04r01 NA+NI: 0 duplicates within
    any single file, 4 storms duplicated across files, and the duplicated rows
    are byte-identical. Deduplicating on (SID, ISO_TIME) is therefore lossless.
    """
    out = []
    for p in sorted(zone("raw", "ibtracs").glob("ibtracs.*.list.v04r01.csv")):
        df = pd.read_csv(
            p, skiprows=[1], na_values=NA_VALUES, keep_default_na=False, low_memory=False,
            usecols=["SID", "SEASON", "BASIN", "ISO_TIME", "LAT", "LON", "NATURE",
                     "TRACK_TYPE", "IFLAG", "USA_WIND", "USA_PRES", "USA_SSHS"])
        out.append(df)
    raw = pd.concat(out, ignore_index=True)

    dup_mask = raw.duplicated(["SID", "ISO_TIME"], keep=False)
    dup_groups = raw[dup_mask].groupby(["SID", "ISO_TIME"])
    n_identical = int(sum(g.drop_duplicates().shape[0] == 1 for _, g in dup_groups))
    overlap = {
        "n_files": len(out),
        "rows_before_dedup": int(len(raw)),
        "cross_file_duplicate_rows": int(dup_mask.sum()),
        "duplicate_groups": int(dup_groups.ngroups),
        "duplicate_groups_with_identical_content": n_identical,
        "storms_affected": sorted(raw.loc[dup_mask, "SID"].unique().tolist())[:20],
    }

    df = raw.drop_duplicates(["SID", "ISO_TIME"]).reset_index(drop=True)
    overlap["rows_after_dedup"] = int(len(df))

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ("LAT", "LON", "USA_WIND", "USA_PRES", "USA_SSHS"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, overlap


def load_frames(sample_dir: Path) -> pd.DataFrame:
    rows = []
    for nc in sorted(sample_dir.rglob("*.hursat-b1.v06.nc")):
        m = FRAME_RE.match(nc.name)
        r = {"file": nc.name, "parsed": bool(m),
             "sid_fname": m.group("sid") if m else None}
        try:
            with xr.open_dataset(nc, decode_timedelta=False) as ds:
                r["sid"] = ds.attrs.get("TC_serial_number")
                r["htime"] = pd.Timestamp(ds["htime"].values[0]).round("s")
                r["sat"] = ds.attrs.get("Satellite_Name")
                r["nlat"], r["nlon"] = ds.sizes.get("lat"), ds.sizes.get("lon")
                for v in ("VZA", "CentLat", "CentLon"):
                    val = ds[v].values.ravel()[0] if v in ds.variables else np.nan
                    r[v] = float(val) if np.isfinite(val) else np.nan
                if "IRWIN" in ds.variables:
                    a = np.asarray(ds["IRWIN"].values, "float32").ravel()
                    phys = a[np.isfinite(a) & (a >= IRWIN_PHYSICAL_K[0])
                             & (a <= IRWIN_PHYSICAL_K[1])]
                    r["irwin_valid_pct"] = 100.0 * phys.size / a.size
                    r["irwin_constant"] = bool(phys.size == 0 or phys.min() == phys.max())
                    r["irwin_min"] = float(phys.min()) if phys.size else np.nan
                    r["irwin_max"] = float(phys.max()) if phys.size else np.nan
                else:
                    r["irwin_valid_pct"], r["irwin_constant"] = 0.0, True
        except Exception as exc:  # noqa: BLE001
            r["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=zone("samples", "hursat"))
    ap.add_argument("--tolerance-min", type=int, default=90)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "qc_gate_phase1.json")
    args = ap.parse_args()

    checks: list[Check] = []
    tr, overlap = load_tracks()
    fr = load_frames(args.sample_dir)
    fr_ok = fr[fr.get("error").isna()] if "error" in fr else fr

    sids = set(fr_ok["sid"].dropna())
    sub = tr[tr["SID"].isin(sids)].copy()

    # --- T1 duplicate (SID, timestamp) in tracks, after documented dedup --
    d = int(tr.duplicated(["SID", "ISO_TIME"]).sum())
    checks.append(Check("T1", "No duplicate (SID, ISO_TIME) in best track", True,
                        "== 0 after cross-file dedup", d == 0, d, overlap))

    # --- T1b cross-file overlap must be lossless -------------------------
    lossless = (overlap["duplicate_groups"]
                == overlap["duplicate_groups_with_identical_content"])
    checks.append(Check("T1b", "Cross-basin-file duplicate rows are byte-identical", True,
                        "all duplicate groups identical", lossless,
                        f"{overlap['duplicate_groups_with_identical_content']}"
                        f"/{overlap['duplicate_groups']}", overlap))

    # --- T2 missing timestamps -------------------------------------------
    n = int(tr["ISO_TIME"].isna().sum())
    checks.append(Check("T2", "No unparseable/missing ISO_TIME", True, "== 0", n == 0, n))

    # --- T3/T4 lat/lon validity ------------------------------------------
    for cid, col in (("T3", "LAT"), ("T4", "LON")):
        lo, hi = RANGES[col]
        bad = int(((tr[col] < lo) | (tr[col] > hi) | tr[col].isna()).sum())
        checks.append(Check(cid, f"{col} within {RANGES[col]} and non-null", True,
                            "== 0", bad == 0, bad,
                            {"observed_min": float(tr[col].min()),
                             "observed_max": float(tr[col].max())}))

    # --- T5/T6 intensity availability on usable rows ---------------------
    t = tr["ISO_TIME"]
    usable = tr[t.dt.hour.isin([0, 6, 12, 18]) & (t.dt.minute == 0)
                & (tr["IFLAG"].astype(str).str[0] == "O")
                & (tr["TRACK_TYPE"] == "main") & tr["SEASON"].between(1980, 2015)]
    for cid, col, thr in (("T5", "USA_WIND", 90.0), ("T6", "USA_PRES", 50.0)):
        pres = 100.0 * usable[col].notna().mean() if len(usable) else 0.0
        checks.append(Check(cid, f"{col} present on usable synoptic rows", cid == "T5",
                            f">= {thr}%", pres >= thr, round(pres, 2),
                            {"n_usable_rows": int(len(usable))}))

    # --- T7 physical ranges ----------------------------------------------
    viol = {c: int(((tr[c] < lo) | (tr[c] > hi)).sum()) for c, (lo, hi) in RANGES.items()}
    checks.append(Check("T7", "Wind/pressure within physical ranges", True, "== 0",
                        sum(viol.values()) == 0, sum(viol.values()), viol))

    # --- S1 frame open errors ---------------------------------------------
    e = int(fr["error"].notna().sum()) if "error" in fr else 0
    checks.append(Check("S1", "All satellite frames open without error", True, "== 0",
                        e == 0, e))

    # --- S2 storm ID consistency (filename vs attribute) ------------------
    mism = int((fr_ok["sid_fname"] != fr_ok["sid"]).sum())
    checks.append(Check("S2", "Frame filename SID == embedded TC_serial_number", True,
                        "== 0", mism == 0, mism, {"n_frames": int(len(fr_ok))}))

    # --- S3 SIDs resolve to IBTrACS --------------------------------------
    unres = sorted(sids - set(tr["SID"]))
    checks.append(Check("S3", "Every frame SID resolves in IBTrACS v04r01", True,
                        "== 0 unresolved", len(unres) == 0, len(unres),
                        {"unresolved": unres[:10], "n_sids": len(sids)}))

    # --- S4 grid geometry --------------------------------------------------
    shapes = fr_ok.groupby(["nlat", "nlon"]).size().to_dict()
    ok = set(shapes) == {(301, 301)}
    checks.append(Check("S4", "All frames are 301x301", False, "single shape 301x301",
                        ok, str({str(k): v for k, v in shapes.items()})))

    # --- S5 duplicate satellite frames + dedup key ------------------------
    g = fr_ok.groupby(["sid", "htime"]).size()
    dup = int((g > 1).sum())
    vza_ok = float(fr_ok["VZA"].notna().mean() * 100)
    checks.append(Check("S5", "Duplicate frames per (SID,time) are resolvable by VZA",
                        True, "VZA present on 100% of frames", vza_ok == 100.0,
                        round(vza_ok, 2),
                        {"n_sid_time": int(len(g)), "n_duplicated": dup,
                         "max_frames_at_one_time": int(g.max()),
                         "frames_before": int(len(fr_ok)), "frames_after_dedup": int(len(g))}))

    # --- dedup then join --------------------------------------------------
    ded = fr_ok.sort_values("VZA").groupby(["sid", "htime"], as_index=False).first()

    # --- S6 temporal join --------------------------------------------------
    matched, dts, seps, missing_frames = 0, [], [], 0
    for sid, grp in ded.groupby("sid"):
        track = sub[sub["SID"] == sid].sort_values("ISO_TIME")
        if track.empty:
            continue
        m = pd.merge_asof(
            grp[["sid", "htime", "CentLat", "CentLon"]].sort_values("htime")
               .rename(columns={"sid": "SID"}),
            track, left_on="htime", right_on="ISO_TIME", by="SID",
            direction="nearest", tolerance=pd.Timedelta(minutes=args.tolerance_min))
        got = m[m["ISO_TIME"].notna()]
        matched += len(got)
        dts.extend(((got["htime"] - got["ISO_TIME"]).abs().dt.total_seconds() / 60).tolist())
        v = got.dropna(subset=["CentLat", "LAT"])
        seps.extend(haversine_km(v["CentLat"].to_numpy(), v["CentLon"].to_numpy(),
                                 v["LAT"].to_numpy(), v["LON"].to_numpy()).tolist())
        # synoptic best-track rows of this storm with no frame
        syn = track[track["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])
                    & (track["ISO_TIME"].dt.minute == 0)]
        span = syn[(syn["ISO_TIME"] >= grp["htime"].min())
                   & (syn["ISO_TIME"] <= grp["htime"].max())]
        missing_frames += int(len(span) - len(set(span["ISO_TIME"]) & set(grp["htime"])))

    pct_join = 100.0 * matched / len(ded) if len(ded) else 0.0
    checks.append(Check("S6", "Frames join to best track within tolerance", True,
                        "== 100%", pct_join == 100.0, round(pct_join, 2),
                        {"attempted": int(len(ded)), "matched": matched,
                         "tolerance_min": args.tolerance_min}))

    # --- S7 timestamp offset ----------------------------------------------
    dts = np.array(dts)
    checks.append(Check("S7", "|dt| within tolerance for all joined frames", True,
                        f"max <= {args.tolerance_min} min",
                        bool(dts.size and dts.max() <= args.tolerance_min),
                        float(dts.max()) if dts.size else None,
                        {"median": float(np.median(dts)) if dts.size else None,
                         "pct_exact_zero": round(100 * float((dts == 0).mean()), 2)
                         if dts.size else None}))

    # --- S8 spatial agreement ----------------------------------------------
    seps = np.array(seps)
    pct50 = 100 * float((seps < 50).mean()) if seps.size else 0.0
    checks.append(Check("S8", "Frame centre agrees with best track < 50 km", True,
                        ">= 99% of rows", pct50 >= 99.0, round(pct50, 2),
                        {"median_km": round(float(np.median(seps)), 2) if seps.size else None,
                         "max_km": round(float(seps.max()), 2) if seps.size else None}))

    # --- S9 missing frames -------------------------------------------------
    checks.append(Check("S9", "Synoptic best-track rows lacking a frame (within span)",
                        False, "reported, not enforced", True, missing_frames))

    # --- S10 image content -------------------------------------------------
    const = int(fr_ok["irwin_constant"].sum())
    pct99 = 100 * float((fr_ok["irwin_valid_pct"] > 99).mean())
    checks.append(Check("S10", "IRWIN not constant/empty", True, "0 constant frames",
                        const == 0, const,
                        {"pct_frames_over_99pct_valid": round(pct99, 2),
                         "irwin_min_K": round(float(fr_ok["irwin_min"].min()), 2),
                         "irwin_max_K": round(float(fr_ok["irwin_max"].max()), 2)}))

    # --- S11 class distribution non-degenerate -----------------------------
    nat = sub["NATURE"].value_counts().to_dict()
    checks.append(Check("S11", "NATURE distribution non-degenerate", False,
                        ">= 2 classes present", len(nat) >= 2, len(nat), nat))

    blocking_failed = [c.id for c in checks if c.blocking and c.passed is False]
    report = {
        "gate": "phase1_preliminary",
        "sample_dir": str(args.sample_dir),
        "n_frames": int(len(fr)), "n_storms": len(sids),
        "n_track_rows_for_sample_storms": int(len(sub)),
        "checks": [asdict(c) for c in checks],
        "summary": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c.passed),
            "failed": sum(1 for c in checks if c.passed is False),
            "blocking_failures": blocking_failed,
            "gate_status": "PASS" if not blocking_failed else "FAIL",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"{'ID':<5}{'B':<3}{'RESULT':<8}{'CHECK':<52}VALUE")
    print("-" * 104)
    for c in checks:
        res = "PASS" if c.passed else ("FAIL" if c.passed is False else "n/a")
        print(f"{c.id:<5}{'*' if c.blocking else ' ':<3}{res:<8}{c.name[:50]:<52}{c.value}")
    s = report["summary"]
    print("-" * 104)
    print(f"GATE: {s['gate_status']}   passed {s['passed']}/{s['total']}   "
          f"blocking failures: {s['blocking_failures'] or 'none'}   (* = blocking)")
    print(f"REPORT: {args.out}")
    return 0 if not blocking_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
