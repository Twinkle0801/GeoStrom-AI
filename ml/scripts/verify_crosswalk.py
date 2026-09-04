"""HURSAT-B1 <-> IBTrACS crosswalk verification (Phase 1, Task 6).

Tests whether HURSAT-B1 storm archives can be reliably mapped to IBTrACS storms.

Hypothesis under test:
    HURSAT-B1 v06 filenames embed the IBTrACS SID directly, in the form
    HURSAT_b1_v06_<SID>_<NAME>_c<created>.tar.gz

Measures, per season:
  Direction A (coverage) : % of IBTrACS storms in a basin that have a HURSAT archive
  Direction B (validity) : % of HURSAT SIDs that resolve to a real IBTrACS storm
  Name agreement         : does the filename NAME token match IBTrACS NAME

Network use is limited to small HTML directory listings (~25 KB per season).
No imagery is downloaded by this script.

Usage:
    python ml/scripts/verify_crosswalk.py --seasons 1985 1995 2005 2015
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, zone  # noqa: E402

HURSAT_BASE = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"

# HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz
FNAME_RE = re.compile(
    r"HURSAT_b1_v06_(?P<sid>\d{4}\d{3}[NS]\d{5})_(?P<name>[A-Z0-9\-]+)_c(?P<created>\d{8})\.tar\.gz"
)
# IBTrACS SID grammar: YYYY + DDD(day-of-year) + hemisphere + lat(3) + lon(3)
SID_RE = re.compile(r"^\d{4}\d{3}[NS]\d{5}$")

NA_VALUES = ["", " ", "  ", "-999", "-9999", "-999.0", "MM"]


def fetch_season_listing(season: int, cache_dir: Path) -> list[dict]:
    """Return parsed HURSAT archive entries for a season (cached on disk)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"hursat_v06_{season}_listing.html"
    if cache.exists():
        html = cache.read_text(encoding="utf-8", errors="replace")
    else:
        resp = requests.get(f"{HURSAT_BASE}/{season}/", timeout=120)
        resp.raise_for_status()
        html = resp.text
        cache.write_text(html, encoding="utf-8")

    entries = []
    # size follows the filename in the Apache index row
    for m in FNAME_RE.finditer(html):
        tail = html[m.end(): m.end() + 260]
        size_m = re.search(r"(\d{4,})\s*<", tail)
        entries.append({
            "filename": m.group(0),
            "sid": m.group("sid"),
            "name": m.group("name"),
            "created": m.group("created"),
            "bytes": int(size_m.group(1)) if size_m else None,
        })
    # de-duplicate (index HTML lists each row twice: link text + href)
    seen, out = set(), []
    for e in entries:
        if e["filename"] not in seen:
            seen.add(e["filename"])
            out.append(e)
    return out


def load_ibtracs(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path, skiprows=[1], na_values=NA_VALUES, keep_default_na=False,
        low_memory=False, usecols=["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[1985, 1995, 2000, 2005, 2010, 2015])
    ap.add_argument("--ibtracs", nargs="+", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "crosswalk_verification.json")
    args = ap.parse_args()

    raw = zone("raw", "ibtracs")
    csvs = args.ibtracs or sorted(raw.glob("ibtracs.*.list.v04r01.csv"))
    if not csvs:
        print(f"No IBTrACS CSVs found in {raw}", file=sys.stderr)
        return 1

    ib = pd.concat([load_ibtracs(p) for p in csvs], ignore_index=True)
    # A storm's basin = basin of its first record (storms can cross basins)
    storm = (ib.sort_values("ISO_TIME")
               .groupby("SID")
               .agg(SEASON=("SEASON", "first"), BASIN=("BASIN", "first"), NAME=("NAME", "first"))
               .reset_index())

    report = {
        "hypothesis": "HURSAT-B1 v06 filenames embed the IBTrACS SID",
        "ibtracs_files": [str(p.name) for p in csvs],
        "ibtracs_storms_loaded": int(len(storm)),
        "ibtracs_basins_loaded": storm["BASIN"].value_counts().to_dict(),
        "seasons": {},
    }
    cache_dir = zone("samples", "hursat_listings", create=True)

    tot_sid_ok = tot_files = 0
    for season in args.seasons:
        entries = fetch_season_listing(season, cache_dir)
        sids = [e["sid"] for e in entries]
        malformed = [s for s in sids if not SID_RE.match(s)]
        dupes = [s for s in set(sids) if sids.count(s) > 1]

        ib_season = storm[storm["SEASON"] == season]
        ib_sids = set(ib_season["SID"])
        h_sids = set(sids)

        per_basin = {}
        for basin in ("NA", "NI"):
            b = ib_season[ib_season["BASIN"] == basin]
            if len(b) == 0:
                continue
            matched = set(b["SID"]) & h_sids
            per_basin[basin] = {
                "ibtracs_storms": int(len(b)),
                "with_hursat_archive": int(len(matched)),
                "pct_with_hursat": round(100 * len(matched) / len(b), 2),
                "missing_sids": sorted(set(b["SID"]) - h_sids)[:10],
            }

        # Direction B: HURSAT SIDs that exist in the loaded IBTrACS basins
        resolvable = h_sids & ib_sids

        # Name agreement on matched storms
        name_map = dict(zip(ib_season["SID"], ib_season["NAME"]))
        agree = mismatch = unnamed = 0
        examples = []
        for e in entries:
            if e["sid"] not in name_map:
                continue
            ib_name = str(name_map[e["sid"]]).upper()
            h_name = e["name"].upper()
            if h_name in ("MISSING", "UNNAMED", "NOT_NAMED") or ib_name in ("NOT_NAMED", "UNNAMED", ""):
                unnamed += 1
            elif h_name == ib_name:
                agree += 1
            else:
                mismatch += 1
                if len(examples) < 5:
                    examples.append({"sid": e["sid"], "hursat": h_name, "ibtracs": ib_name})

        sizes = [e["bytes"] for e in entries if e["bytes"]]
        report["seasons"][str(season)] = {
            "hursat_archives": len(entries),
            "hursat_sids_wellformed": len(sids) - len(malformed),
            "hursat_sids_malformed": malformed,
            "hursat_duplicate_sids": dupes,
            "hursat_sids_resolving_to_loaded_ibtracs": int(len(resolvable)),
            "coverage_by_basin": per_basin,
            "name_agreement": {"agree": agree, "mismatch": mismatch,
                               "unnamed_either_side": unnamed, "examples": examples},
            "archive_bytes": {
                "n": len(sizes),
                "total_gb": round(sum(sizes) / 1e9, 3) if sizes else None,
                "mean_mb": round(sum(sizes) / len(sizes) / 1e6, 2) if sizes else None,
                "min_mb": round(min(sizes) / 1e6, 2) if sizes else None,
                "max_mb": round(max(sizes) / 1e6, 2) if sizes else None,
            },
        }
        tot_sid_ok += len(sids) - len(malformed)
        tot_files += len(sids)

    report["overall"] = {
        "total_hursat_archives_scanned": tot_files,
        "total_wellformed_sids": tot_sid_ok,
        "pct_wellformed": round(100 * tot_sid_ok / tot_files, 2) if tot_files else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"IBTrACS storms loaded : {report['ibtracs_storms_loaded']:,} "
          f"{report['ibtracs_basins_loaded']}")
    print(f"HURSAT archives scanned: {tot_files}   well-formed SIDs: {tot_sid_ok} "
          f"({report['overall']['pct_wellformed']}%)")
    print()
    hdr = f"{'season':>6} {'hursat':>7} {'malf':>5} {'dup':>4} " \
          f"{'NA cov':>16} {'NI cov':>16} {'name ok':>8} {'GB':>7}"
    print(hdr); print("-" * len(hdr))
    for s, d in report["seasons"].items():
        na = d["coverage_by_basin"].get("NA")
        ni = d["coverage_by_basin"].get("NI")
        f = lambda x: f"{x['with_hursat_archive']}/{x['ibtracs_storms']} ({x['pct_with_hursat']}%)" if x else "-"
        n = d["name_agreement"]
        print(f"{s:>6} {d['hursat_archives']:>7} {len(d['hursat_sids_malformed']):>5} "
              f"{len(d['hursat_duplicate_sids']):>4} {f(na):>16} {f(ni):>16} "
              f"{n['agree']:>3}/{n['agree']+n['mismatch']:<4} {d['archive_bytes']['total_gb']:>7}")
    print(f"\nREPORT: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
