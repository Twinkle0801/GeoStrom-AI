"""Dataset manifest generator (Phase 1, Task 14).

Records provenance for every file retrieved into DATA_ROOT: source, official URL,
version, size, SHA-256, DATA_ROOT-relative path, retrieval date, and verification
status. The manifest is committed to the repository; the data itself is not.

Checksums are computed from the bytes actually on disk. Nothing is fabricated:
a file that is absent is recorded as MISSING rather than given a placeholder.

Usage:
    python ml/scripts/make_manifest.py [--verify]   # --verify re-checks existing hashes
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, get_data_root  # noqa: E402

IBTRACS_CSV = ("https://www.ncei.noaa.gov/data/international-best-track-archive-"
               "for-climate-stewardship-ibtracs/v04r01/access/csv")
HURSAT_B1 = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
ADT = ("https://www.ncei.noaa.gov/data/oceans/archive/arc0239/0307249/1.1/"
       "data/0-data")

# (relative path under DATA_ROOT, dataset, source URL, version, notes)
ENTRIES = [
    ("raw/ibtracs/ibtracs.NA.list.v04r01.csv", "IBTrACS",
     f"{IBTRACS_CSV}/ibtracs.NA.list.v04r01.csv", "v04r01",
     "North Atlantic basin subset. 174 columns. Row 2 is a units row."),
    ("raw/ibtracs/ibtracs.NI.list.v04r01.csv", "IBTrACS",
     f"{IBTRACS_CSV}/ibtracs.NI.list.v04r01.csv", "v04r01",
     "North Indian basin subset, retrieved for the basin comparison."),
    ("samples/hursat/HURSAT_b1_v06_1995222N24265_GABRIELLE_c20170721.tar.gz",
     "HURSAT-B1", f"{HURSAT_B1}/1995/HURSAT_b1_v06_1995222N24265_GABRIELLE_c20170721.tar.gz",
     "v06", "Verification sample storm (1995 season)."),
    ("samples/hursat/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz",
     "HURSAT-B1", f"{HURSAT_B1}/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz",
     "v06", "Verification sample storm (2005 season, Hurricane Katrina)."),
    ("samples/hursat/HURSAT_b1_v06_2015193N35285_CLAUDETTE_c20170721.tar.gz",
     "HURSAT-B1", f"{HURSAT_B1}/2015/HURSAT_b1_v06_2015193N35285_CLAUDETTE_c20170721.tar.gz",
     "v06", "Verification sample storm (2015 season, end of HURSAT-B1 coverage)."),
]

LICENSING = {
    "IBTrACS": {
        "producer": "NOAA National Centers for Environmental Information (NCEI)",
        "citation": ("Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and "
                     "C. J. Neumann (2010): The International Best Track Archive for "
                     "Climate Stewardship (IBTrACS). Bull. Amer. Meteor. Soc., 91, 363-376."),
        "landing_page": "https://www.ncei.noaa.gov/products/international-best-track-archive",
        "terms": "US Government work / NOAA open data. Attribution requested.",
        "verified": "landing page and column documentation retrieved 2026-09-01",
    },
    "HURSAT-B1": {
        "producer": "NOAA NCEI (Ken Knapp)",
        "citation": ("Knapp, K. R., and J. P. Kossin (2007): New global tropical cyclone "
                     "data set from ISCCP B1 geostationary satellite observations. "
                     "J. Appl. Remote Sens., 1, 013505."),
        "landing_page": "https://www.ncei.noaa.gov/products/hurricane-satellite-data",
        "terms": "US Government work / NOAA open data. Attribution requested.",
        "verified": "landing page and archive directory retrieved 2026-09-01",
    },
    "ADT-HURSAT": {
        "producer": "UW-Madison/CIMSS (Tim Olander) + NOAA NCEI; NCEI Accession 0307249",
        "citation": ("Kossin, J. P., K. R. Knapp, T. L. Olander, and C. S. Velden (2020): "
                     "Global increase in major tropical cyclone exceedance probability over "
                     "the past four decades. PNAS, 117(22), 11975-11980."),
        "landing_page": ("https://www.ncei.noaa.gov/products/"
                         "advanced-dvorak-technique-hurricane-satellite"),
        "terms": "US Government work / NOAA open data. Attribution requested.",
        "caveat": ("NCEI states this dataset 'should not be used to determine actual storm "
                   "intensities'. Use as a structural-label source and cross-check only; "
                   "IBTrACS remains ground truth for intensity."),
        "verified": "product page and data directory retrieved 2026-09-01",
    },
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-hash files and compare against the existing manifest")
    ap.add_argument("--out", type=Path, default=MANIFEST_DIR / "datasets.json")
    args = ap.parse_args()

    root = get_data_root()
    previous = {}
    if args.out.exists():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        previous = {f["path"]: f for f in prev.get("files", [])}

    files, mismatches = [], []
    for rel, dataset, url, version, notes in ENTRIES:
        p = root / rel
        rec = {"dataset": dataset, "path": rel, "source_url": url,
               "version": version, "notes": notes}
        if not p.exists():
            rec |= {"status": "MISSING", "bytes": None, "sha256": None,
                    "retrieved_utc": None}
        else:
            digest = sha256(p)
            rec |= {"status": "PRESENT", "bytes": p.stat().st_size, "sha256": digest,
                    "retrieved_utc": dt.datetime.fromtimestamp(
                        p.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")}
            old = previous.get(rel, {}).get("sha256")
            if args.verify and old and old != digest:
                rec["status"] = "CHECKSUM_MISMATCH"
                mismatches.append(rel)
        files.append(rec)

    # ADT sample files are numerous and small; summarise as a group
    adt_dir = root / "samples/adt"
    adt_files = sorted(adt_dir.glob("*.nc")) if adt_dir.exists() else []
    adt_group = {
        "dataset": "ADT-HURSAT", "path": "samples/adt/", "source_url": ADT,
        "version": "ADT v9.0 over HURSAT V07b (NCEI Accession 0307249, v1.1)",
        "status": "PRESENT" if adt_files else "MISSING",
        "n_files": len(adt_files),
        "bytes": sum(f.stat().st_size for f in adt_files),
        "sha256_of_sorted_file_hashes": hashlib.sha256(
            "".join(sha256(f) for f in adt_files).encode()).hexdigest() if adt_files else None,
        "file_naming": "<IBTrACS SID>.nc, one file per storm",
        "notes": "Per-storm ADT history files. Contains the Dvorak 'Scene' field.",
    }

    manifest = {
        "manifest_version": 1,
        "project": "GeoStrom AI",
        "phase": "Phase 1 - Foundation & Dataset Verification",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(root),
        "data_root_note": ("Data lives outside the Git repository and outside OneDrive. "
                           "Only this manifest is version-controlled."),
        "verification_status_legend": {
            "PRESENT": "file on disk, checksum computed from actual bytes",
            "MISSING": "declared but not downloaded",
            "CHECKSUM_MISMATCH": "on-disk bytes differ from the recorded checksum",
        },
        "files": files,
        "file_groups": [adt_group],
        "licensing_and_attribution": LICENSING,
        "totals": {
            "n_declared": len(files),
            "n_present": sum(1 for f in files if f["status"] == "PRESENT"),
            "bytes_present": sum(f["bytes"] or 0 for f in files) + adt_group["bytes"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"DATA_ROOT: {root}")
    print(f"{'STATUS':<20}{'BYTES':>12}  DATASET / PATH")
    print("-" * 96)
    for f in files:
        print(f"{f['status']:<20}{(f['bytes'] or 0):>12,}  {f['dataset']}  {f['path']}")
    print(f"{adt_group['status']:<20}{adt_group['bytes']:>12,}  "
          f"{adt_group['dataset']}  {adt_group['path']} ({adt_group['n_files']} files)")
    print("-" * 96)
    t = manifest["totals"]
    print(f"{t['n_present']}/{t['n_declared']} declared files present; "
          f"{t['bytes_present']:,} bytes total on disk")
    if mismatches:
        print(f"CHECKSUM MISMATCHES: {mismatches}")
    print(f"MANIFEST: {args.out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
