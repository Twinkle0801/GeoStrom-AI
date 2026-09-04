"""Phase 4 Task 1-3: real (not extrapolated) HURSAT-B1 archive discovery.

Fetches the actual NCEI directory listing for every requested season (small
HTML pages only, no imagery) and cross-references it against the frozen
Phase 2 storm-level split manifest to report exactly how many storms/files/
bytes the requested basin+season MVP subset requires -- replacing Phase 1's
sample-extrapolated `ml/scripts/estimate_dataset.py` figures with measured
per-season truth for every season actually requested.

This script NEVER downloads imagery. Always run this (or trust its cached
report) before `download_hursat_sample.py`, per the Phase 4 task's explicit
"determine coverage before downloading" instruction.

Usage:
    python ml/scripts/discover_hursat_archive.py --seasons 1980 2015
    python ml/scripts/discover_hursat_archive.py --seasons 1980 2015 --basin NA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, REPORT_DIR, zone  # noqa: E402
from ml.geostrom_ml.satellite.discovery import fetch_season_listing, season_of_sid  # noqa: E402
from ml.geostrom_ml.satellite.pipeline import load_split_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs=2, type=int, default=[1980, 2015],
                    metavar=("START", "END"))
    ap.add_argument("--basin", default="NA")
    ap.add_argument("--splits", type=Path, default=MANIFEST_DIR / "splits_v1.json")
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "hursat_archive_discovery.json")
    args = ap.parse_args()

    split_map = load_split_map(args.splits)
    storms_by_season: dict[int, list[str]] = {}
    for sid in split_map:
        storms_by_season.setdefault(season_of_sid(sid), []).append(sid)

    listing_cache = zone("raw", "hursat_listings", create=True)
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))

    per_season = {}
    total_bytes = total_files = total_storms_covered = total_storms_expected = 0
    for season in seasons:
        entries = fetch_season_listing(season, listing_cache)
        available_sids = {e["sid"] for e in entries}
        expected = storms_by_season.get(season, [])
        covered = [s for s in expected if s in available_sids]
        sizes = [e["bytes"] for e in entries if e["bytes"]]
        rec = {
            "hursat_archives_in_listing": len(entries),
            "frozen_split_storms_expected": len(expected),
            "frozen_split_storms_covered": len(covered),
            "pct_covered": round(100 * len(covered) / len(expected), 2) if expected else None,
            "missing_sids": sorted(set(expected) - available_sids)[:15],
            "total_gb_all_listed_archives": round(sum(sizes) / 1e9, 3) if sizes else 0.0,
            "expected_storm_gb": round(
                sum(e["bytes"] for e in entries if e["sid"] in covered and e["bytes"]) / 1e9, 3
            ) if entries else 0.0,
        }
        per_season[str(season)] = rec
        total_bytes += sum(e["bytes"] for e in entries if e["sid"] in covered and e["bytes"])
        total_files += len(covered)
        total_storms_covered += len(covered)
        total_storms_expected += len(expected)
        print(f"{season}  listing={rec['hursat_archives_in_listing']:>4}  "
              f"expected(frozen split)={rec['frozen_split_storms_expected']:>4}  "
              f"covered={rec['frozen_split_storms_covered']:>4} ({rec['pct_covered']}%)  "
              f"{rec['expected_storm_gb']:>6} GB")

    report = {
        "MEASURED": "Every number below comes from a real NCEI directory listing fetch, "
                    "not extrapolation.",
        "basin": args.basin,
        "seasons": seasons,
        "splits_source": str(args.splits),
        "per_season": per_season,
        "totals": {
            "storms_expected_in_frozen_split": total_storms_expected,
            "storms_with_hursat_archive": total_storms_covered,
            "pct_covered": round(100 * total_storms_covered / total_storms_expected, 2)
            if total_storms_expected else None,
            "estimated_download_gb_for_full_coverage": round(total_bytes / 1e9, 2),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = report["totals"]
    print(f"\nTOTAL: {t['storms_with_hursat_archive']}/{t['storms_expected_in_frozen_split']} "
          f"frozen-split storms covered ({t['pct_covered']}%), "
          f"~{t['estimated_download_gb_for_full_coverage']} GB to download for full coverage.")
    print(f"REPORT: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
