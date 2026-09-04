"""Phase 4 Task 4-6: configurable, deterministic HURSAT-B1 sample downloader.

Selects storms FROM the frozen Phase 2 storm-level split manifest only
(never an arbitrary IBTrACS storm) so every downloaded storm is guaranteed
split-compatible, then downloads + extracts exactly those storms' HURSAT-B1
archives. Idempotent: already-downloaded archives and already-extracted
frames are skipped.

Selection modes (mutually exclusive):
  --storm-ids SID [SID ...]   explicit list
  --season SEASON              every frozen-split storm in one season
  --sample-storms N [--seed S] deterministic stratified sample across
                                train/val/test, proportional to split size

Always supports --dry-run (discovery only, no downloads).

Usage:
    python ml/scripts/download_hursat_sample.py --dry-run --sample-storms 20
    python ml/scripts/download_hursat_sample.py --season 2005
    python ml/scripts/download_hursat_sample.py --sample-storms 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, REPORT_DIR, get_data_root, zone  # noqa: E402
from ml.geostrom_ml.satellite.discovery import season_of_sid  # noqa: E402
from ml.geostrom_ml.satellite.download import (  # noqa: E402
    download_adt_archive, download_storm_archive, extract_archive,
)
from ml.geostrom_ml.satellite.pipeline import load_split_map  # noqa: E402


def stratified_sample(split_map: dict[str, str], n: int, seed: int) -> list[str]:
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid, split in split_map.items():
        by_split[split].append(sid)
    for v in by_split.values():
        v.sort()  # deterministic order before seeding

    total = sum(len(v) for v in by_split.values())
    rng = random.Random(seed)
    selected: list[str] = []
    for split, sids in by_split.items():
        quota = max(1, round(n * len(sids) / total)) if sids else 0
        selected.extend(rng.sample(sids, min(quota, len(sids))))
    return sorted(set(selected))[:n] if len(selected) > n else sorted(set(selected))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--storm-ids", nargs="+")
    g.add_argument("--season", type=int)
    g.add_argument("--sample-storms", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--splits", type=Path, default=MANIFEST_DIR / "splits_v1.json")
    ap.add_argument("--skip-adt", action="store_true",
                    help="skip attempting the (best-effort, optional) ADT-HURSAT Scene-label download")
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "hursat_download_log.json")
    args = ap.parse_args()

    split_map = load_split_map(args.splits)

    if args.storm_ids:
        sids = [s for s in args.storm_ids if s in split_map]
        skipped = [s for s in args.storm_ids if s not in split_map]
        if skipped:
            print(f"SKIPPED (not in frozen split manifest): {skipped}", file=sys.stderr)
    elif args.season is not None:
        sids = sorted(s for s in split_map if season_of_sid(s) == args.season)
    else:
        sids = stratified_sample(split_map, args.sample_storms, args.seed)

    root = get_data_root(create=True)
    raw_dir = zone("raw", "hursat", create=True)
    interim_dir = zone("interim", "hursat", create=True)
    listing_cache = zone("raw", "hursat_listings", create=True)
    adt_dir = zone("samples", "adt", create=True)  # reuse the Phase 1 ADT sample location

    results = []
    for sid in sids:
        season = season_of_sid(sid)
        res = download_storm_archive(sid, season, raw_dir, listing_cache, dry_run=args.dry_run)
        rec = {"sid": sid, "season": season, "split": split_map[sid],
               "status": res.status, "bytes": res.bytes, "detail": res.detail}
        if res.status in ("downloaded", "cached") and not args.dry_run and res.archive_path:
            extracted = extract_archive(Path(res.archive_path), interim_dir)
            rec["n_frames_extracted"] = len(extracted)
        if not args.skip_adt:
            adt_res = download_adt_archive(sid, adt_dir, dry_run=args.dry_run)
            rec["adt_status"] = adt_res.status
        results.append(rec)
        print(f"{sid} ({rec['split']}, {season})  {res.status:<20} "
              f"{(res.bytes or 0)/1e6:>7.2f} MB  adt={rec.get('adt_status', 'skipped')}  "
              f"{res.detail or ''}")

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    log = {
        "dry_run": args.dry_run, "seed": args.seed, "n_requested": len(sids),
        "status_counts": status_counts,
        "total_mb_downloaded": round(sum(r["bytes"] or 0 for r in results
                                         if r["status"] == "downloaded") / 1e6, 2),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n{status_counts}  total downloaded: {log['total_mb_downloaded']} MB")
    print(f"LOG: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
