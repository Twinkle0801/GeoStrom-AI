"""Phase 4: end-to-end satellite dataset build.

RAW SATELLITE DATA -> HURSAT ingestion -> metadata extraction -> IRWIN QC ->
spatial normalization -> temporal alignment -> IBTrACS join -> ADT-HURSAT
Scene join -> duplicate-frame resolution -> QC gate -> dataset manifest ->
Zarr + sample metadata index.

Reads only what `download_hursat_sample.py` (or an equivalent extraction
into DATA_ROOT/interim/hursat/) has already placed on disk -- this script
never downloads anything itself.

Outputs (all under DATA_ROOT, none committed):
  processed/satellite/<version>/images.zarr        canonical imagery
  processed/satellite/<version>/sample_index.parquet   sample metadata

Outputs (committed, small):
  ml/manifests/satellite_dataset_v1_manifest.json
  ml/reports/satellite_qc_gate.json

Usage:
    python ml/scripts/build_satellite_dataset.py --dry-run
    python ml/scripts/build_satellite_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, REPORT_DIR, get_data_root, zone  # noqa: E402
from ml.geostrom_ml.data.ibtracs import load_ibtracs_raw  # noqa: E402
from ml.geostrom_ml.satellite import hursat as hursat_mod  # noqa: E402
from ml.geostrom_ml.satellite.manifest import build_manifest  # noqa: E402
from ml.geostrom_ml.satellite.pipeline import load_split_map, run_pipeline  # noqa: E402
from ml.geostrom_ml.satellite.schema import DATASET_VERSION  # noqa: E402


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report frame/storm counts only; write nothing")
    ap.add_argument("--basin", default="NA")
    ap.add_argument("--tolerance-min", type=int, default=90)
    ap.add_argument("--splits", type=Path, default=MANIFEST_DIR / "splits_v1.json")
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    args = ap.parse_args()

    # Two valid input locations: the Phase 1 verification sample (already
    # extracted, under samples/hursat/) and Phase 4 downloads extracted by
    # download_hursat_sample.py (under interim/hursat/).
    input_dirs = [zone("samples", "hursat"), zone("interim", "hursat")]
    adt_dir = zone("samples", "adt")
    root = get_data_root()

    if args.dry_run:
        frames = hursat_mod.discover_frame_files(input_dirs)
        sids = {hursat_mod.FRAME_RE.match(p.name).group("sid")
                for p in frames if hursat_mod.FRAME_RE.match(p.name)}
        print(f"DRY RUN: {len(frames)} extracted NetCDF frames found under {input_dirs}")
        print(f"         spanning {len(sids)} distinct storm IDs")
        print("No files written. Re-run without --dry-run to build the dataset.")
        return 0

    print("Loading full IBTrACS track (unfiltered, for observed/interpolated distinction)...")
    ib_raw, _overlap = load_ibtracs_raw(basin=args.basin)

    zarr_path = root / "processed" / "satellite" / args.dataset_version / "images.zarr"
    parquet_path = root / "processed" / "satellite" / args.dataset_version / "sample_index.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running pipeline: discover -> parse -> IRWIN QC -> dedup -> IBTrACS join -> "
          "ADT join -> split assignment -> Zarr write...")
    out = run_pipeline(
        interim_hursat_dir=input_dirs,
        adt_dir=adt_dir,
        splits_path=args.splits,
        ibtracs_full_track=ib_raw,
        zarr_out_path=zarr_path,
        tolerance_min=args.tolerance_min,
        dataset_version=args.dataset_version,
    )

    final_index = out["final_index"]
    final_index.to_parquet(parquet_path, index=False)
    print(f"Sample index written: {parquet_path} ({len(final_index)} rows)")
    print(f"Zarr store written  : {zarr_path}")

    qc_out = REPORT_DIR / "satellite_qc_gate.json"
    qc_out.write_text(json.dumps(out["qc_report"], indent=2, default=str), encoding="utf-8")
    print(f"QC report written   : {qc_out}")

    split_map = load_split_map(args.splits)
    manifest = build_manifest(
        data_root=root,
        final_index=final_index,
        qc_report=out["qc_report"],
        basin=args.basin,
        seasons_covered=sorted({int(s[:4]) for s in split_map}),
        storms_requested=sorted(set(out["inventory"].get("storm_id", []).dropna())
                                if "storm_id" in out["inventory"] else []),
        zarr_path=zarr_path,
        parquet_path=parquet_path,
        splits_path_repo_relative="ml/manifests/splits_v1.json",
        code_version=_git_commit(),
    )
    manifest_out = MANIFEST_DIR / "satellite_dataset_v1_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Manifest written    : {manifest_out}")

    summary = out["qc_report"]["summary"]
    counts = out["qc_report"]["counts"]
    print(f"\nGATE: {summary['gate_status']}  "
          f"({summary['passed']}/{summary['total_checks']} checks passed)")
    print(f"Final fused samples : {counts['14_final_fused_samples']}")
    print(f"Scene-labeled       : {manifest['scene_labeled_samples']}")
    return 0 if summary["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
