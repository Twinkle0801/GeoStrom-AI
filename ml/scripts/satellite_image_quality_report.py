"""Phase 4 "Image Quality Report" (task section, verbatim requirements):

  sample satellite images; histogram/statistics of IRWIN values; invalid
  pixel statistics; per-season frame counts; per-storm frame counts;
  temporal offset distribution; spatial separation distribution; Scene
  class distribution; duplicate statistics.

Reads the already-built sample index (Parquet) and canonical Zarr store
from `build_satellite_dataset.py` -- this script performs no ingestion or
QC of its own, it only reports on what the pipeline already produced.

Per-season/per-storm counts, Scene class distribution, and duplicate
statistics are already in `ml/reports/satellite_qc_gate.json`
(`ml.geostrom_ml.satellite.qc`); this script adds the pieces that report
does not cover: distribution statistics (not just pass/fail counts) for
temporal offset and spatial separation, a real IRWIN value histogram
sampled from the canonical Zarr store, invalid-pixel-fraction statistics,
and rendered sample thumbnails.

Usage:
    python ml/scripts/satellite_image_quality_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, get_data_root  # noqa: E402
from ml.geostrom_ml.satellite.schema import DATASET_VERSION  # noqa: E402


def _distribution(series) -> dict:
    import numpy as np

    vals = series.dropna().to_numpy(dtype="float64")
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "min": round(float(vals.min()), 3),
        "max": round(float(vals.max()), 3),
        "mean": round(float(vals.mean()), 3),
        "median": round(float(np.median(vals)), 3),
        "p95": round(float(np.percentile(vals, 95)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    ap.add_argument("--n-thumbnails", type=int, default=6)
    ap.add_argument("--n-histogram-samples", type=int, default=50,
                    help="number of frames to sample for the IRWIN pixel-value histogram")
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "satellite_image_quality.json")
    ap.add_argument("--figures-dir", type=Path, default=REPORT_DIR / "figures")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import zarr

    root = get_data_root()
    parquet_path = root / "processed" / "satellite" / args.dataset_version / "sample_index.parquet"
    zarr_path = root / "processed" / "satellite" / args.dataset_version / "images.zarr"
    if not parquet_path.exists():
        print(f"No sample index at {parquet_path}. Run build_satellite_dataset.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet_path)
    zroot = zarr.open_group(store=zarr.storage.LocalStore(str(zarr_path)), mode="r")
    irwin_k = zroot["irwin_k"]
    valid_mask = zroot["valid_mask"]

    report: dict = {
        "n_samples": int(len(df)),
        "temporal_offset_minutes_distribution": _distribution(df["temporal_offset_minutes"]),
        "spatial_distance_km_distribution": _distribution(df["spatial_distance_km"]),
        "vza_distribution": _distribution(df["vza"]),
    }

    # ---- IRWIN pixel-value histogram (sampled, not full population) -------
    rng = np.random.default_rng(42)  # deterministic sample selection
    idx = df["zarr_index"].dropna().astype(int).to_numpy()
    sample_idx = rng.choice(idx, size=min(args.n_histogram_samples, len(idx)), replace=False) \
        if len(idx) else np.array([], dtype=int)
    sample_idx.sort()

    all_valid_pixels = []
    invalid_fracs = []
    for i in sample_idx:
        k = np.asarray(irwin_k[int(i)])
        m = np.asarray(valid_mask[int(i)])
        all_valid_pixels.append(k[m])
        invalid_fracs.append(1.0 - m.mean())
    if all_valid_pixels:
        pixels = np.concatenate(all_valid_pixels)
        counts, edges = np.histogram(pixels, bins=20, range=(150.0, 350.0))
        report["irwin_value_histogram_kelvin"] = {
            "sampled_frames": int(len(sample_idx)),
            "sampled_pixels": int(pixels.size),
            "bin_edges_k": [round(float(e), 1) for e in edges],
            "counts": [int(c) for c in counts],
            "mean_k": round(float(pixels.mean()), 2),
            "std_k": round(float(pixels.std()), 2),
        }
        report["invalid_pixel_fraction"] = _distribution(pd.Series(invalid_fracs))
    else:
        report["irwin_value_histogram_kelvin"] = None
        report["invalid_pixel_fraction"] = {"n": 0}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Stats written: {args.out}")

    # ---- rendered figures --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.figures_dir.mkdir(parents=True, exist_ok=True)

        # Sample thumbnails: earliest N distinct storms' first frame each.
        thumb_rows = (df.sort_values(["storm_id", "satellite_timestamp"])
                        .groupby("storm_id").first().reset_index()
                        .head(args.n_thumbnails))
        n = len(thumb_rows)
        if n:
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
            axes = np.atleast_1d(axes)
            for ax, (_, row) in zip(axes, thumb_rows.iterrows()):
                k = np.asarray(irwin_k[int(row["zarr_index"])])
                im = ax.imshow(k, cmap="gray_r", vmin=190, vmax=300)
                ax.set_title(f"{row['storm_id']}\n{row['satellite_timestamp']}", fontsize=7)
                ax.axis("off")
            fig.suptitle("Sample IRWIN frames (brightness temperature, K; darker = colder/higher cloud)")
            fig.colorbar(im, ax=list(axes), shrink=0.7, label="Kelvin")
            fig.savefig(args.figures_dir / "satellite_sample_thumbnails.png", dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"Figure written: {args.figures_dir / 'satellite_sample_thumbnails.png'}")

        if all_valid_pixels:
            fig, axs = plt.subplots(1, 3, figsize=(13, 4))
            axs[0].hist(pixels, bins=40, range=(150, 350), color="steelblue")
            axs[0].set_title("IRWIN pixel value distribution (sampled)")
            axs[0].set_xlabel("Kelvin")
            axs[1].hist(df["temporal_offset_minutes"].dropna(), bins=30, color="darkorange")
            axs[1].set_title("Temporal offset to matched IBTrACS row")
            axs[1].set_xlabel("minutes")
            axs[2].hist(df["spatial_distance_km"].dropna(), bins=30, color="seagreen")
            axs[2].axvline(50.0, color="red", linestyle="--", label="50 km QC gate")
            axs[2].set_title("Spatial separation to best track")
            axs[2].set_xlabel("km")
            axs[2].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.figures_dir / "satellite_distributions.png", dpi=110)
            plt.close(fig)
            print(f"Figure written: {args.figures_dir / 'satellite_distributions.png'}")
    except ImportError:
        print("matplotlib not available -- skipped figure rendering (stats JSON still written).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
