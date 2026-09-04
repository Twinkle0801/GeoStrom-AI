"""Phase 5 Task 2: reproducible Scene-label audit of the real Phase 4 dataset.

Reads ONLY the already-built Phase 4 sample index (never re-ingests, never
re-downloads, never re-runs the fusion pipeline). Writes
`ml/reports/phase5_scene_audit.json` plus overview figures.

Usage:
    python ml/scripts/audit_scene_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, get_data_root  # noqa: E402
from ml.geostrom_ml.classification.audit import build_scene_audit  # noqa: E402
from ml.geostrom_ml.satellite.schema import DATASET_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "phase5_scene_audit.json")
    ap.add_argument("--figures-dir", type=Path, default=REPORT_DIR / "figures")
    args = ap.parse_args()

    import pandas as pd

    root = get_data_root()
    parquet_path = root / "processed" / "satellite" / args.dataset_version / "sample_index.parquet"
    if not parquet_path.exists():
        print(f"No Phase 4 sample index at {parquet_path}. Run Phase 4's "
              f"build_satellite_dataset.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet_path)
    audit = build_scene_audit(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"Audit written: {args.out}")

    print(f"\nTotal samples: {audit['total_samples']}  storms: {audit['total_storms']}  "
          f"seasons: {audit['total_seasons']}")
    print("\nClass counts (sample-level -> storm-level):")
    for cls, n in audit["class_counts"].items():
        n_storms = audit["unique_storms_per_class"].get(cls, 0)
        pct_storms = audit["pct_storms_per_class"].get(cls, 0.0)
        print(f"  {cls:<12} {n:>4} samples ({audit['class_percentages'][cls]:>5.1f}%)  "
              f"{n_storms:>2} storms ({pct_storms:>5.1f}%)")
    print(f"\nClasses with zero test-split samples: {audit['classes_with_zero_test_samples']}")
    print(f"Classes with zero val-split samples : {audit['classes_with_zero_val_samples']}")
    print(f"Classes below the {audit['min_storms_for_generalization_threshold']}-storm "
          f"generalisation threshold: {audit['classes_below_generalization_storm_threshold']}")
    print(f"Missing labels: {audit['missing_labels']}   "
          f"Duplicate (sample_id, label) pairs: {audit['duplicate_sample_id_label_pairs']}")

    try:
        _render_figures(df, audit, args.figures_dir)
    except ImportError:
        print("matplotlib not available -- skipped figure rendering.")

    return 0


def _render_figures(df, audit: dict, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figures_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(audit["class_counts"], key=lambda c: -audit["class_counts"][c])

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [audit["class_counts"][c] for c in order]
    ax.bar(order, counts, color="steelblue")
    ax.set_title("Overall Scene distribution (sample-level, 627 real samples)")
    ax.set_ylabel("samples")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_scene_distribution_overall.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    split_df = pd.crosstab(df["scene_label"], df["split"]).reindex(order)
    split_df = split_df[[c for c in ("train", "val", "test") if c in split_df.columns]]
    split_df.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Scene distribution by split")
    ax.set_ylabel("samples")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_scene_distribution_by_split.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    storms = [audit["unique_storms_per_class"][c] for c in order]
    ax.bar(order, storms, color="darkorange")
    ax.axhline(audit["min_storms_for_generalization_threshold"], color="red", linestyle="--",
               label=f"generalisation threshold ({audit['min_storms_for_generalization_threshold']} storms)")
    ax.set_title("Number of storms per class (storm-level support)")
    ax.set_ylabel("storms")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_storms_per_class.png", dpi=110)
    plt.close(fig)

    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
