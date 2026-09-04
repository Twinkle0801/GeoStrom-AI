"""Phase 5 Task 2: formal Scene-label audit of the real Phase 4 dataset.

Operates on the Phase 4 sample index DataFrame (columns per
`ml/geostrom_ml/satellite/schema.py::SAMPLE_COLUMNS`). Computes every
statistic the Phase 5 task requires, at both sample level and storm level
-- storm level is the one that actually determines whether a class can
generalise, per the task's explicit instruction not to claim adequate
representation from image count alone.
"""

from __future__ import annotations

import pandas as pd

# A class present in fewer than this many storms cannot demonstrate
# generalisation across storms -- any per-class metric for it is really a
# statement about those specific 1-3 storms, not about the class in
# general. This threshold is a judgment call, stated explicitly rather than
# left implicit: with only 12 storms total in the current dataset, 4 storms
# is already a third of the whole dataset's storm population.
MIN_STORMS_FOR_GENERALIZATION = 4


def build_scene_audit(df: pd.DataFrame) -> dict:
    """Full Task 2 audit report as a JSON-serialisable dict."""
    total_samples = int(len(df))
    total_storms = int(df["storm_id"].nunique())
    total_seasons = int(df["season"].nunique())

    class_counts = df["scene_label"].value_counts(dropna=False)
    class_pct = (100 * class_counts / total_samples).round(2)

    missing_labels = int(df["scene_label"].isna().sum())

    dup_mask = df.duplicated(subset=["sample_id", "scene_label"], keep=False)
    n_duplicate_label_image = int(dup_mask.sum())

    storms_per_class = df.groupby("scene_label", dropna=False)["storm_id"].nunique()
    pct_storms_per_class = (100 * storms_per_class / total_storms).round(2)

    class_by_split = pd.crosstab(df["scene_label"], df["split"])
    class_by_storm = pd.crosstab(df["scene_label"], df["storm_id"])
    class_by_season = pd.crosstab(df["scene_label"], df["season"])
    storms_by_class_by_split = (
        df.groupby(["scene_label", "split"], dropna=False)["storm_id"].nunique().unstack(fill_value=0)
    )

    # Classes present in only one split -- a real, quantifiable risk that a
    # per-class test/val metric is undefined (zero support) or that a class
    # was never seen during training at all.
    n_splits_per_class = (class_by_split > 0).sum(axis=1)
    single_split_classes = n_splits_per_class[n_splits_per_class <= 1].index.tolist()
    zero_test_classes = class_by_split.index[class_by_split.get("test", 0) == 0].tolist() \
        if "test" in class_by_split.columns else class_by_split.index.tolist()
    zero_val_classes = class_by_split.index[class_by_split.get("val", 0) == 0].tolist() \
        if "val" in class_by_split.columns else class_by_split.index.tolist()

    thin_storm_classes = storms_per_class[storms_per_class < MIN_STORMS_FOR_GENERALIZATION].index.tolist()

    # Temporal distribution: samples per class per season, already captured
    # in class_by_season; also report each class's season span (min/max).
    temporal = {}
    for label, g in df.groupby("scene_label", dropna=False):
        key = str(label)
        temporal[key] = {
            "seasons_present": sorted(int(s) for s in g["season"].dropna().unique()),
            "n_seasons": int(g["season"].nunique()),
            "first_timestamp": str(g["satellite_timestamp"].min()),
            "last_timestamp": str(g["satellite_timestamp"].max()),
        }

    return {
        "total_samples": total_samples,
        "total_storms": total_storms,
        "total_seasons": total_seasons,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_percentages": {str(k): float(v) for k, v in class_pct.items()},
        "class_counts_by_split": {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in class_by_split.iterrows()
        },
        "class_counts_by_storm": {
            str(idx): {str(c): int(v) for c, v in row.items() if v > 0}
            for idx, row in class_by_storm.iterrows()
        },
        "class_counts_by_season": {
            str(idx): {str(c): int(v) for c, v in row.items() if v > 0}
            for idx, row in class_by_season.iterrows()
        },
        "unique_storms_per_class": {str(k): int(v) for k, v in storms_per_class.items()},
        "pct_storms_per_class": {str(k): float(v) for k, v in pct_storms_per_class.items()},
        "storms_per_class_by_split": {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in storms_by_class_by_split.iterrows()
        },
        "temporal_distribution_by_class": temporal,
        "missing_labels": missing_labels,
        "duplicate_sample_id_label_pairs": n_duplicate_label_image,
        "classes_present_in_only_one_split": single_split_classes,
        "classes_with_zero_test_samples": zero_test_classes,
        "classes_with_zero_val_samples": zero_val_classes,
        "min_storms_for_generalization_threshold": MIN_STORMS_FOR_GENERALIZATION,
        "classes_below_generalization_storm_threshold": thin_storm_classes,
    }
