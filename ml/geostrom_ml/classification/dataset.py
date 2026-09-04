"""Phase 5 Task 6: canonical classification dataset index.

Derives a classification-specific index from Phase 4's already-fused
sample index. Never overwrites `scene_label` (kept verbatim as
`original_scene`); every sample is retained (including excluded ones) with
an explicit reason, per the task's "do not silently drop samples"
instruction.
"""

from __future__ import annotations

import pandas as pd

from ml.geostrom_ml.classification.taxonomy import LABEL_VERSION, apply_taxonomy

CLASSIFICATION_COLUMNS: list[str] = [
    "sample_id",
    "storm_id",
    "satellite_timestamp",
    "season",
    "split",
    "zarr_index",
    "original_scene",
    "final_class",
    "label_version",
    "qc_status",
    "exclusion_reason",
    "source_qc_status",  # the Phase 4 pipeline's own qc_status, for audit
]


def build_classification_index(sample_index: pd.DataFrame) -> pd.DataFrame:
    """Build the Phase 5 classification index from Phase 4's sample_index.

    Every row of `sample_index` produces exactly one row here -- nothing is
    dropped. `qc_status` is "included" or "excluded"; excluded rows always
    carry a non-null `exclusion_reason`.
    """
    out = sample_index.copy()

    mapped = out["scene_label"].apply(apply_taxonomy)
    out["final_class"] = mapped.apply(lambda t: t[0])
    out["exclusion_reason"] = mapped.apply(lambda t: t[1])
    out["original_scene"] = out["scene_label"]
    out["label_version"] = LABEL_VERSION
    out["source_qc_status"] = sample_index["qc_status"] if "qc_status" in sample_index.columns else None
    out["qc_status"] = out["final_class"].apply(lambda v: "included" if v is not None else "excluded")

    missing = [c for c in CLASSIFICATION_COLUMNS if c not in out.columns]
    for c in missing:
        out[c] = None
    return out[CLASSIFICATION_COLUMNS].reset_index(drop=True)


def split_summary(classification_index: pd.DataFrame) -> dict:
    """Included-only sample/storm counts per split -- the numbers that
    actually feed the classification baseline."""
    kept = classification_index[classification_index["qc_status"] == "included"]
    return {
        "samples_by_split": kept["split"].value_counts().to_dict(),
        "storms_by_split": kept.groupby("split")["storm_id"].nunique().to_dict(),
        "classes_by_split": {
            split: sorted(g["final_class"].unique().tolist())
            for split, g in kept.groupby("split")
        },
        "total_included": int(len(kept)),
        "total_excluded": int(len(classification_index) - len(kept)),
        "excluded_by_reason": classification_index.loc[
            classification_index["qc_status"] == "excluded", "exclusion_reason"
        ].value_counts().to_dict(),
    }
