"""Phase 5 Task 8: classification evaluation metrics.

MACRO-F1 is the primary model-selection metric (task's explicit
instruction, because of severe class imbalance -- accuracy alone would be
dominated by the majority class). All metrics are computed with
scikit-learn (already a Phase 2 dependency, no new package introduced).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    """Full metric suite for one (y_true, y_pred) pair over a fixed label set.

    `labels` should be the FULL taxonomy class list, not just classes
    present in this split -- so a class the model never had a chance to see
    or predict is still reported explicitly (support=0), never hidden by
    sklearn's default "only classes present" behaviour.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    present_in_true = sorted(set(y_true))
    absent_from_true = [c for c in labels if c not in present_in_true]

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred)) if present_in_true else float("nan")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_samples": len(y_true),
        "n_classes": len(labels),
        "labels": labels,
        "classes_absent_from_this_split": absent_from_true,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }


def macro_f1_over_present_classes(y_true, y_pred) -> tuple[float, list[str]]:
    """Macro-F1 computed ONLY over classes with support>0 in y_true.

    Documented as a SEPARATE, clearly-labelled number from the full-label
    `evaluate()` macro_f1 (which counts an absent class's precision/recall
    as 0 by construction via zero_division=0) -- reporting both makes the
    zero-test-support-class limitation visible rather than silently
    penalising or silently ignoring it.
    """
    present = sorted(set(y_true))
    f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)) if present else float("nan")
    return f1, present
