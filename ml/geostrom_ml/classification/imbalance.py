"""Phase 5 Task 5: class-imbalance strategy -- designed and applied to the
non-deep-learning baselines here, specified (not trained) for Phase 6.

Explicitly NOT used in this phase: naive random oversampling (can duplicate
a storm's images across a resample and, worse, is the mechanism by which
oversampling could ever leak a storm's images into multiple splits if
applied carelessly), synthetic image generation, and SMOTE on pixels --
all forbidden by the Phase 5 task and never implemented anywhere in this
package.
"""

from __future__ import annotations

import pandas as pd


def compute_class_weights(train_labels: pd.Series) -> dict[str, float]:
    """Balanced class weights computed from the TRAINING split ONLY.

    `weight[c] = n_train_samples / (n_classes * n_train_samples_of_c)` --
    scikit-learn's standard 'balanced' formula, reproduced explicitly here
    (rather than only relying on `class_weight='balanced'` inside a
    estimator) so the weights are visible, logged, and testable on their
    own, and so the training-split-only rule is enforced by the function
    signature itself (it only ever receives training labels).
    """
    counts = train_labels.value_counts()
    n_classes = len(counts)
    n_samples = len(train_labels)
    return {str(cls): float(n_samples / (n_classes * n)) for cls, n in counts.items()}


PHASE_6_IMBALANCE_STRATEGY: dict = {
    "approach": "weighted_loss",
    "rationale": (
        "With 4-6x sample-count imbalance (CDO 188 vs Shear 83 in the "
        "recommended taxonomy, worse pre-merge) and only 12 storms total, "
        "any resampling scheme risks either duplicating a single storm's "
        "frames (inflating apparent performance on that storm's imagery) "
        "or requires synthesising new imagery -- both explicitly forbidden "
        "this phase. Weighted loss changes the OPTIMISATION objective, not "
        "the DATA, so it cannot introduce a storm-level leak."
    ),
    "class_weights": "computed via compute_class_weights() on the TRAINING "
                      "split's final_class labels only, every re-run",
    "applies_to": [
        "sklearn LogisticRegression(class_weight=...) -- already applied in "
        "this phase's baseline, see baselines.py",
        "Phase 6 CNN: torch.nn.CrossEntropyLoss(weight=...) with the same "
        "training-split-only class_weights tensor",
    ],
    "additional_measures_for_phase_6": [
        "label_smoothing (e.g. 0.05-0.1) to reduce overconfidence on the "
        "majority class without touching the data",
        "macro-F1 (not accuracy) as the model-selection metric, computed on "
        "the validation split only, exactly as in this phase's baseline",
        "per-class recall and the confusion matrix reported every "
        "evaluation, not just an aggregate score, so a class the model "
        "learns to ignore is visible immediately",
    ],
    "explicitly_forbidden": [
        "naive random oversampling of minority-class images (duplicates a "
        "storm's frames; only 12 storms means near-certain storm dominance "
        "within a class)",
        "SMOTE or any pixel-space synthetic sample generation",
        "any resampling applied after the storm-level split (must never be "
        "able to move a storm's samples across the train/val/test boundary)",
    ],
    "storm_level_independence": (
        "Whatever Phase 6 does, it must operate strictly within one split's "
        "already-assigned storms -- the frozen splits_v1.json storm-level "
        "partition (reused unchanged since Phase 2) is the only split "
        "boundary, and no imbalance technique may cross it."
    ),
}
