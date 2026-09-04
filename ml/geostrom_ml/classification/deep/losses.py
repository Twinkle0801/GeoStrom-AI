"""Phase 6 loss function: class-weighted, label-smoothed cross-entropy.

Directly implements Phase 5's already-specified (not yet trained)
imbalance strategy (`ml/geostrom_ml/classification/imbalance.py::
PHASE_6_IMBALANCE_STRATEGY`): "torch.nn.CrossEntropyLoss(weight=...) with
the same training-split-only class_weights tensor" plus label smoothing.
Retained unchanged rather than re-derived, per the Phase 6 task's
"retain the Phase 5 imbalance strategy unless experiments provide evidence
for improvement" instruction -- no such evidence was sought or found this
phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.geostrom_ml.classification.imbalance import compute_class_weights
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1


def build_class_weighted_loss(train_labels, label_smoothing: float = 0.05,
                              device: str = "cpu") -> nn.CrossEntropyLoss:
    """`train_labels`: the TRAINING split's `final_class` Series ONLY --
    the same discipline enforced and tested in Phase 5
    (`ml/tests/test_classification_leakage.py::TestVector10...`)."""
    weights_dict = compute_class_weights(train_labels)
    weight_tensor = torch.tensor(
        [weights_dict[c] for c in FINAL_CLASSES_V1], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
