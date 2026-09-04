"""Phase 5 Task 7: non-deep-learning classification baselines.

Explicitly NOT deep learning: no CNN/ResNet/EfficientNet/ViT/GRU/LSTM/
Transformer anywhere in this module, per the Phase 5 task's explicit
prohibition -- these baselines exist to set a minimum bar before Phase 6,
not to compete with it.

All randomness is seeded (`RANDOM_SEED`) for the reproducibility
requirement (Task 10): identical inputs must always produce identical
fitted models and identical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


@dataclass
class MajorityClassBaseline:
    """Predicts the training split's single most frequent class always."""

    majority_class: str | None = None

    def fit(self, y_train: pd.Series) -> "MajorityClassBaseline":
        self.majority_class = y_train.value_counts().idxmax()
        return self

    def predict(self, n: int) -> list[str]:
        if self.majority_class is None:
            raise RuntimeError("fit() must be called before predict()")
        return [self.majority_class] * n


def build_logistic_regression_pipeline(class_weight: dict[str, float] | str = "balanced") -> Pipeline:
    """Median-impute -> standardise -> multinomial logistic regression.

    `class_weight='balanced'` (sklearn-computed, TRAINING split only, since
    the pipeline is only ever `.fit()` on training data) is the imbalance
    strategy applied to this baseline, matching `imbalance.py`'s Phase 6
    plan of a weighted-loss approach rather than resampling.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight, max_iter=2000, random_state=RANDOM_SEED,
        )),
    ])


def build_lightgbm_model(class_weight: dict[str, float] | None = None):
    """Multiclass LightGBM classifier (already a Phase 2 dependency).

    Justification for including a tree-based baseline alongside logistic
    regression, per the task's "if justified by the existing dependency
    stack" allowance: `lightgbm==4.7.0` is already pinned in
    `ml/requirements.txt` (Phase 2's intensity/track baselines use it), so
    this introduces no new dependency.
    """
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="multiclass",
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,  # small dataset (a few hundred train rows) -- avoid over-pruning
        verbosity=-1,
    )


def impute_median_from_train(train_X: pd.DataFrame, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Fit a median imputer on `train_X` only, apply to every given frame.

    Standalone helper (mirrors what `build_logistic_regression_pipeline`
    does internally) for baselines/tests that need imputed features without
    going through the full sklearn Pipeline, e.g. LightGBM's `.fit`.
    """
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X)
    return [pd.DataFrame(imputer.transform(f), columns=train_X.columns, index=f.index) for f in frames]
