"""Intensity baselines: Persistence, Ridge, LightGBM.

All three predict absolute wind (y_wind_abs_{h}h) at each configured
horizon. Per ML_ARCHITECTURE.md §6.5, one model is fit per horizon for the
learned baselines (Ridge, LightGBM) -- this is a simpler, more diagnostic
setup than a single multi-output model and matches the architecture's stated
design ("LightGBM ... one model per horizon").

Persistence needs no fit: it is defined as "future wind = current wind"
(this Phase 2 task's own definition, matching ML_ARCHITECTURE.md §6.5 Tier
1a). It is intentionally the strongest possible naive reference -- if a
learned model cannot beat it, that is reported, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.features.engineering import HORIZONS_H, flattened_feature_columns  # noqa: E402
from ml.geostrom_ml.models.base import BaselineModel  # noqa: E402

FEATURE_COLS = flattened_feature_columns()


def target_col(h: int) -> str:
    return f"y_wind_abs_{h}h"


class PersistenceIntensity(BaselineModel):
    """future wind = current (reference-time) wind, at every horizon."""

    task = "intensity"

    def __init__(self, horizons_h=HORIZONS_H):
        super().__init__(name="intensity_persistence_v1")
        self.horizons_h = horizons_h

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # stateless

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {target_col(h): df["ref_wind"].to_numpy(dtype=np.float64)
                for h in self.horizons_h}


class RidgeIntensity(BaselineModel):
    """Ridge regression on the flattened L=8 causal feature window, per horizon."""

    task = "intensity"

    def __init__(self, horizons_h=HORIZONS_H, alpha: float = 1.0, random_state: int = 42):
        super().__init__(name="intensity_ridge_v1")
        self.horizons_h = horizons_h
        self.alpha = alpha
        self.random_state = random_state
        self._models: dict[int, Ridge] = {}
        self._col_medians: pd.Series | None = None

    def _prep_X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLS].copy()
        # Ridge cannot handle NaN (early-storm tendency features can be NaN
        # only in principle -- construction requires a full L=8 window, so
        # in practice none are NaN here; median-impute defensively, fit on
        # TRAIN medians only per the leakage rule in PROJECT_REQUIREMENTS
        # §4.1: "fit all scalers/encoders on the training split only.")
        if self._col_medians is not None:
            X = X.fillna(self._col_medians)
        return X

    def fit(self, train_df: pd.DataFrame) -> None:
        self._col_medians = train_df[FEATURE_COLS].median()
        X = self._prep_X(train_df)
        for h in self.horizons_h:
            y = train_df[target_col(h)].to_numpy()
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            self._models[h] = model

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._prep_X(df)
        return {target_col(h): self._models[h].predict(X) for h in self.horizons_h}


class LightGBMIntensity(BaselineModel):
    """Gradient-boosted trees on the flattened L=8 causal feature window."""

    task = "intensity"

    def __init__(self, horizons_h=HORIZONS_H, random_state: int = 42, n_estimators: int = 300):
        super().__init__(name="intensity_lightgbm_v1")
        self.horizons_h = horizons_h
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._models: dict[int, LGBMRegressor] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        X = train_df[FEATURE_COLS]
        for h in self.horizons_h:
            y = train_df[target_col(h)].to_numpy()
            model = LGBMRegressor(
                n_estimators=self.n_estimators, learning_rate=0.05,
                num_leaves=15, min_child_samples=10,
                random_state=self.random_state, verbosity=-1,
            )
            model.fit(X, y)
            self._models[h] = model

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = df[FEATURE_COLS]
        return {target_col(h): self._models[h].predict(X) for h in self.horizons_h}
