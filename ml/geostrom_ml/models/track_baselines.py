"""Track baselines: Persistence (constant-velocity), CLIPER-style, LightGBM.

All three predict displacement (y_dlat_{h}h, y_dlon_{h}h) at each horizon,
per the Phase 0 locked decision (docs/PROJECT_REQUIREMENTS.md §2.D): models
never predict absolute coordinates directly. Absolute future position is
reconstructed for evaluation via `ml.geostrom_ml.features.geo.displace`.

ML_ARCHITECTURE.md §7.4 Tier 1:
  (a) Persistence: constant velocity from the last two positions.
  (b) CLIPER-style: linear/ridge regression of displacement on current
      position, motion, intensity, day-of-year, and their interactions.
  (c) LightGBM per output.
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
from ml.geostrom_ml.features.geo import destination_point, wrap_lon_diff  # noqa: E402
from ml.geostrom_ml.models.base import BaselineModel  # noqa: E402

FEATURE_COLS = flattened_feature_columns()


def dlat_col(h: int) -> str:
    return f"y_dlat_{h}h"


def dlon_col(h: int) -> str:
    return f"y_dlon_{h}h"


class PersistenceTrack(BaselineModel):
    """Constant-velocity extrapolation of the last observed motion vector.

    Uses the causal storm_speed_kt / storm_dir(sin,cos) at lag0 -- computed
    in build_per_timestep_features from (t-6h -> t) only -- to project the
    same speed and bearing forward by h hours from the reference position.
    """

    task = "track"

    def __init__(self, horizons_h=HORIZONS_H):
        super().__init__(name="track_persistence_v1")
        self.horizons_h = horizons_h

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # stateless

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        speed_kt = df["x__storm_speed_kt__lag0"].to_numpy(dtype=np.float64)
        dir_sin = df["x__storm_dir_sin__lag0"].to_numpy(dtype=np.float64)
        dir_cos = df["x__storm_dir_cos__lag0"].to_numpy(dtype=np.float64)
        bearing = np.degrees(np.arctan2(dir_sin, dir_cos)) % 360.0
        speed_kmh = speed_kt / 0.539957

        ref_lat = df["ref_lat"].to_numpy(dtype=np.float64)
        ref_lon = df["ref_lon"].to_numpy(dtype=np.float64)

        # A storm with no prior motion (speed=0/NaN, i.e. genesis-adjacent
        # rows that still cleared the L=8 window requirement) persists at
        # zero displacement rather than propagating a NaN prediction.
        speed_kmh = np.nan_to_num(speed_kmh, nan=0.0)
        bearing = np.nan_to_num(bearing, nan=0.0)

        out = {}
        for h in self.horizons_h:
            dist_km = speed_kmh * h
            fut_lat, fut_lon = destination_point(ref_lat, ref_lon, bearing, dist_km)
            out[dlat_col(h)] = fut_lat - ref_lat
            out[dlon_col(h)] = wrap_lon_diff(ref_lon, fut_lon)
        return out


# Compact CLIPER-style feature set: current position, motion, intensity,
# day-of-year, per ML_ARCHITECTURE.md §7.4 -- deliberately NOT the full
# flattened 160-column window (that differentiates it from the LightGBM
# baseline below, matching the architecture's tiered baseline design).
_CLIPER_BASE = [
    "x__lat__lag0", "x__abs_lat__lag0", "x__lon_sin__lag0", "x__lon_cos__lag0",
    "x__USA_WIND__lag0", "x__USA_PRES__lag0",
    "x__storm_speed_kt__lag0", "x__storm_dir_sin__lag0", "x__storm_dir_cos__lag0",
    "x__doy_sin__lag0", "x__doy_cos__lag0", "x__hours_since_genesis__lag0",
]


def _cliper_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[_CLIPER_BASE].copy()
    # Explicit interaction terms: position x seasonality, motion x motion.
    X["ix_lat_doy_sin"] = df["x__lat__lag0"] * df["x__doy_sin__lag0"]
    X["ix_lat_doy_cos"] = df["x__lat__lag0"] * df["x__doy_cos__lag0"]
    X["ix_speed_dirsin"] = df["x__storm_speed_kt__lag0"] * df["x__storm_dir_sin__lag0"]
    X["ix_speed_dircos"] = df["x__storm_speed_kt__lag0"] * df["x__storm_dir_cos__lag0"]
    X["ix_wind_abslat"] = df["x__USA_WIND__lag0"] * df["x__abs_lat__lag0"]
    return X


class CliperTrack(BaselineModel):
    """Ridge regression of displacement on a compact CLIPER-style feature set."""

    task = "track"

    def __init__(self, horizons_h=HORIZONS_H, alpha: float = 1.0, random_state: int = 42):
        super().__init__(name="track_cliper_v1")
        self.horizons_h = horizons_h
        self.alpha = alpha
        self.random_state = random_state
        self._lat_models: dict[int, Ridge] = {}
        self._lon_models: dict[int, Ridge] = {}
        self._col_medians: pd.Series | None = None

    def _prep_X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _cliper_design_matrix(df)
        if self._col_medians is not None:
            X = X.fillna(self._col_medians)
        return X

    def fit(self, train_df: pd.DataFrame) -> None:
        X_raw = _cliper_design_matrix(train_df)
        self._col_medians = X_raw.median()
        X = self._prep_X(train_df)
        for h in self.horizons_h:
            lat_model = Ridge(alpha=self.alpha, random_state=self.random_state)
            lat_model.fit(X, train_df[dlat_col(h)].to_numpy())
            self._lat_models[h] = lat_model

            lon_model = Ridge(alpha=self.alpha, random_state=self.random_state)
            lon_model.fit(X, train_df[dlon_col(h)].to_numpy())
            self._lon_models[h] = lon_model

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._prep_X(df)
        out = {}
        for h in self.horizons_h:
            out[dlat_col(h)] = self._lat_models[h].predict(X)
            out[dlon_col(h)] = self._lon_models[h].predict(X)
        return out


class LightGBMTrack(BaselineModel):
    """Gradient-boosted trees on the flattened L=8 causal feature window."""

    task = "track"

    def __init__(self, horizons_h=HORIZONS_H, random_state: int = 42, n_estimators: int = 300):
        super().__init__(name="track_lightgbm_v1")
        self.horizons_h = horizons_h
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._lat_models: dict[int, LGBMRegressor] = {}
        self._lon_models: dict[int, LGBMRegressor] = {}

    def _new_model(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=0.05,
            num_leaves=15, min_child_samples=10,
            random_state=self.random_state, verbosity=-1,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        X = train_df[FEATURE_COLS]
        for h in self.horizons_h:
            lat_model = self._new_model()
            lat_model.fit(X, train_df[dlat_col(h)].to_numpy())
            self._lat_models[h] = lat_model

            lon_model = self._new_model()
            lon_model.fit(X, train_df[dlon_col(h)].to_numpy())
            self._lon_models[h] = lon_model

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = df[FEATURE_COLS]
        out = {}
        for h in self.horizons_h:
            out[dlat_col(h)] = self._lat_models[h].predict(X)
            out[dlon_col(h)] = self._lon_models[h].predict(X)
        return out
