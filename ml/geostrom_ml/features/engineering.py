"""Causal per-timestep feature engineering and sliding-window construction.

Every feature computed here uses ONLY the current row and rows strictly
BEFORE it, within the same storm. This module is the single place causality
is enforced; ml/tests/test_leakage.py asserts this holds by construction and
by a randomised probe (shuffling future rows must not change any feature
value at time t).

Temporal window per feature, explicit as required by the Phase 2 task brief:

    at row t (t = a specific synoptic observation of one storm):
        lag_k (k=1..8)   uses the state at t - 6*k hours   (t-6h .. t-48h)
        tendency_Xh      uses state at t and t-Xh           (X in {6,12,24})
        max_wind_so_far  uses rows [storm_start .. t] inclusive
        hours_since_genesis, doy_sin/cos, abs_lat, dist2land  use row t only
        motion (speed/bearing) uses positions at t-6h and t

    NOTHING here ever reads a row at t+k for any k > 0.

Sequence windows (docs/DATA_STRATEGY.md §5.4, ML_ARCHITECTURE.md §6.1/§7.1):
    input window L=8 steps (t-42h .. t, i.e. 8 consecutive synoptic obs
    ending at the reference time t) -> horizon H=4 steps (t+6h, t+12h,
    t+18h, t+24h). A window is emitted ONLY if all L+H=12 steps exist at an
    exact, contiguous 6-hour cadence within one storm. No interpolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.data.ibtracs import WIND_COLUMN, PRESSURE_COLUMN  # noqa: E402
from ml.geostrom_ml.features.geo import (  # noqa: E402
    haversine_km, initial_bearing_deg, wrap_lon_diff,
)

STEP_HOURS = 6
L_STEPS = 8          # 48 h of input history
H_STEPS_ALL = 4       # +6,+12,+18,+24 h horizon steps
HORIZONS_H = (6, 12, 18, 24)   # matches ML_ARCHITECTURE.md §6.3/§7.1
HEADLINE_HORIZON_H = 24        # the MVP-required evaluation horizon

# Per-timestep engineered feature names, in a fixed order (used for flattening)
PER_TIMESTEP_FEATURES = [
    "lat", "abs_lat", "lon_sin", "lon_cos",
    WIND_COLUMN, PRESSURE_COLUMN,
    "storm_speed_kt", "storm_dir_sin", "storm_dir_cos",
    "d_wind_6h", "d_wind_12h", "d_wind_24h",
    "d_pres_6h", "d_pres_12h", "d_pres_24h",
    "max_wind_so_far", "hours_since_genesis",
    "doy_sin", "doy_cos", "dist2land",
]


def build_per_timestep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the causal per-timestep feature table for a filtered IBTrACS df.

    `df` must already be `filter_usable_rows()` output: one row per
    (storm, synoptic timestamp), sorted by (SID, ISO_TIME). This function
    adds engineered columns and does not drop or reorder rows.
    """
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    g = df.groupby("SID", sort=False)

    out = df.copy()
    out["abs_lat"] = out["LAT"].abs()
    out["lon_sin"] = np.sin(np.radians(out["LON"]))
    out["lon_cos"] = np.cos(np.radians(out["LON"]))

    # --- motion features: use ONLY t-6h -> t (one step back) --------------
    prev_lat = g["LAT"].shift(1)
    prev_lon = g["LON"].shift(1)
    prev_time = g["ISO_TIME"].shift(1)
    gap_hours = (out["ISO_TIME"] - prev_time).dt.total_seconds() / 3600.0
    one_step_back = np.isclose(gap_hours, STEP_HOURS)

    dist_km = haversine_km(prev_lat.to_numpy(), prev_lon.to_numpy(),
                           out["LAT"].to_numpy(), out["LON"].to_numpy())
    bearing = initial_bearing_deg(prev_lat.to_numpy(), prev_lon.to_numpy(),
                                  out["LAT"].to_numpy(), out["LON"].to_numpy())
    speed_kt = (dist_km / STEP_HOURS) * 0.539957  # km/h -> knots
    speed_kt = np.where(one_step_back, speed_kt, np.nan)
    bearing = np.where(one_step_back, bearing, np.nan)

    out["storm_speed_kt"] = speed_kt
    out["storm_dir_sin"] = np.sin(np.radians(bearing))
    out["storm_dir_cos"] = np.cos(np.radians(bearing))

    # --- intensity tendencies: exact-lag-only, causal ----------------------
    for hours in (6, 12, 24):
        steps = hours // STEP_HOURS
        prev_wind = g[WIND_COLUMN].shift(steps)
        prev_pres = g[PRESSURE_COLUMN].shift(steps)
        prev_time_k = g["ISO_TIME"].shift(steps)
        gap_k = (out["ISO_TIME"] - prev_time_k).dt.total_seconds() / 3600.0
        exact = np.isclose(gap_k, hours)
        out[f"d_wind_{hours}h"] = np.where(exact, out[WIND_COLUMN] - prev_wind, np.nan)
        out[f"d_pres_{hours}h"] = np.where(exact, out[PRESSURE_COLUMN] - prev_pres, np.nan)

    # --- max wind SO FAR: expanding max over [genesis .. t] inclusive -----
    # This is a running/expanding aggregate, NEVER the lifetime max. It is
    # recomputed independently for every row using only that row and earlier
    # rows of the same storm. Explicitly NOT the "max_wind" summary field
    # the Phase 2 brief prohibits (that field is a whole-storm lifetime
    # aggregate computed once per storm, which leaks the future into early
    # timesteps -- this is the opposite: it grows monotonically with t).
    out["max_wind_so_far"] = g[WIND_COLUMN].cummax()

    # --- storm age / seasonality: single-row, no lookback needed ----------
    genesis = g["ISO_TIME"].transform("min")
    out["hours_since_genesis"] = (out["ISO_TIME"] - genesis).dt.total_seconds() / 3600.0
    doy = out["ISO_TIME"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out = out.rename(columns={"LAT": "lat", "DIST2LAND": "dist2land"})
    return out


def _contiguous_run_id(sid_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Assign a run id that increments whenever the 6h cadence breaks."""
    gap_hours = time_series.diff().dt.total_seconds() / 3600.0
    same_storm = sid_series == sid_series.shift(1)
    is_contiguous = same_storm & np.isclose(gap_hours, STEP_HOURS)
    return (~is_contiguous).cumsum()


def build_sequence_windows(
    feat_df: pd.DataFrame,
    L: int = L_STEPS,
    horizons_h: tuple[int, ...] = HORIZONS_H,
) -> pd.DataFrame:
    """Build L-step-input / multi-horizon-target sliding windows.

    One output row per valid window. Columns:
      sid, t_ref (the reference/last-input timestep), season
      x__<feature>__lag<k>   for k=0..L-1 (lag0 = t_ref, lag(L-1) = oldest)
      y_wind_abs_{h}h, y_wind_delta_{h}h     for h in horizons_h
      y_dlat_{h}h, y_dlon_{h}h               wrap-safe displacement targets
      y_lat_future_{h}h, y_lon_future_{h}h   absolute future position (eval only)

    A window is emitted only if all L input steps AND all requested horizon
    steps exist at an exact contiguous 6-hour cadence within one storm --
    verified via `_contiguous_run_id`, not assumed. No gap is ever bridged
    by interpolation.
    """
    df = feat_df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    run_id = _contiguous_run_id(df["SID"], df["ISO_TIME"])
    df = df.assign(_run=run_id)

    max_h_steps = max(horizons_h) // STEP_HOURS
    rows = []

    for _, run_df in df.groupby(["SID", "_run"], sort=False):
        run_df = run_df.reset_index(drop=True)
        n = len(run_df)
        needed = L + max_h_steps
        if n < needed:
            continue
        for start in range(0, n - needed + 1):
            in_end = start + L               # exclusive; ref row = in_end-1
            ref_idx = in_end - 1
            window = run_df.iloc[start:in_end]
            ref = run_df.iloc[ref_idx]

            row = {
                "sid": ref["SID"], "t_ref": ref["ISO_TIME"], "season": ref["SEASON"],
                "ref_lat": ref["lat"], "ref_lon": ref["LON"],
                "ref_wind": ref[WIND_COLUMN], "ref_pres": ref[PRESSURE_COLUMN],
            }
            # lag0 = most recent (t_ref) ... lag(L-1) = oldest (t_ref-42h)
            for k in range(L):
                src = window.iloc[L - 1 - k]
                for feat in PER_TIMESTEP_FEATURES:
                    row[f"x__{feat}__lag{k}"] = src[feat]

            ok = True
            for h in horizons_h:
                h_steps = h // STEP_HOURS
                tgt_idx = ref_idx + h_steps
                if tgt_idx >= n:
                    ok = False
                    break
                tgt = run_df.iloc[tgt_idx]
                row[f"y_wind_abs_{h}h"] = tgt[WIND_COLUMN]
                row[f"y_wind_delta_{h}h"] = tgt[WIND_COLUMN] - ref[WIND_COLUMN]
                row[f"y_lat_future_{h}h"] = tgt["lat"]
                row[f"y_lon_future_{h}h"] = tgt["LON"]
                row[f"y_dlat_{h}h"] = tgt["lat"] - ref["lat"]
                row[f"y_dlon_{h}h"] = wrap_lon_diff(ref["LON"], tgt["LON"])
            if not ok:
                continue
            rows.append(row)

    return pd.DataFrame(rows)


def flattened_feature_columns(L: int = L_STEPS) -> list[str]:
    """Column names of the flattened per-timestep block, in a fixed order."""
    return [f"x__{feat}__lag{k}" for k in range(L) for feat in PER_TIMESTEP_FEATURES]
