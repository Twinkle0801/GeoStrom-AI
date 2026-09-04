"""Tests for causal feature engineering and sequence-window construction.

These are the tests that would fail if future information ever entered the
feature set, per the Phase 2 task brief's explicit requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.features.engineering import (
    H_STEPS_ALL, HORIZONS_H, L_STEPS, PER_TIMESTEP_FEATURES, STEP_HOURS,
    build_per_timestep_features, build_sequence_windows, flattened_feature_columns,
)


class TestPerTimestepFeatures:
    def test_row_count_preserved(self, synthetic_storm_df):
        feat = build_per_timestep_features(synthetic_storm_df)
        assert len(feat) == len(synthetic_storm_df)

    def test_first_row_has_no_motion_or_tendency(self, synthetic_storm_df):
        """The storm's genesis row has no prior observation, so every
        backward-looking feature must be NaN, not silently zero or copied
        from elsewhere."""
        feat = build_per_timestep_features(synthetic_storm_df)
        first = feat.iloc[0]
        assert pd.isna(first["storm_speed_kt"])
        assert pd.isna(first["d_wind_6h"])
        assert pd.isna(first["d_wind_12h"])
        assert pd.isna(first["d_wind_24h"])
        assert first["hours_since_genesis"] == 0.0
        assert first["max_wind_so_far"] == first["USA_WIND"]

    def test_max_wind_so_far_is_expanding_not_lifetime(self, synthetic_storm_df):
        """max_wind_so_far at an early row must NOT equal the storm's
        eventual lifetime maximum (that would be the prohibited
        future-leaking lifetime aggregate)."""
        feat = build_per_timestep_features(synthetic_storm_df)
        lifetime_max = feat["USA_WIND"].max()
        early_row = feat.iloc[2]  # still early in the intensifying phase
        assert early_row["max_wind_so_far"] < lifetime_max
        assert early_row["max_wind_so_far"] == feat["USA_WIND"].iloc[:3].max()

    def test_max_wind_so_far_is_monotonic_nondecreasing(self, synthetic_storm_df):
        feat = build_per_timestep_features(synthetic_storm_df)
        vals = feat["max_wind_so_far"].to_numpy()
        assert np.all(np.diff(vals) >= 0)

    def test_causality_future_mutation_does_not_change_past_features(self, synthetic_storm_df):
        """THE core leakage regression test: corrupting rows strictly after
        time t must not change ANY engineered feature value AT time t."""
        original = build_per_timestep_features(synthetic_storm_df)

        mutated_input = synthetic_storm_df.copy()
        future_mask = mutated_input.index >= 10
        mutated_input.loc[future_mask, "USA_WIND"] = 9999.0
        mutated_input.loc[future_mask, "USA_PRES"] = 1.0
        mutated_input.loc[future_mask, "LAT"] = -89.0
        mutated_input.loc[future_mask, "LON"] = 179.0
        mutated = build_per_timestep_features(mutated_input)

        # every row BEFORE the mutated region must be byte-identical
        past_cols = [c for c in PER_TIMESTEP_FEATURES if c in original.columns]
        pd.testing.assert_frame_equal(
            original.loc[:9, past_cols].reset_index(drop=True),
            mutated.loc[:9, past_cols].reset_index(drop=True),
        )

    def test_tendency_requires_exact_lag_else_nan(self, storm_with_gap):
        """If the row 6h earlier doesn't exist (a gap), the tendency feature
        must be NaN, never silently computed from a farther/wrong row."""
        feat = build_per_timestep_features(storm_with_gap)
        # row that immediately follows the gap: its "previous" row is 12h
        # back, not 6h, so d_wind_6h must be NaN there.
        gap_time = storm_with_gap["ISO_TIME"].iloc[5] + pd.Timedelta(hours=12)
        row = feat[feat["ISO_TIME"] == gap_time].iloc[0]
        assert pd.isna(row["d_wind_6h"])


class TestSequenceWindows:
    def test_window_count_matches_hand_calculation(self, synthetic_storm_df):
        # 16 rows, need L + H_STEPS_ALL = 8 + 4 = 12 contiguous rows per
        # window => 16 - 12 + 1 = 5 windows
        feat = build_per_timestep_features(synthetic_storm_df)
        windows = build_sequence_windows(feat, L=L_STEPS, horizons_h=HORIZONS_H)
        assert len(windows) == 16 - (L_STEPS + H_STEPS_ALL) + 1

    def test_no_windows_span_storm_boundary(self, two_synthetic_storms):
        feat = build_per_timestep_features(two_synthetic_storms)
        windows = build_sequence_windows(feat)
        assert set(windows["sid"].unique()) <= set(two_synthetic_storms["SID"].unique())
        # every window's flattened lag times must belong to exactly one SID
        # (guaranteed by construction: groupby(["SID","_run"]) before
        # windowing) -- spot-check via t_ref falling within that storm's span
        for sid, grp in windows.groupby("sid"):
            storm_span = two_synthetic_storms[two_synthetic_storms["SID"] == sid]["ISO_TIME"]
            assert grp["t_ref"].between(storm_span.min(), storm_span.max()).all()

    def test_gap_breaks_window_eligibility(self, storm_with_gap, synthetic_storm_df):
        """Removing one mid-storm observation must strictly reduce the
        window count relative to the ungapped storm (no interpolation
        bridges the gap)."""
        feat_full = build_per_timestep_features(synthetic_storm_df)
        feat_gap = build_per_timestep_features(storm_with_gap)
        w_full = build_sequence_windows(feat_full)
        w_gap = build_sequence_windows(feat_gap)
        assert len(w_gap) < len(w_full)

    def test_targets_are_strictly_future_of_reference(self, synthetic_storm_df):
        feat = build_per_timestep_features(synthetic_storm_df)
        windows = build_sequence_windows(feat)
        for h in HORIZONS_H:
            # the future position must differ from the reference position
            # for a moving storm, and must correspond to t_ref + h hours
            implied_time = windows["t_ref"] + pd.Timedelta(hours=h)
            # reconstruct actual time of the target row via original df
            merged = windows.merge(
                synthetic_storm_df[["ISO_TIME", "LAT", "LON"]].rename(
                    columns={"ISO_TIME": "_t", "LAT": "_lat", "LON": "_lon"}),
                left_on=implied_time, right_on="_t", how="left")
            np.testing.assert_allclose(
                merged[f"y_lat_future_{h}h"].to_numpy(),
                merged["_lat"].to_numpy(), atol=1e-9)

    def test_lag0_equals_reference_row_state(self, synthetic_storm_df):
        feat = build_per_timestep_features(synthetic_storm_df)
        windows = build_sequence_windows(feat)
        np.testing.assert_allclose(
            windows["x__USA_WIND__lag0"].to_numpy(), windows["ref_wind"].to_numpy())

    def test_dlon_uses_wrap_safe_difference(self):
        """Construct a synthetic storm crossing the antimeridian and verify
        the window target y_dlon does NOT show a ~358-degree jump."""
        n = 14
        times = pd.date_range("2003-09-01T00:00:00", periods=n, freq="6h")
        lon = 179.0 + 0.3 * np.arange(n)  # drifts past +180, should wrap negative
        lon = ((lon + 180) % 360) - 180
        df = pd.DataFrame({
            "SID": "2003244N10179", "SEASON": 2003, "NUMBER": 1,
            "BASIN": "NA", "SUBBASIN": "MM", "NAME": "X",
            "ISO_TIME": times, "NATURE": "TS",
            "LAT": 10.0 + 0.05 * np.arange(n), "LON": lon,
            "USA_WIND": 40.0, "USA_PRES": 990.0,
            "USA_SSHS": 0, "USA_STATUS": "TS",
            "TRACK_TYPE": "main", "IFLAG": "O_____________",
            "STORM_SPEED": 10.0, "STORM_DIR": 90.0,
            "DIST2LAND": 500.0, "LANDFALL": 999.0,
        })
        feat = build_per_timestep_features(df)
        windows = build_sequence_windows(feat)
        assert len(windows) >= 1
        assert (windows["y_dlon_6h"].abs() < 10).all()   # never near 358


class TestFlattenedColumns:
    def test_column_count(self):
        cols = flattened_feature_columns(L=L_STEPS)
        assert len(cols) == L_STEPS * len(PER_TIMESTEP_FEATURES)

    def test_lag_ordering(self):
        cols = flattened_feature_columns(L=3)
        assert cols[0].endswith("__lag0")
        assert cols[len(PER_TIMESTEP_FEATURES)].endswith("__lag1")
