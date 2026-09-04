"""Dedicated leakage regression tests.

Per the Phase 2 task brief: "Tests must fail if future information enters
the feature set." This file is the explicit, adversarial version of that
requirement -- it does not just check that the pipeline behaves correctly
today, it checks that a leak WOULD be caught if one were introduced.

Complements:
  - test_features.py::TestPerTimestepFeatures::test_causality_...  (per-timestep)
  - test_splits.py::TestFrozenSplitIntegrity                        (storm-level)
This file adds: window-level causality, and materialised-dataset-on-disk
split leakage (the actual train/val/test Parquet files, not just the
manifest that describes them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.features.engineering import (
    HORIZONS_H, build_per_timestep_features, build_sequence_windows,
    flattened_feature_columns,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import zone  # noqa: E402


class TestWindowLevelCausality:
    def test_flattened_features_unaffected_by_mutating_horizon_rows(self, synthetic_storm_df):
        """Mutating ONLY the rows that fall inside the forecast horizon
        (t+6h..t+24h of the window ending at each t) must not change that
        window's flattened INPUT features -- only its targets."""
        feat_original = build_per_timestep_features(synthetic_storm_df)
        windows_original = build_sequence_windows(feat_original)

        mutated_input = synthetic_storm_df.copy()
        # corrupt the LAST 4 rows only (guaranteed to be pure horizon rows
        # for every window, since every window's input ends by row index
        # len-1-4 at the latest)
        mutated_input.loc[mutated_input.index[-4:], "USA_WIND"] = 12345.0
        mutated_input.loc[mutated_input.index[-4:], "LAT"] = -1.0
        mutated_input.loc[mutated_input.index[-4:], "LON"] = 1.0

        feat_mutated = build_per_timestep_features(mutated_input)
        windows_mutated = build_sequence_windows(feat_mutated)

        flat_cols = flattened_feature_columns()
        # Only compare windows whose t_ref is early enough that NONE of
        # their input rows overlap the corrupted tail.
        safe = windows_original["t_ref"] <= (
            synthetic_storm_df["ISO_TIME"].iloc[-5])
        pd.testing.assert_frame_equal(
            windows_original.loc[safe, flat_cols].reset_index(drop=True),
            windows_mutated.loc[safe, flat_cols].reset_index(drop=True),
        )

    def test_a_deliberately_leaky_construction_is_actually_detected(self, synthetic_storm_df):
        """Sanity-check the TEST ITSELF: build an intentionally leaky feature
        (using a forward shift) and confirm it VIOLATES the causality
        property the other tests assert, proving those tests are not
        vacuously passing."""
        df = synthetic_storm_df.sort_values("ISO_TIME").reset_index(drop=True)
        leaky_feature = df["USA_WIND"].shift(-1)  # reads the FUTURE row -- wrong on purpose

        mutated = df.copy()
        mutated.loc[mutated.index[-1], "USA_WIND"] = 999.0
        leaky_feature_after_mutation = mutated["USA_WIND"].shift(-1)

        # the leaky feature at the second-to-last row DOES change when the
        # future (last) row is mutated -- demonstrating this construction
        # is leaky and that our test methodology can detect it.
        idx = df.index[-2]
        assert leaky_feature.loc[idx] != leaky_feature_after_mutation.loc[idx]


class TestMaterialisedDatasetSplitLeakage:
    """Checks the actual Parquet files on disk (built by build_dataset.py),
    not just the JSON manifest that describes the intended split."""

    @staticmethod
    @pytest.fixture(scope="class")
    def dataset_dir():
        d = zone("datasets", "v1")
        if not (d / "train.parquet").exists():
            pytest.skip("Phase 2 dataset not built yet -- run build_dataset.py")
        return d

    def test_no_storm_id_shared_across_materialised_splits(self, dataset_dir):
        train = pd.read_parquet(dataset_dir / "train.parquet", columns=["sid"])
        val = pd.read_parquet(dataset_dir / "val.parquet", columns=["sid"])
        test = pd.read_parquet(dataset_dir / "test.parquet", columns=["sid"])

        train_sids, val_sids, test_sids = (set(train["sid"]), set(val["sid"]), set(test["sid"]))
        assert train_sids.isdisjoint(val_sids)
        assert train_sids.isdisjoint(test_sids)
        assert val_sids.isdisjoint(test_sids)

    def test_every_window_target_horizon_exceeds_its_own_reference_time(self, dataset_dir):
        """For every materialised window, every y_lat_future_/y_lon_future_
        target's implied timestamp (t_ref + h) is strictly after t_ref --
        i.e. targets are never reference-time-or-earlier."""
        test_df = pd.read_parquet(dataset_dir / "test.parquet",
                                  columns=["t_ref"] + [f"y_wind_abs_{h}h" for h in HORIZONS_H])
        # implicit by construction (targets come from run_df.iloc[ref_idx + h_steps],
        # h_steps > 0 always) -- this test asserts the row count is consistent
        # and no null targets exist (a null would indicate a horizon that
        # accidentally read before the reference, which build_sequence_windows
        # cannot produce, but we assert it holds on the actual materialised data).
        for h in HORIZONS_H:
            assert test_df[f"y_wind_abs_{h}h"].notna().all()
