"""Phase 7 dedicated leakage/scientific-validation tests.

Follows the SAME adversarial philosophy already established in
`ml/tests/test_leakage.py` (Phase 2) rather than inventing a new one: build
normally, mutate the future, assert the output is unaffected -- and include
at least one test that proves a DELIBERATELY leaky construction WOULD be
caught (a "sanity-check the test itself" case), so these tests are not
vacuously passing.

Covers every vector the Phase 7 task names:
  1. No future target leakage
  2. No future track information
  3. No cross-storm contamination
  4. No train/test storm overlap
  5. No validation/test contamination
  6. No feature construction using future observations
  7. No target-derived feature leakage
  8. Correct horizon alignment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.config import zone
from ml.geostrom_ml.features.engineering import HORIZONS_H, L_STEPS, PER_TIMESTEP_FEATURES
from ml.geostrom_ml.models.intensity_gru import (
    FEATURE_COLS, delta_target_col, reshape_to_sequence, target_col,
)
from ml.geostrom_ml.splits.split import load_split_manifest, storm_to_split_map, validate_split_integrity


class TestVector7NoTargetDerivedFeatureLeakage:
    """7. No target-derived feature leakage -- the GRU's input feature
    matrix must never contain a `y_*` (target) or `*_future_*` column."""

    def test_feature_cols_contains_no_target_columns(self):
        assert not any(c.startswith("y_") for c in FEATURE_COLS)
        assert not any("future" in c for c in FEATURE_COLS)

    def test_adversarial_leaky_feature_list_is_detected(self):
        """Sanity-check the test methodology: a deliberately-leaky feature
        list (target column smuggled in) DOES trip the assertion above --
        proving it is not vacuous."""
        leaky_cols = FEATURE_COLS + [target_col(24)]
        assert any(c.startswith("y_") for c in leaky_cols)  # the check WOULD catch this


class TestVector1And2And6NoFutureInformation:
    """1. No future target leakage. 2. No future track information.
    6. No feature construction using future observations."""

    def _windows(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        data = {"sid": ["S0"] * n, "ref_wind": rng.uniform(20, 100, n)}
        for col in FEATURE_COLS:
            data[col] = rng.normal(size=n)
        for h in HORIZONS_H:
            data[target_col(h)] = rng.uniform(20, 100, n)
            data[delta_target_col(h)] = rng.uniform(-20, 20, n)
        return pd.DataFrame(data)

    def test_mutating_target_columns_does_not_change_the_reshaped_features(self):
        """The GRU's input tensor is built purely from FEATURE_COLS; mutating
        every y_* (future target) column must leave reshape_to_sequence()'s
        output completely unchanged."""
        df = self._windows()
        seq_before = reshape_to_sequence(df, L=L_STEPS)

        mutated = df.copy()
        for h in HORIZONS_H:
            mutated[target_col(h)] = 99999.0
            mutated[delta_target_col(h)] = 99999.0
        seq_after = reshape_to_sequence(mutated, L=L_STEPS)

        np.testing.assert_array_equal(seq_before, seq_after)

    def test_a_deliberately_leaky_reshape_would_be_caught(self):
        """Sanity-check the test itself: if a (hypothetical, deliberately
        broken) reshape function DID read a target column, mutating that
        column WOULD change its output -- proving the equality check above
        is a real, non-vacuous test."""
        df = self._windows()

        def leaky_reshape(frame):
            base = reshape_to_sequence(frame, L=L_STEPS)
            base[:, 0, 0] = frame[target_col(24)].to_numpy()  # deliberately leaky
            return base

        before = leaky_reshape(df)
        mutated = df.copy()
        mutated[target_col(24)] = 12345.0
        after = leaky_reshape(mutated)
        assert not np.array_equal(before, after)  # the leak IS detectable


class TestVector3NoCrossStormContamination:
    """3. No cross-storm contamination -- reshape_to_sequence() must never
    mix feature values across different storms' rows."""

    def test_reshape_is_purely_row_wise(self):
        """Two storms' windows, interleaved; each row's reshaped sequence
        must depend only on that row's own columns, never a neighbour's."""
        n = 10
        data = {"sid": (["A", "B"] * (n // 2)), "ref_wind": np.arange(n, dtype=float)}
        for col in FEATURE_COLS:
            data[col] = np.arange(n, dtype=float)  # row i's value == i, everywhere
        df = pd.DataFrame(data)
        seq = reshape_to_sequence(df, L=L_STEPS)
        for i in range(n):
            # every feature/lag for row i must equal i (its own row index),
            # never a neighbouring row's value (which would indicate mixing)
            assert np.all(seq[i] == float(i))


class TestVector8CorrectHorizonAlignment:
    """8. Correct horizon alignment -- y_wind_abs_{h}h must equal
    ref_wind + y_wind_delta_{h}h by construction (engineering.py's own
    invariant); re-verified here against the REAL materialised dataset,
    not just synthetic data, since this is exactly the identity the GRU's
    delta-reconstruction (`predict()`) relies on."""

    def test_real_dataset_delta_plus_ref_equals_absolute(self):
        data_dir = zone("datasets", "v1")
        path = data_dir / "test.parquet"
        if not path.exists():
            pytest.skip("Phase 2 materialised dataset not built on this machine")
        df = pd.read_parquet(path, columns=["ref_wind"] +
                             [target_col(h) for h in HORIZONS_H] +
                             [delta_target_col(h) for h in HORIZONS_H])
        for h in HORIZONS_H:
            reconstructed = df["ref_wind"] + df[delta_target_col(h)]
            np.testing.assert_allclose(reconstructed, df[target_col(h)], atol=1e-6)


class TestVector4And5NoSplitOverlapOrContamination:
    """4. No train/test storm overlap. 5. No validation/test contamination.

    Reuses the existing, already-adversarially-tested
    `validate_split_integrity` (ml/tests/test_splits.py already proves it
    raises on a synthetic overlap) rather than re-implementing an equivalent
    check -- re-verified here against the REAL materialised train/val/test
    parquet files (not just the manifest), since that is what the GRU
    training script actually loads.
    """

    def test_materialised_parquet_storms_match_frozen_split_disjointly(self):
        data_dir = zone("datasets", "v1")
        paths = {s: data_dir / f"{s}.parquet" for s in ("train", "val", "test")}
        if not all(p.exists() for p in paths.values()):
            pytest.skip("Phase 2 materialised dataset not built on this machine")

        manifest = load_split_manifest()
        sid_to_split = storm_to_split_map(manifest)

        storm_sets = {}
        for split_name, path in paths.items():
            df = pd.read_parquet(path, columns=["sid"])
            storm_sets[split_name] = set(df["sid"].unique())
            # every storm in this file must map to THIS split in the frozen manifest
            wrong = {sid for sid in storm_sets[split_name] if sid_to_split.get(sid) != split_name}
            assert not wrong, f"{split_name}.parquet contains storms not assigned to it: {wrong}"

        assert storm_sets["train"].isdisjoint(storm_sets["val"])
        assert storm_sets["train"].isdisjoint(storm_sets["test"])
        assert storm_sets["val"].isdisjoint(storm_sets["test"])

    def test_adversarial_overlap_is_caught_by_the_reused_validator(self):
        """Deliberately construct an overlapping manifest and confirm the
        (reused, not reimplemented) validator raises."""
        bad = {
            "train": {"storm_ids": ["2001213N15040"]},
            "val": {"storm_ids": ["2001213N15040"]},  # same storm in both
            "test": {"storm_ids": ["2002100N10100"]},
        }
        with pytest.raises(ValueError, match="Split integrity violated"):
            validate_split_integrity(bad)
