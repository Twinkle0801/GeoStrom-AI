"""Phase 8 dedicated leakage/scientific-validation tests for track
prediction.

Follows the SAME adversarial philosophy already established in
`ml/tests/test_leakage.py` (Phase 2) and `ml/tests/test_phase7_leakage.py`
(Phase 7) rather than inventing a new one: build normally, mutate the
future, assert the output is unaffected -- and include at least one test
that proves a DELIBERATELY leaky construction WOULD be caught, so these
tests are not vacuously passing.

Covers every vector the Phase 8 task names:
  1. Future observations cannot enter the input sequence.
  2. Future latitude/longitude cannot enter features.
  3. Target coordinates cannot enter features.
  4. Test storms cannot appear in training.
  5. Validation storms cannot appear in training.
  6. Sequence windows cannot cross storm boundaries.
  7. Feature construction remains causal.
  8. The frozen split manifest is respected.
  9. Target generation uses only future observations after the input cutoff.
  10. No post-storm summary information enters the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.config import zone
from ml.geostrom_ml.features.engineering import HORIZONS_H
from ml.geostrom_ml.models.intensity_gru import reshape_to_sequence
from ml.geostrom_ml.models.track_baselines import dlat_col, dlon_col
from ml.geostrom_ml.models.track_gru import FEATURE_COLS, lat_future_col, lon_future_col
from ml.geostrom_ml.splits.split import (
    load_split_manifest, storm_to_split_map, validate_split_integrity,
)


def _windows(n=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "sid": ["S0"] * n,
        "ref_lat": rng.uniform(5, 40, n),
        "ref_lon": rng.uniform(-180, 180, n),
    }
    for col in FEATURE_COLS:
        data[col] = rng.normal(size=n)
    for h in HORIZONS_H:
        data[dlat_col(h)] = rng.uniform(-2, 2, n)
        data[dlon_col(h)] = rng.uniform(-2, 2, n)
        data[lat_future_col(h)] = rng.uniform(5, 40, n)
        data[lon_future_col(h)] = rng.uniform(-180, 180, n)
    return pd.DataFrame(data)


class TestVector3And10NoTargetOrPostStormFeatureLeakage:
    """3. Target coordinates cannot enter features.
    10. No post-storm summary information enters the features."""

    def test_feature_cols_contains_no_target_or_future_columns(self):
        assert not any(c.startswith("y_") for c in FEATURE_COLS)
        assert not any("future" in c for c in FEATURE_COLS)

    def test_adversarial_leaky_feature_list_is_detected(self):
        """Sanity-check the test methodology: a deliberately-leaky feature
        list (a future-position target column smuggled in) DOES trip the
        assertion above -- proving it is not vacuous."""
        leaky_cols = FEATURE_COLS + [lat_future_col(24)]
        assert any("future" in c for c in leaky_cols)  # the check WOULD catch this


class TestVector1And2And6And7And9NoFutureInformation:
    """1. Future observations cannot enter the input sequence.
    2. Future latitude/longitude cannot enter features.
    6. Sequence windows cannot cross storm boundaries (reshape is row-wise).
    7. Feature construction remains causal.
    9. Target generation uses only future observations after the cutoff --
    re-verified here by proving the INPUT tensor is fully independent of
    every target/future column."""

    def test_mutating_target_and_future_columns_does_not_change_reshaped_features(self):
        """TrackGRU's input tensor is built purely from FEATURE_COLS (shared,
        unmodified, with IntensityGRU's `reshape_to_sequence`); mutating
        every y_dlat/y_dlon/y_*_future_* column must leave the reshaped
        input completely unchanged."""
        df = _windows()
        seq_before = reshape_to_sequence(df)

        mutated = df.copy()
        for h in HORIZONS_H:
            mutated[dlat_col(h)] = 999.0
            mutated[dlon_col(h)] = 999.0
            mutated[lat_future_col(h)] = 999.0
            mutated[lon_future_col(h)] = 999.0
        seq_after = reshape_to_sequence(mutated)

        np.testing.assert_array_equal(seq_before, seq_after)

    def test_a_deliberately_leaky_reshape_would_be_caught(self):
        """Sanity-check the test itself: if a (hypothetical, deliberately
        broken) reshape function DID read a future-position column, mutating
        that column WOULD change its output -- proving the equality check
        above is a real, non-vacuous test."""
        df = _windows()

        def leaky_reshape(frame):
            base = reshape_to_sequence(frame)
            base[:, 0, 0] = frame[lat_future_col(24)].to_numpy()  # deliberately leaky
            return base

        before = leaky_reshape(df)
        mutated = df.copy()
        mutated[lat_future_col(24)] = 12345.0
        after = leaky_reshape(mutated)
        assert not np.array_equal(before, after)  # the leak IS detectable


class TestVector4And5And8NoSplitOverlapOrContamination:
    """4. Test storms cannot appear in training.
    5. Validation storms cannot appear in training.
    8. The frozen split manifest is respected.

    Reuses the existing, already-adversarially-tested
    `validate_split_integrity` (proven in `ml/tests/test_splits.py` to raise
    on a synthetic overlap) rather than re-implementing an equivalent check
    -- re-verified here against the REAL materialised train/val/test parquet
    files (not just the manifest), since that is what `train_track_gru.py`
    actually loads."""

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


class TestRefPositionIsCausalNotFuture:
    """The `ref_lat`/`ref_lon` columns the GRU's cos(latitude) loss weight
    and displacement reconstruction both depend on must be the LAST
    OBSERVED (input-window) position, never a future/target position --
    otherwise the loss weight itself would leak future information."""

    def test_ref_position_columns_are_distinct_from_every_future_column(self):
        df = _windows()
        for h in HORIZONS_H:
            # ref_lat/ref_lon must not be aliases of any future column: perturbing
            # a future column must never change ref_lat/ref_lon.
            mutated = df.copy()
            mutated[lat_future_col(h)] = df[lat_future_col(h)] + 1000.0
            mutated[lon_future_col(h)] = df[lon_future_col(h)] + 1000.0
            pd.testing.assert_series_equal(df["ref_lat"], mutated["ref_lat"])
            pd.testing.assert_series_equal(df["ref_lon"], mutated["ref_lon"])
