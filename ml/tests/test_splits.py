"""Tests for storm-level split integrity and determinism.

Operates on the real, frozen split (already built by build_splits.py in this
environment) since the split logic reads actual IBTrACS data; this also
means these tests double as an integration check that the frozen split on
disk is still valid.
"""

from __future__ import annotations

import pytest

from ml.geostrom_ml.splits.split import (
    build_split_manifest, load_split_manifest, validate_split_integrity,
)


@pytest.fixture(scope="module")
def manifest():
    return load_split_manifest()


class TestFrozenSplitIntegrity:
    def test_train_val_disjoint(self, manifest):
        train = set(manifest["train"]["storm_ids"])
        val = set(manifest["val"]["storm_ids"])
        assert train.isdisjoint(val)

    def test_train_test_disjoint(self, manifest):
        train = set(manifest["train"]["storm_ids"])
        test = set(manifest["test"]["storm_ids"])
        assert train.isdisjoint(test)

    def test_val_test_disjoint(self, manifest):
        val = set(manifest["val"]["storm_ids"])
        test = set(manifest["test"]["storm_ids"])
        assert val.isdisjoint(test)

    def test_no_storm_appears_more_than_once(self, manifest):
        all_ids = (manifest["train"]["storm_ids"] + manifest["val"]["storm_ids"]
                   + manifest["test"]["storm_ids"])
        assert len(all_ids) == len(set(all_ids))

    def test_integrity_check_field_matches_reality(self, manifest):
        assert manifest["integrity_check"]["all_disjoint"] is True
        assert manifest["integrity_check"]["intersection_train_val"] == []
        assert manifest["integrity_check"]["intersection_train_test"] == []
        assert manifest["integrity_check"]["intersection_val_test"] == []

    def test_sid_season_cross_check_passed(self, manifest):
        assert manifest["sid_season_cross_check"]["n_mismatches"] == 0

    def test_season_blocks_are_temporally_ordered(self, manifest):
        """Train seasons strictly precede val seasons strictly precede test
        seasons -- the season-block methodology, not a random storm split."""
        assert manifest["train"]["seasons"][1] < manifest["val"]["seasons"][0]
        assert manifest["val"]["seasons"][1] < manifest["test"]["seasons"][0]

    def test_nonzero_storms_in_every_split(self, manifest):
        assert manifest["train"]["n_storms"] > 0
        assert manifest["val"]["n_storms"] > 0
        assert manifest["test"]["n_storms"] > 0


class TestDeterminism:
    def test_rebuilding_split_is_byte_identical(self, manifest):
        """The split rule has no randomness (it's a season-block rule), so
        rebuilding it from scratch must reproduce the exact same storm
        lists -- this is what 'frozen' and 'reproducible' both require."""
        rebuilt = build_split_manifest()
        assert rebuilt["train"]["storm_ids"] == manifest["train"]["storm_ids"]
        assert rebuilt["val"]["storm_ids"] == manifest["val"]["storm_ids"]
        assert rebuilt["test"]["storm_ids"] == manifest["test"]["storm_ids"]


class TestValidateSplitIntegrityFunction:
    """Direct unit tests of the guard function on synthetic, deliberately
    broken manifests -- these must FAIL LOUDLY, not silently pass."""

    def test_raises_on_overlap(self):
        bad = {
            "train": {"storm_ids": ["A", "B"]},
            "val": {"storm_ids": ["B", "C"]},   # B leaked into both
            "test": {"storm_ids": ["D"]},
        }
        with pytest.raises(ValueError, match="Split integrity violated"):
            validate_split_integrity(bad)

    def test_passes_on_disjoint(self):
        good = {
            "train": {"storm_ids": ["A", "B"]},
            "val": {"storm_ids": ["C"]},
            "test": {"storm_ids": ["D", "E"]},
        }
        validate_split_integrity(good)  # must not raise
        assert good["integrity_check"]["all_disjoint"] is True
