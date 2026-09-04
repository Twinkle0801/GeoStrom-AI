"""Deterministic duplicate-frame resolution (VZA-based)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.satellite.dedup import duplicate_summary, select_canonical_frames


def _inv(rows):
    return pd.DataFrame(rows)


class TestCanonicalSelection:
    def test_single_frame_group_passes_through_unchanged(self):
        inv = _inv([{"storm_id": "A", "satellite_timestamp": pd.Timestamp("2001-01-01"),
                     "vza": 20.0, "source_file": "a.nc"}])
        canonical, rejected = select_canonical_frames(inv)
        assert len(canonical) == 1
        assert len(rejected) == 0

    def test_lowest_vza_wins(self):
        t = pd.Timestamp("2001-01-01")
        inv = _inv([
            {"storm_id": "A", "satellite_timestamp": t, "vza": 40.0, "source_file": "high_vza.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 15.0, "source_file": "low_vza.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 25.0, "source_file": "mid_vza.nc"},
        ])
        canonical, rejected = select_canonical_frames(inv)
        assert len(canonical) == 1
        assert canonical.iloc[0]["source_file"] == "low_vza.nc"
        assert len(rejected) == 2
        assert set(rejected["rejection_reason"]) == {"duplicate_frame_higher_vza"}

    def test_missing_vza_falls_back_to_filename_tiebreak_deterministically(self):
        t = pd.Timestamp("2001-01-01")
        inv = _inv([
            {"storm_id": "A", "satellite_timestamp": t, "vza": None, "source_file": "z_last.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": None, "source_file": "a_first.nc"},
        ])
        canonical, rejected = select_canonical_frames(inv)
        assert canonical.iloc[0]["source_file"] == "a_first.nc"
        assert rejected.iloc[0]["rejection_reason"] == "duplicate_frame_no_vza_tiebreak_filename"

    def test_selection_is_reproducible_across_repeated_runs(self):
        t = pd.Timestamp("2001-01-01")
        inv = _inv([
            {"storm_id": "A", "satellite_timestamp": t, "vza": 30.0, "source_file": "b.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 10.0, "source_file": "a.nc"},
        ])
        c1, _ = select_canonical_frames(inv)
        c2, _ = select_canonical_frames(inv.sample(frac=1.0, random_state=0))  # shuffled input
        assert c1.iloc[0]["source_file"] == c2.iloc[0]["source_file"] == "a.nc"

    def test_different_timestamps_are_not_treated_as_duplicates(self):
        inv = _inv([
            {"storm_id": "A", "satellite_timestamp": pd.Timestamp("2001-01-01T00:00"),
             "vza": 20.0, "source_file": "t0.nc"},
            {"storm_id": "A", "satellite_timestamp": pd.Timestamp("2001-01-01T06:00"),
             "vza": 20.0, "source_file": "t1.nc"},
        ])
        canonical, rejected = select_canonical_frames(inv)
        assert len(canonical) == 2
        assert len(rejected) == 0

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError):
            select_canonical_frames(pd.DataFrame({"storm_id": ["A"]}))


class TestDuplicateSummary:
    def test_counts_groups_and_duplicates(self):
        t = pd.Timestamp("2001-01-01")
        inv = _inv([
            {"storm_id": "A", "satellite_timestamp": t, "vza": 1.0, "source_file": "1"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 2.0, "source_file": "2"},
            {"storm_id": "B", "satellite_timestamp": t, "vza": 1.0, "source_file": "3"},
        ])
        s = duplicate_summary(inv)
        assert s["n_candidates"] == 3
        assert s["n_groups"] == 2
        assert s["n_groups_with_duplicates"] == 1
        assert s["max_frames_at_one_time"] == 2

    def test_empty_inventory(self):
        s = duplicate_summary(pd.DataFrame(columns=["storm_id", "satellite_timestamp"]))
        assert s["n_candidates"] == 0
