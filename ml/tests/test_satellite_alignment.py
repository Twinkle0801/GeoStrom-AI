"""HURSAT <-> IBTrACS temporal alignment and spatial QC."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.satellite.alignment import join_frames_to_ibtracs


def _canonical(storm_id="2001213N15040", ts="2001-08-01T00:00:00", lat=15.0, lon=-40.0):
    return pd.DataFrame([{
        "storm_id": storm_id, "satellite_timestamp": pd.Timestamp(ts),
        "satellite_lat": lat, "satellite_lon": lon, "vza": 20.0,
        "source_file": "f.nc", "satellite_id": "GOE-8",
    }])


class TestTemporalJoin:
    def test_exact_match_is_observed_and_ok(self, synthetic_ibtracs_full_track):
        joined = join_frames_to_ibtracs(_canonical(), synthetic_ibtracs_full_track)
        row = joined.iloc[0]
        assert row["qc_status"] == "ok"
        assert row["temporal_offset_minutes"] == pytest.approx(0.0)
        assert bool(row["is_observed"]) is True
        assert bool(row["is_interpolated"]) is False

    def test_frame_nearest_an_interpolated_row_is_flagged_not_observed(self, synthetic_ibtracs_full_track):
        # track's 5th row (index 4, t=+24h -> lat 15.6, lon -40.8) has IFLAG 'I...'
        # per the fixture; position must match closely enough to pass the
        # spatial QC gate independently of the observed/interpolated check.
        joined = join_frames_to_ibtracs(_canonical(ts="2001-08-02T00:00:00", lat=15.6, lon=-40.8),
                                        synthetic_ibtracs_full_track)
        row = joined.iloc[0]
        assert row["qc_status"] == "ok"
        assert bool(row["is_observed"]) is False
        assert bool(row["is_interpolated"]) is True

    def test_no_row_within_tolerance_is_rejected(self, synthetic_ibtracs_full_track):
        far_future = _canonical(ts="2001-08-10T00:00:00")  # days beyond the track
        joined = join_frames_to_ibtracs(far_future, synthetic_ibtracs_full_track, tolerance_min=90)
        row = joined.iloc[0]
        assert row["qc_status"] == "rejected"
        assert row["qc_reason"] == "no_ibtracs_row_within_90min"

    def test_unknown_storm_id_is_rejected(self, synthetic_ibtracs_full_track):
        joined = join_frames_to_ibtracs(_canonical(storm_id="9999999N99999"),
                                        synthetic_ibtracs_full_track)
        row = joined.iloc[0]
        assert row["qc_status"] == "rejected"
        assert row["qc_reason"] == "storm_id_not_in_ibtracs"

    def test_offset_within_tolerance_but_nonzero_is_recorded(self, synthetic_ibtracs_full_track):
        joined = join_frames_to_ibtracs(_canonical(ts="2001-08-01T00:45:00"),
                                        synthetic_ibtracs_full_track, tolerance_min=90)
        row = joined.iloc[0]
        assert row["qc_status"] == "ok"
        assert row["temporal_offset_minutes"] == pytest.approx(45.0)


class TestSpatialQC:
    def test_close_position_passes_the_50km_gate(self, synthetic_ibtracs_full_track):
        joined = join_frames_to_ibtracs(_canonical(lat=15.05, lon=-40.05), synthetic_ibtracs_full_track)
        row = joined.iloc[0]
        assert row["qc_status"] == "ok"
        assert row["spatial_distance_km"] < 50.0

    def test_far_position_fails_the_50km_gate(self, synthetic_ibtracs_full_track):
        joined = join_frames_to_ibtracs(_canonical(lat=25.0, lon=-50.0), synthetic_ibtracs_full_track)
        row = joined.iloc[0]
        assert row["qc_status"] == "rejected"
        assert "spatial_separation" in row["qc_reason"]
        assert row["spatial_distance_km"] >= 50.0

    def test_threshold_is_not_configurable_below_the_locked_50km_default(self, synthetic_ibtracs_full_track):
        """The Phase 4 task explicitly forbids loosening the 50 km gate --
        this test locks the DEFAULT so a future edit cannot silently raise it."""
        from ml.geostrom_ml.satellite.schema import SPATIAL_QC_KM
        assert SPATIAL_QC_KM == 50.0
