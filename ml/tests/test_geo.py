"""Tests for ml/geostrom_ml/features/geo.py: longitude wrapping, Haversine,
bearing, and along/cross-track decomposition.

Directly implements the antimeridian test cases the Phase 2 task brief
specifies verbatim:
    179 deg -> -179 deg   should NOT be interpreted as ~358 deg
    -179 deg -> 179 deg   should NOT be interpreted as ~358 deg
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.geostrom_ml.features.geo import (
    R_EARTH_KM, along_cross_track_km, destination_point, displace,
    haversine_km, initial_bearing_deg, wrap_lon_deg, wrap_lon_diff,
)


class TestWrapLonDiff:
    def test_antimeridian_eastward(self):
        """179 -> -179 is a 2-degree EASTWARD step, not -358."""
        assert wrap_lon_diff(179, -179) == pytest.approx(2.0)

    def test_antimeridian_westward(self):
        """-179 -> 179 is a 2-degree WESTWARD step, not +358."""
        assert wrap_lon_diff(-179, 179) == pytest.approx(-2.0)

    def test_no_wrap_needed(self):
        assert wrap_lon_diff(0, 10) == pytest.approx(10.0)
        assert wrap_lon_diff(10, 0) == pytest.approx(-10.0)

    def test_large_negative_no_wrap(self):
        assert wrap_lon_diff(-100, -90) == pytest.approx(10.0)

    def test_vectorised(self):
        lon1 = np.array([179.0, -179.0, 0.0])
        lon2 = np.array([-179.0, 179.0, 10.0])
        out = wrap_lon_diff(lon1, lon2)
        np.testing.assert_allclose(out, [2.0, -2.0, 10.0])

    def test_result_never_exceeds_180(self):
        rng = np.random.default_rng(0)
        lon1 = rng.uniform(-180, 180, 2000)
        lon2 = rng.uniform(-180, 180, 2000)
        d = wrap_lon_diff(lon1, lon2)
        assert np.all(np.abs(d) <= 180.0)


class TestWrapLonDeg:
    def test_identity_within_range(self):
        assert wrap_lon_deg(90.0) == pytest.approx(90.0)
        assert wrap_lon_deg(-90.0) == pytest.approx(-90.0)

    def test_wraps_above_180(self):
        assert wrap_lon_deg(190.0) == pytest.approx(-170.0)

    def test_wraps_below_neg180(self):
        assert wrap_lon_deg(-190.0) == pytest.approx(170.0)

    def test_exact_180_stays_180(self):
        assert wrap_lon_deg(180.0) == pytest.approx(180.0)
        assert wrap_lon_deg(-180.0) == pytest.approx(180.0)


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(10, 20, 10, 20) == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_longitude_at_equator(self):
        # 1 deg longitude at the equator ~ 111.19 km (Earth circumference / 360)
        d = haversine_km(0, 0, 0, 1)
        expected = 2 * np.pi * R_EARTH_KM / 360.0
        assert d == pytest.approx(expected, rel=1e-3)

    def test_antimeridian_short_distance_not_inflated(self):
        """A storm at 179.9E and one at 179.9W (i.e. -179.9) are ~22 km apart,
        not ~40,000 km apart. This is THE regression test for the antimeridian
        bug the Phase 2 task brief is guarding against.
        """
        d = haversine_km(10, 179.9, 10, -179.9)
        assert d < 50.0

    def test_symmetric(self):
        d1 = haversine_km(10, 20, 30, 40)
        d2 = haversine_km(30, 40, 10, 20)
        assert d1 == pytest.approx(d2)

    def test_triangle_inequality_sanity(self):
        # known: NYC (40.7,-74.0) to London (51.5,-0.1) ~ 5570 km
        d = haversine_km(40.7, -74.0, 51.5, -0.1)
        assert 5400 < d < 5750


class TestBearing:
    def test_due_east(self):
        b = initial_bearing_deg(0, 0, 0, 10)
        assert b == pytest.approx(90.0, abs=1e-6)

    def test_due_north(self):
        b = initial_bearing_deg(0, 0, 10, 0)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_due_south(self):
        b = initial_bearing_deg(10, 0, 0, 0)
        assert b == pytest.approx(180.0, abs=1e-6)

    def test_across_antimeridian_is_short_path(self):
        # from 10N,179 to 10N,-179 -- bearing should be ~east (~90), not west
        b = initial_bearing_deg(10, 179, 10, -179)
        assert 45 < b < 135


class TestDestinationPoint:
    def test_round_trip_with_bearing_and_haversine(self):
        lat1, lon1 = 15.0, -40.0
        bearing = 47.0
        dist_km = 250.0
        lat2, lon2 = destination_point(lat1, lon1, bearing, dist_km)

        recovered_dist = haversine_km(lat1, lon1, lat2, lon2)
        recovered_bearing = initial_bearing_deg(lat1, lon1, lat2, lon2)
        assert recovered_dist == pytest.approx(dist_km, rel=1e-3)
        assert recovered_bearing == pytest.approx(bearing, abs=0.5)

    def test_zero_distance_returns_origin(self):
        lat2, lon2 = destination_point(10.0, 20.0, 90.0, 0.0)
        assert lat2 == pytest.approx(10.0, abs=1e-6)
        assert lon2 == pytest.approx(20.0, abs=1e-6)

    def test_crossing_antimeridian_wraps_correctly(self):
        lat2, lon2 = destination_point(10.0, 179.5, 90.0, 150.0)
        assert lon2 < 0  # should have wrapped past +180 into negative territory


class TestDisplace:
    def test_simple(self):
        lat, lon = displace(10.0, 20.0, 1.0, 2.0)
        assert lat == pytest.approx(11.0)
        assert lon == pytest.approx(22.0)

    def test_wraps_longitude(self):
        lat, lon = displace(10.0, 179.0, 0.0, 5.0)
        assert lon == pytest.approx(-176.0)


class TestAlongCrossTrack:
    def test_zero_error_when_prediction_matches_actual(self):
        along, cross = along_cross_track_km(0, 0, 5, 5, 5, 5)
        assert along == pytest.approx(0.0, abs=1e-6)
        assert cross == pytest.approx(0.0, abs=1e-6)

    def test_pure_overshoot_is_along_track_only(self):
        # motion due east; prediction overshoots further east on the same line
        along, cross = along_cross_track_km(0, 0, 0, 2, 0, 3)
        assert along > 0
        assert cross == pytest.approx(0.0, abs=1e-3)
