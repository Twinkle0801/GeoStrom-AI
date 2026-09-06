"""Phase 11 integration audit: `app/services/geo.py`'s own module docstring
claims it is "byte-for-byte the same formulas as ml/geostrom_ml/features/
geo.py ... No behaviour differs; only the import graph does" -- a real,
specific, checkable claim that had never actually been tested (found via
"do not assume documentation and implementation are identical; compare
them", Phase 11's explicit instruction).

This is the ONLY place in the whole repository that imports both packages
in the same process -- a test-only, deliberate exception to "backend never
imports ml/" (docs/SYSTEM_ARCHITECTURE.md §8.1), since the rule is about the
served FastAPI application's runtime dependency graph, not about verifying
two independently-maintained implementations stay identical. No application
code imports `ml` as a result of this file existing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.services import geo as backend_geo  # noqa: E402
from ml.geostrom_ml.features import geo as ml_geo  # noqa: E402

# A representative set of cases, deliberately including the exact
# antimeridian scenarios ml/tests/test_geo.py already exercises (Phase 2),
# so parity is checked precisely where a divergence would be most dangerous.
LON_DIFF_CASES = [
    (0.0, 0.0), (10.0, 20.0), (-170.0, 170.0), (179.0, -179.0), (-179.0, 179.0),
    (170.0, -170.0), (0.0, 179.9), (0.0, -179.9),
]
WRAP_CASES = [0.0, 90.0, 180.0, -180.0, 181.0, -181.0, 360.0, -360.0, 359.9, -359.9]
HAVERSINE_CASES = [
    (0.0, 0.0, 0.0, 0.0), (25.4, -87.6, 25.9, -88.9), (0.0, 179.5, 0.0, -179.5),
    (20.0, 179.9, 20.1, -179.9), (-10.0, 100.0, 10.0, -100.0), (89.9, 0.0, 89.9, 180.0),
]
DISPLACE_CASES = [
    (25.4, -87.6, 0.5, -1.3), (0.0, 179.5, 0.1, 2.0), (0.0, -179.5, 0.1, -2.0),
    (10.0, 0.0, 0.0, 0.0),
]


class TestBackendGeoMatchesMlGeoExactly:
    @pytest.mark.parametrize("lon1,lon2", LON_DIFF_CASES)
    def test_wrap_lon_diff_parity(self, lon1, lon2):
        backend_result = backend_geo.wrap_lon_diff(lon1, lon2)
        ml_result = float(ml_geo.wrap_lon_diff(lon1, lon2))
        assert backend_result == ml_result

    @pytest.mark.parametrize("lon", WRAP_CASES)
    def test_wrap_lon_deg_parity(self, lon):
        backend_result = backend_geo.wrap_lon_deg(lon)
        ml_result = float(ml_geo.wrap_lon_deg(lon))
        assert backend_result == ml_result

    @pytest.mark.parametrize("lat1,lon1,lat2,lon2", HAVERSINE_CASES)
    def test_haversine_km_parity(self, lat1, lon1, lat2, lon2):
        backend_result = backend_geo.haversine_km(lat1, lon1, lat2, lon2)
        ml_result = float(ml_geo.haversine_km(lat1, lon1, lat2, lon2))
        assert backend_result == pytest.approx(ml_result, abs=1e-9)

    @pytest.mark.parametrize("lat,lon,dlat,dlon", DISPLACE_CASES)
    def test_displace_parity(self, lat, lon, dlat, dlon):
        backend_lat, backend_lon = backend_geo.displace(lat, lon, dlat, dlon)
        ml_lat, ml_lon = ml_geo.displace(lat, lon, dlat, dlon)
        assert backend_lat == pytest.approx(float(ml_lat), abs=1e-9)
        assert backend_lon == pytest.approx(float(ml_lon), abs=1e-9)

    def test_antimeridian_short_distance_parity(self):
        """The exact regression case ml/tests/test_geo.py pins (Phase 2):
        a pair either side of +/-180 must be a SHORT great-circle distance,
        never inflated by a naive unwrapped subtraction -- re-verified here
        against the backend's own copy specifically."""
        backend_km = backend_geo.haversine_km(0.0, 179.9, 0.0, -179.9)
        assert backend_km < 50.0  # ~0.2 degrees of longitude at the equator
        assert backend_km == pytest.approx(float(ml_geo.haversine_km(0.0, 179.9, 0.0, -179.9)), abs=1e-9)

    def test_r_earth_km_constant_parity(self):
        assert backend_geo.R_EARTH_KM == ml_geo.R_EARTH_KM
