"""Minimal geodesic helpers, vendored (not imported) from
ml/geostrom_ml/features/geo.py.

docs/SYSTEM_ARCHITECTURE.md §8.1 states the rule explicitly: "backend ...
Never imports from ml/. Depends on the database schema ..., not on training
code." That rule is honoured here at the cost of ~30 lines of duplicated
pure-math functions, rather than importing the `ml` package into the served
application (which would pull pandas/heavy ML deps into backend's runtime
dependency graph and blur the ml/backend boundary the whole architecture is
built on).

This is a DELIBERATE, narrow exception to "don't duplicate logic": the
functions below are byte-for-byte the same formulas as ml/geostrom_ml/
features/geo.py (haversine great-circle distance, antimeridian-safe
longitude wrapping, and destination-point reckoning), already unit-tested
in ml/tests/test_geo.py (Phase 2, 21 passing tests including the exact
antimeridian regression cases). No behaviour differs; only the import graph
does. See docs/PHASE_3_VERTICAL_SLICE.md for the rationale written out in
full.
"""

from __future__ import annotations

import math

R_EARTH_KM = 6371.0088


def wrap_lon_deg(lon: float) -> float:
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def wrap_lon_diff(lon1: float, lon2: float) -> float:
    diff = lon2 - lon1
    wrapped = ((diff + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(wrap_lon_diff(lon1, lon2))
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return R_EARTH_KM * 2.0 * math.asin(math.sqrt(a))


def displace(lat: float, lon: float, dlat: float, dlon: float) -> tuple[float, float]:
    """Apply a (dlat, dlon) displacement, wrapping longitude. Used to
    reconstruct an absolute predicted position from a stored displacement --
    predictions are stored as displacements, never absolute coordinates
    (docs/PROJECT_REQUIREMENTS.md §2.D)."""
    return lat + dlat, wrap_lon_deg(lon + dlon)


def validate_lat(lat: float, *, field: str = "lat") -> None:
    if lat is None or not (-90.0 <= lat <= 90.0):
        raise ValueError(f"{field}={lat!r} outside valid range [-90, 90]")


def validate_lon(lon: float, *, field: str = "lon") -> None:
    if lon is None or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{field}={lon!r} outside valid range [-180, 180]")
