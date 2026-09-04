"""Geodesic utilities: Haversine distance, bearings, and longitude wrapping.

docs/DATA_STRATEGY.md pitfall #5 (antimeridian crossing) and Phase 2 task 7
require longitude handling to be explicit, never raw subtraction. Every
function here that touches longitude differences goes through
`wrap_lon_diff`, which returns the shortest signed angular difference in
(-180, 180].

docs/PHASE_1_DATASET_VERIFICATION.md §13 flagged antimeridian behaviour as
NOT YET code-path-tested (neither Phase 1 sample basin crossed +/-180 deg).
This module is the first code in the project to implement and test that path
explicitly, per the Phase 2 task instructions.
"""

from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0088  # matches ML_ARCHITECTURE.md §7.2


def wrap_lon_deg(lon):
    """Wrap a longitude (scalar or array) into (-180, 180]."""
    lon = np.asarray(lon, dtype=np.float64)
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    # np.mod maps -180 -> -180, but a value that lands exactly on -180 after
    # wrapping should read as +180 by the (-180, 180] convention used here.
    wrapped = np.where(wrapped == -180.0, 180.0, wrapped)
    return wrapped if wrapped.ndim else float(wrapped)


def wrap_lon_diff(lon1, lon2):
    """Shortest signed angular difference lon2 - lon1, wrapped to (-180, 180].

    This is the ONLY sanctioned way to compute a longitude displacement in
    this codebase. Raw subtraction (`lon2 - lon1`) is wrong whenever the pair
    straddles the antimeridian: e.g. lon1=179, lon2=-179 is a 2-degree
    eastward displacement, not a 358-degree one.

    Examples (see ml/tests/test_geo.py):
        wrap_lon_diff(179, -179)  ==  2.0   (not -358.0)
        wrap_lon_diff(-179, 179)  == -2.0   (not  358.0)
    """
    lon1 = np.asarray(lon1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    diff = lon2 - lon1
    wrapped = ((diff + 180.0) % 360.0) - 180.0
    wrapped = np.where(wrapped == -180.0, 180.0, wrapped)
    return wrapped if wrapped.ndim else float(wrapped)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorised; accepts scalars or arrays."""
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = np.radians(wrap_lon_diff(lon1, lon2))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    d = R_EARTH_KM * c
    return d if np.asarray(d).ndim else float(d)


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, in [0, 360) deg."""
    lat1r = np.radians(np.asarray(lat1, dtype=np.float64))
    lat2r = np.radians(np.asarray(lat2, dtype=np.float64))
    dlon = np.radians(wrap_lon_diff(lon1, lon2))
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brng if np.asarray(brng).ndim else float(brng)


def displace(lat, lon, dlat, dlon):
    """Apply a (dlat, dlon) displacement to (lat, lon), wrapping longitude.

    Used to reconstruct an absolute predicted position from a model's
    predicted displacement (docs/PROJECT_REQUIREMENTS.md §2.D: predictions
    are displacements, never absolute coordinates).
    """
    new_lat = np.asarray(lat, dtype=np.float64) + np.asarray(dlat, dtype=np.float64)
    new_lon = wrap_lon_deg(np.asarray(lon, dtype=np.float64) + np.asarray(dlon, dtype=np.float64))
    return new_lat, new_lon


def destination_point(lat, lon, bearing_deg, distance_km):
    """Great-circle destination given a start point, bearing, and distance.

    Inverse of (haversine_km, initial_bearing_deg): used by the track
    persistence baseline to extrapolate a constant-velocity motion vector
    forward in time.
    """
    lat1 = np.radians(np.asarray(lat, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon, dtype=np.float64))
    brng = np.radians(np.asarray(bearing_deg, dtype=np.float64))
    d_r = np.asarray(distance_km, dtype=np.float64) / R_EARTH_KM

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d_r) * np.cos(lat1),
        np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
    )
    lat2_deg = np.degrees(lat2)
    lon2_deg = wrap_lon_deg(np.degrees(lon2))
    return (lat2_deg if np.asarray(lat2_deg).ndim else float(lat2_deg),
            lon2_deg if np.asarray(lon2_deg).ndim else float(lon2_deg))


def along_cross_track_km(origin_lat, origin_lon, actual_lat, actual_lon,
                          pred_lat, pred_lon):
    """Decompose track error into along-track / cross-track components (km).

    Reference axis = the bearing of ACTUAL motion (origin -> actual future
    position). The predicted point's error is projected onto that axis
    (along-track, positive = predicted overshoots in the direction of
    motion) and perpendicular to it (cross-track, positive = predicted point
    lies to the right of the actual motion direction).

    Flat-earth projection is used for the decomposition itself (valid at the
    sub-1000 km errors relevant here); the total error distance itself is
    still full-precision Haversine. See ML_ARCHITECTURE.md §7.2.
    """
    ref_bearing = initial_bearing_deg(origin_lat, origin_lon, actual_lat, actual_lon)
    err_bearing = initial_bearing_deg(actual_lat, actual_lon, pred_lat, pred_lon)
    err_dist = haversine_km(actual_lat, actual_lon, pred_lat, pred_lon)

    angle = np.radians(np.asarray(err_bearing) - np.asarray(ref_bearing))
    along = err_dist * np.cos(angle)
    cross = err_dist * np.sin(angle)
    return (along if np.asarray(along).ndim else float(along),
            cross if np.asarray(cross).ndim else float(cross))
