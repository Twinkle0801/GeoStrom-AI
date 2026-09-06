"""GET /api/v1/tracks/{sid} -- the map's GeoJSON payload.

Verifies the mandatory Phase 3 rule: observed and predicted geometry are
visually and structurally distinguishable, never merged.
"""

from __future__ import annotations


def test_track_geojson_shape(client, sample_storm, sample_observations, sample_prediction):
    r = client.get(f"/api/v1/tracks/{sample_storm.sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    kinds = {f["properties"]["kind"] for f in body["features"]}
    assert "observed_track" in kinds
    assert "observed_point" in kinds
    assert "predicted_track" in kinds
    assert "predicted_point" in kinds


def test_observed_and_predicted_are_never_the_same_feature(
    client, sample_storm, sample_observations, sample_prediction,
):
    r = client.get(f"/api/v1/tracks/{sample_storm.sid}")
    body = r.json()
    data_kinds_by_feature_kind: dict[str, set[str]] = {}
    for f in body["features"]:
        k = f["properties"]["kind"]
        data_kinds_by_feature_kind.setdefault(k, set()).add(f["properties"]["data_kind"])
    # every observed_* feature is exclusively data_kind='observed'
    assert data_kinds_by_feature_kind["observed_point"] == {"observed"}
    # every predicted_* feature is exclusively data_kind='model_prediction'
    assert data_kinds_by_feature_kind["predicted_point"] == {"model_prediction"}


def test_predicted_point_carries_model_identity(
    client, sample_storm, sample_observations, sample_prediction,
):
    r = client.get(f"/api/v1/tracks/{sample_storm.sid}")
    body = r.json()
    predicted_points = [f for f in body["features"] if f["properties"]["kind"] == "predicted_point"]
    assert len(predicted_points) == 1
    props = predicted_points[0]["properties"]
    assert props["model_name"] == "track_cliper"
    assert props["model_version"] == "v1"
    assert "disclaimer" in props
    assert "not an operational forecast" in props["disclaimer"].lower()


def test_geometry_coordinates_are_lon_lat_order(
    client, sample_storm, sample_observations, sample_prediction,
):
    """GeoJSON spec mandates [lon, lat]. sample_observations use lon in
    [-90,-80], lat in [16,18] -- if these were swapped, coordinates[0]
    would fall outside a plausible longitude range for this fixture."""
    r = client.get(f"/api/v1/tracks/{sample_storm.sid}")
    body = r.json()
    points = [f for f in body["features"] if f["geometry"]["type"] == "Point"]
    for p in points:
        lon, lat = p["geometry"]["coordinates"]
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90
        assert lon < -80  # this fixture's storms sit west of -80 lon
        assert 15 < lat < 20  # and between 15-20 lat


def test_antimeridian_crossing_coordinates_pass_through_unmangled(client, db_session):
    """Phase 11 audit: no North Atlantic storm in the real dataset actually
    crosses +/-180 longitude, so this path has never been exercised by a
    real storm end-to-end. `geometry.py` does no antimeridian-sensitive
    computation of its own -- it only copies already-stored lon/lat
    verbatim into GeoJSON -- so the only thing to verify here is that
    verbatim pass-through, with a synthetic storm, since none exists in the
    real data."""
    import datetime as dt

    from app.db.models import Observation, Storm

    storm = Storm(
        sid="ANTIMERIDIAN_TEST", name=None, season=2011, basin="NA", subbasin=None,
        start_time=dt.datetime(2011, 1, 1, tzinfo=dt.timezone.utc),
        end_time=dt.datetime(2011, 1, 1, 12, tzinfo=dt.timezone.utc),
        n_observations=2, max_wind_kt=50.0, min_pressure_hpa=1000.0, max_category=0,
        made_landfall=None, split="test",
        track_geom="SRID=4326;LINESTRING(179.5 20.0, -179.8 20.1)",
        bbox="SRID=4326;POLYGON((179.5 20.0,-179.8 20.0,-179.8 20.1,179.5 20.1,179.5 20.0))",
    )
    db_session.add(storm)
    db_session.flush()
    for i, (lon, lat) in enumerate([(179.5, 20.0), (-179.8, 20.1)]):
        db_session.add(Observation(
            sid=storm.sid, ts=dt.datetime(2011, 1, 1, 6 * i, tzinfo=dt.timezone.utc),
            step_index=i, lat=lat, lon=lon, geom=f"SRID=4326;POINT({lon} {lat})",
            wind_kt=50.0, pressure_hpa=1000.0, category=0, nature=None,
            storm_speed_kt=None, storm_dir_deg=None, dist2land_km=None,
            is_synoptic=True, is_observed=True,
        ))
    db_session.flush()

    r = client.get(f"/api/v1/tracks/{storm.sid}")
    assert r.status_code == 200
    body = r.json()
    points = sorted(
        (f["geometry"]["coordinates"] for f in body["features"] if f["geometry"]["type"] == "Point"),
        key=lambda c: c[0],
    )
    # Verbatim pass-through: the raw stored values, un-wrapped, un-averaged,
    # un-corrupted by treating 179.5/-179.8 as "close to 0" or similar --
    # a naive implementation might, e.g., accidentally normalise or clamp.
    assert points == [[-179.8, 20.1], [179.5, 20.0]]
    linestring = next(
        f for f in body["features"]
        if f["geometry"]["type"] == "LineString" and f["properties"]["kind"] == "observed_track"
    )
    assert linestring["geometry"]["coordinates"] == [[179.5, 20.0], [-179.8, 20.1]]


def test_track_for_unknown_storm_is_404(client):
    r = client.get("/api/v1/tracks/NOPE")
    assert r.status_code == 404


def test_no_predicted_features_when_no_predictions_exist(
    client, sample_storm, sample_observations,
):
    r = client.get(f"/api/v1/tracks/{sample_storm.sid}")
    body = r.json()
    kinds = {f["properties"]["kind"] for f in body["features"]}
    assert "predicted_point" not in kinds
    assert "predicted_track" not in kinds
    assert "observed_point" in kinds  # observed data is still there
