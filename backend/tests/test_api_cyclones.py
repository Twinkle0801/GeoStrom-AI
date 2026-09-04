"""Storm retrieval endpoints."""

from __future__ import annotations


def test_list_cyclones_returns_seeded_storm(client, sample_storm, sample_observations):
    r = client.get("/api/v1/cyclones")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["sid"] == "2010176N16278"
    assert body["items"][0]["split"] == "test"


def test_get_cyclone_detail(client, sample_storm, sample_observations, sample_prediction):
    r = client.get(f"/api/v1/cyclones/{sample_storm.sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["sid"] == sample_storm.sid
    assert body["has_predictions"] is True
    assert body["bbox"] is not None
    assert len(body["bbox"]) == 4


def test_get_cyclone_detail_no_predictions(client, sample_storm, sample_observations):
    r = client.get(f"/api/v1/cyclones/{sample_storm.sid}")
    assert r.status_code == 200
    assert r.json()["has_predictions"] is False


def test_invalid_storm_returns_404_problem_json(client):
    r = client.get("/api/v1/cyclones/NOT_A_REAL_STORM")
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == 404
    assert "title" in body and "detail" in body


def test_observations_returns_ordered_points(client, sample_storm, sample_observations):
    r = client.get(f"/api/v1/cyclones/{sample_storm.sid}/observations")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    timestamps = [o["ts"] for o in body]
    assert timestamps == sorted(timestamps)
    for o in body:
        assert o["data_kind"] == "observed"
        assert -90 <= o["lat"] <= 90
        assert -180 <= o["lon"] <= 180


def test_observations_for_unknown_storm_is_404(client):
    r = client.get("/api/v1/cyclones/NOPE/observations")
    assert r.status_code == 404


def test_season_filter(client, sample_storm, sample_observations):
    r = client.get("/api/v1/cyclones?season=2010")
    assert r.json()["total"] == 1
    r2 = client.get("/api/v1/cyclones?season=1999")
    assert r2.json()["total"] == 0


def test_pagination_limit_respected(client, sample_storm, sample_observations):
    r = client.get("/api/v1/cyclones?limit=1&offset=0")
    body = r.json()
    assert body["limit"] == 1
    assert len(body["items"]) <= 1
