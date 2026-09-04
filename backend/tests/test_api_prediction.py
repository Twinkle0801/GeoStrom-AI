"""GET /api/v1/prediction/{sid}* endpoints."""

from __future__ import annotations


def test_prediction_returns_seeded_row(client, sample_storm, sample_observations, sample_prediction):
    r = client.get(f"/api/v1/prediction/{sample_storm.sid}")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    row = body[0]
    assert row["model_name"] == "track_cliper"
    assert row["model_version"] == "v1"
    assert row["task"] == "track"
    assert row["lead_hours"] == 6
    assert row["pred_lat"] == 17.15
    assert row["true_lat"] == 17.2
    assert row["track_error_km"] == 6.4


def test_prediction_empty_when_storm_has_no_predictions_ever(
    client, sample_storm, sample_observations,
):
    """A storm that exists but has no predictions at all (no origin_ts to
    default to) returns an empty list with 200, the same convention as
    /observations on an empty result -- not a 404, since the storm resource
    itself is valid."""
    r = client.get(f"/api/v1/prediction/{sample_storm.sid}")
    assert r.status_code == 200
    assert r.json() == []


def test_prediction_404_when_requested_origin_has_no_matching_forecast(
    client, sample_storm, sample_observations, sample_prediction,
):
    """A storm WITH predictions, queried for a filter combination that
    matches none of them, is a genuine 404 (the specific forecast asked
    for does not exist)."""
    r = client.get(f"/api/v1/prediction/{sample_storm.sid}?task=intensity")
    assert r.status_code == 404


def test_prediction_unknown_storm_is_404(client):
    r = client.get("/api/v1/prediction/NOPE")
    assert r.status_code == 404


def test_prediction_series(client, sample_storm, sample_observations, sample_prediction):
    r = client.get(f"/api/v1/prediction/{sample_storm.sid}/series")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_prediction_filter_by_task_matches(client, sample_storm, sample_observations, sample_prediction):
    r = client.get(f"/api/v1/prediction/{sample_storm.sid}?task=track")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_models_list(client, sample_model, sample_intensity_model):
    r = client.get("/api/v1/prediction/models/list")
    assert r.status_code == 200
    body = r.json()
    names = {m["name"] for m in body}
    assert {"track_cliper", "intensity_lightgbm"} <= names
    for m in body:
        assert "metrics" in m and isinstance(m["metrics"], dict)


def test_models_list_task_filter(client, sample_model, sample_intensity_model):
    r = client.get("/api/v1/prediction/models/list?task=track")
    body = r.json()
    assert all(m["task"] == "track" for m in body)
    assert len(body) == 1
