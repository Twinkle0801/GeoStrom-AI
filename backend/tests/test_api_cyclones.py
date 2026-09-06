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


# ---------------------------------------------------------------------------
# Phase 12 -- API robustness audit (task §11: malformed params, oversized
# requests, response schema consistency).
# ---------------------------------------------------------------------------

def test_requested_limit_above_max_page_size_is_clamped_not_rejected(client, sample_storm, sample_observations):
    """`limit` has no upper bound in the Query schema itself (only `ge=1`)
    -- the clamp to `settings.max_page_size` happens in the route body.
    A client asking for an absurd page size must get a valid, bounded
    response (its actual `max_page_size`), never a 422 and never literally
    that many rows requested passed through to the database."""
    from app.core.config import get_settings
    r = client.get("/api/v1/cyclones?limit=999999")
    assert r.status_code == 200
    assert r.json()["limit"] == get_settings().max_page_size


def test_non_numeric_season_returns_422_problem_json_not_500(client):
    r = client.get("/api/v1/cyclones?season=not-a-year")
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == 422
    assert "title" in body and "detail" in body


def test_negative_offset_returns_422(client):
    r = client.get("/api/v1/cyclones?offset=-1")
    assert r.status_code == 422


def test_zero_limit_returns_422(client):
    r = client.get("/api/v1/cyclones?limit=0")
    assert r.status_code == 422


def test_path_traversal_like_storm_id_is_a_plain_404_not_an_error(client):
    """`sid` flows straight into a parameterized `db.get(Storm, sid)` --
    there is no file-path or raw-SQL use of it anywhere, so a
    path-traversal-shaped or otherwise adversarial string is just a
    non-existent primary key, not a security-relevant special case."""
    for weird_sid in ("../../etc/passwd", "'; DROP TABLE storms; --", "a" * 500, "😀🌀"):
        r = client.get(f"/api/v1/cyclones/{weird_sid}")
        assert r.status_code == 404, f"sid={weird_sid!r} returned {r.status_code}"


def test_empty_string_q_filter_is_treated_as_no_filter(client, sample_storm, sample_observations):
    r = client.get("/api/v1/cyclones?q=")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_observations_range_with_from_after_to_returns_an_empty_list_not_an_error(
    client, sample_storm, sample_observations,
):
    """Documents existing, deliberate behaviour: an inverted date range is
    simply an empty result set (200), not a validation error -- the two
    query params are independent filters, not a single range object."""
    r = client.get(f"/api/v1/cyclones/{sample_storm.sid}/observations?from=2099-01-01&to=2000-01-01")
    assert r.status_code == 200
    assert r.json() == []
