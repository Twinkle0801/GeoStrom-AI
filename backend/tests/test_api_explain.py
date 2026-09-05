"""POST /api/v1/explain/forecast -- API-level tests.

Uses the real test database (`client`/`db_session` fixtures from
conftest.py) plus a dependency override for the Gemini client, exactly the
way `get_db` is already overridden -- no real Gemini API call happens in
this file either.
"""

from __future__ import annotations

import datetime as dt

from app.api.v1.explain import get_gemini_client
from app.db.models import Prediction
from app.main import app
from tests.gemini_mocks import MockGeminiClient

GOOD_JSON = (
    '{"summary": "The model predicts about 92 kt at +24h.", '
    '"intensity_explanation": "About 92 kt at +24h.", '
    '"track_explanation": "Near 17.15, -87.15 at +6h via CLIPER-style Ridge.", '
    '"classification_explanation": "No classification result is available.", '
    '"limitations": "This is not an operational forecast."}'
)


def _override_client(mock_client):
    app.dependency_overrides[get_gemini_client] = lambda: mock_client


def _clear_override():
    app.dependency_overrides.pop(get_gemini_client, None)


def test_unknown_storm_is_404(client):
    r = client.post("/api/v1/explain/forecast", json={"sid": "NOPE"})
    assert r.status_code == 404


def test_no_gemini_configured_returns_fallback_source(client, sample_storm, sample_observations,
                                                       sample_model, sample_intensity_model,
                                                       sample_prediction):
    _override_client(None)
    try:
        r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "fallback"
        assert body["fallback_reason"] == "not_configured"
        assert body["evidence_schema_version"] == "v1"
    finally:
        _clear_override()


def test_valid_gemini_response_returns_gemini_source(client, db_session, sample_storm,
                                                     sample_observations, sample_model,
                                                     sample_intensity_model, sample_prediction):
    p = Prediction(
        sid=sample_storm.sid, task="intensity",
        origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
        lead_hours=24, valid_ts=dt.datetime(2010, 6, 27, 12, tzinfo=dt.timezone.utc),
        model_id=sample_intensity_model.id, pred_wind_kt=92.4, true_wind_kt=90.0, wind_error_kt=2.4,
    )
    db_session.add(p)
    db_session.flush()

    _override_client(MockGeminiClient(responses=[GOOD_JSON]))
    try:
        r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "gemini"
        assert body["fallback_reason"] is None
        assert body["validation_violations"] == []
    finally:
        _clear_override()


def test_hallucinated_response_falls_back_through_the_api(
    client, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    bad = GOOD_JSON.replace("92 kt", "150 kt")
    _override_client(MockGeminiClient(responses=[bad, bad]))
    try:
        r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "fallback"
        assert body["fallback_reason"] == "ungrounded_claim"
    finally:
        _clear_override()


def test_response_never_contains_the_api_key(client, monkeypatch, sample_storm,
                                             sample_observations, sample_model,
                                             sample_intensity_model, sample_prediction):
    secret = "sk-totally-secret-test-key-should-never-leak"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    from app.core.config import get_settings
    get_settings.cache_clear()
    try:
        _override_client(MockGeminiClient(responses=[GOOD_JSON]))
        r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert secret not in r.text
        assert "GEMINI_API_KEY" not in r.text
    finally:
        _clear_override()
        get_settings.cache_clear()


def test_response_model_carries_model_versions(client, sample_storm, sample_observations,
                                               sample_model, sample_intensity_model,
                                               sample_prediction, db_session):
    _add = Prediction(
        sid=sample_storm.sid, task="intensity",
        origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
        lead_hours=24, valid_ts=dt.datetime(2010, 6, 27, 12, tzinfo=dt.timezone.utc),
        model_id=sample_intensity_model.id, pred_wind_kt=92.4,
    )
    db_session.add(_add)
    db_session.flush()
    _override_client(None)
    try:
        r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        body = r.json()
        assert body["track_model"] == {"name": "track_cliper", "version": "v1"}
        assert body["intensity_model"] == {"name": "intensity_lightgbm", "version": "v1"}
        assert body["classification_model"] is None
    finally:
        _clear_override()


def test_openapi_schema_never_mentions_the_api_key_field():
    """The generated OpenAPI contract documents request/response SHAPES
    only; `Settings` (which holds `gemini_api_key`) must never be one of
    them -- this would be the failure mode of accidentally using `Settings`
    itself as a `response_model` somewhere."""
    import json
    schema_text = json.dumps(app.openapi())
    assert "gemini_api_key" not in schema_text.lower()
    assert "sk-" not in schema_text
