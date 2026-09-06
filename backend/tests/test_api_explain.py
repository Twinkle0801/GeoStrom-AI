"""POST /api/v1/explain/forecast -- API-level tests.

Uses the real test database (`client`/`db_session` fixtures from
conftest.py) plus a dependency override for the Gemini client, exactly the
way `get_db` is already overridden -- no real Gemini API call happens in
this file either.
"""

from __future__ import annotations

import datetime as dt

from app.api.v1.explain import get_explain_cache, get_gemini_client, get_rate_limiter
from app.db.models import Prediction
from app.gemini.cache import ExplainCache
from app.gemini.ratelimit import RateLimiter
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


# ---------------------------------------------------------------------------
# Phase 12 -- caching and rate limiting (task §14: "cache hit/miss/expiry/
# isolation", "allowed/repeated/limit exceeded/reset", "cache + rate-limit
# interaction"). The `client` fixture already installs a fresh, generously
# bounded ExplainCache/RateLimiter per test (see conftest.py); tests here
# override again where a specific bound matters.
# ---------------------------------------------------------------------------

def _seed_intensity_prediction(db_session, sample_storm, sample_intensity_model) -> None:
    """GOOD_JSON's intensity claim ("92 kt") is only grounded if a matching
    intensity Prediction row exists -- `sample_prediction` alone only seeds
    a TRACK-task row, exactly like `test_valid_gemini_response_returns_gemini_source`
    above already had to work around."""
    db_session.add(Prediction(
        sid=sample_storm.sid, task="intensity",
        origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
        lead_hours=24, valid_ts=dt.datetime(2010, 6, 27, 12, tzinfo=dt.timezone.utc),
        model_id=sample_intensity_model.id, pred_wind_kt=92.4, true_wind_kt=90.0, wind_error_kt=2.4,
    ))
    db_session.flush()


def test_repeated_identical_request_hits_the_cache_and_calls_gemini_only_once(
    client, db_session, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    _seed_intensity_prediction(db_session, sample_storm, sample_intensity_model)
    mock_client = MockGeminiClient(responses=[GOOD_JSON])
    _override_client(mock_client)
    try:
        r1 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        r2 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["source"] == "gemini"
        assert r2.json()["source"] == "gemini"
        assert r2.json()["explanation"] == r1.json()["explanation"]
        # The second, evidence-identical request must be served from the
        # cache -- the mock Gemini client is called exactly once, not twice.
        assert len(mock_client.calls) == 1
    finally:
        _clear_override()


def test_fallback_results_are_never_cached_each_request_calls_gemini_again(
    client, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    from tests.gemini_mocks import timeout_client
    mock_client = timeout_client()
    _override_client(mock_client)
    try:
        r1 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        r2 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r1.json()["source"] == "fallback" and r1.json()["fallback_reason"] == "timeout"
        assert r2.json()["source"] == "fallback" and r2.json()["fallback_reason"] == "timeout"
        # Neither call was cached -- both requests reached the (mocked) Gemini client.
        assert len(mock_client.calls) == 2
    finally:
        _clear_override()


def test_rate_limit_exceeded_returns_429_with_retry_after(
    client, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    # A fresh cache PER CALL (via a lambda that builds a new empty cache
    # each time it's invoked would defeat the point of testing a shared
    # cache, so instead: one shared limiter with max_requests=1, and the
    # cache cleared between calls so each request is a genuine cache MISS
    # and therefore actually consults the limiter (a cache hit would bypass
    # it entirely, by design -- see the dedicated bypass test below).
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    _override_client(MockGeminiClient(responses=[GOOD_JSON]))
    try:
        app.dependency_overrides[get_explain_cache] = lambda: ExplainCache(maxsize=10, ttl_seconds=3600.0)
        r1 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r1.status_code == 200

        app.dependency_overrides[get_explain_cache] = lambda: ExplainCache(maxsize=10, ttl_seconds=3600.0)
        r2 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r2.status_code == 429
        assert "Retry-After" in r2.headers
        assert int(r2.headers["Retry-After"]) > 0
    finally:
        _clear_override()


def test_cache_hit_bypasses_the_rate_limiter_entirely(
    client, db_session, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    """A rate limiter of max_requests=1 must still allow unlimited repeat
    views of an ALREADY-cached explanation -- only the first (cache-miss)
    call should ever consult the limiter."""
    _seed_intensity_prediction(db_session, sample_storm, sample_intensity_model)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=1, window_seconds=60.0)
    mock_client = MockGeminiClient(responses=[GOOD_JSON])
    _override_client(mock_client)
    try:
        for _ in range(5):
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            assert r.status_code == 200
            assert r.json()["source"] == "gemini"
        # Only the very first call was a cache miss and touched the limiter/Gemini.
        assert len(mock_client.calls) == 1
    finally:
        _clear_override()


def test_rate_limiter_reset_allows_requests_again(
    client, sample_storm, sample_observations, sample_model, sample_intensity_model, sample_prediction,
):
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_explain_cache] = lambda: ExplainCache(maxsize=10, ttl_seconds=3600.0)
    _override_client(MockGeminiClient(responses=[GOOD_JSON]))
    try:
        r1 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r1.status_code == 200
        r2 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r2.status_code == 429
        limiter.reset()
        r3 = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
        assert r3.status_code == 200
    finally:
        _clear_override()
