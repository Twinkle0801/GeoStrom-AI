"""Phase 11 integration audit: traces DB row -> evidence_builder ->
GeminiExplanationService -> ExplainResponse -> JSON, asserting the SAME
numeric values survive every hop unchanged. Existing Phase 9 tests
(`test_api_explain.py`) check `source`/`fallback_reason`/schema-version but
never assert the `evidence` field's actual CONTENT matches what was seeded
in the database -- a real, findable gap ("do not assume documentation and
implementation are identical; compare them").

Also verifies storm-ID consistency, model-version propagation, and that
observed values (current_state, from `Observation` rows) are never
confused with predicted values (forecasts, from `Prediction` rows) --
task §11/§18's explicit audit points.
"""

from __future__ import annotations

import datetime as dt

from app.api.v1.explain import get_gemini_client
from app.db.models import Prediction
from app.main import app
from tests.gemini_mocks import MockGeminiClient

GOOD_JSON = (
    '{"summary": "About 92 kt at +24h.", "intensity_explanation": "About 92 kt at +24h.", '
    '"track_explanation": "Near 17.15, -87.15 at +6h via CLIPER-style Ridge.", '
    '"classification_explanation": "No classification result is available.", '
    '"limitations": "This is not an operational forecast."}'
)


def _override_client(mock_client):
    app.dependency_overrides[get_gemini_client] = lambda: mock_client


def _clear_override():
    app.dependency_overrides.pop(get_gemini_client, None)


class TestEvidenceChainMatchesSeededRows:
    def test_intensity_forecast_values_survive_the_full_chain_unchanged(
        self, client, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        seeded_pred_wind = 92.4
        seeded_true_wind = 90.0
        p = Prediction(
            sid=sample_storm.sid, task="intensity",
            origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
            lead_hours=24, valid_ts=dt.datetime(2010, 6, 27, 12, tzinfo=dt.timezone.utc),
            model_id=sample_intensity_model.id, pred_wind_kt=seeded_pred_wind,
            true_wind_kt=seeded_true_wind, wind_error_kt=seeded_pred_wind - seeded_true_wind,
        )
        db_session.add(p)
        db_session.flush()

        _override_client(None)  # forces fallback -- no Gemini variance in what we're checking
        try:
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            assert r.status_code == 200
            body = r.json()
            fc = next(f for f in body["evidence"]["intensity"]["forecasts"] if f["lead_hours"] == 24)
            # Exact float round-trip through evidence_builder -> Pydantic -> JSON -> HTTP.
            assert fc["pred_wind_kt"] == seeded_pred_wind
            assert fc["true_wind_kt"] == seeded_true_wind
        finally:
            _clear_override()

    def test_model_name_and_version_propagate_unchanged_end_to_end(
        self, client, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _override_client(None)
        try:
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            body = r.json()
            assert body["track_model"] == {"name": "track_cliper", "version": "v1"}
            assert body["evidence"]["track"]["context"]["model_name"] == "track_cliper"
            assert body["evidence"]["track"]["context"]["model_version"] == "v1"
            assert body["evidence"]["track"]["context"]["display_name"] == "CLIPER-style Ridge"
        finally:
            _clear_override()

    def test_storm_id_is_identical_across_request_and_every_response_layer(
        self, client, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _override_client(None)
        try:
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            body = r.json()
            assert body["sid"] == sample_storm.sid
            assert body["evidence"]["storm"]["sid"] == sample_storm.sid
        finally:
            _clear_override()

    def test_current_state_comes_from_observations_never_from_predictions(
        self, client, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        """The evidence packet's `current_state` (an OBSERVED value) must
        match a real `Observation` row's lat/lon/wind -- never a
        `Prediction` row's predicted value, even though both tables can
        hold a plausible-looking wind_kt number at the same timestamp."""
        _override_client(None)
        try:
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            body = r.json()
            current_state = body["evidence"]["current_state"]
            observed_winds = {o.wind_kt for o in sample_observations}
            assert current_state["wind_kt"] in observed_winds
            # sample_prediction's pred_wind_kt is None (it's a track-only
            # fixture) -- confirming current_state didn't pull from it by
            # accident would be vacuous; instead assert current_state's
            # position matches a real observation's position exactly.
            observed_positions = {(o.lat, o.lon) for o in sample_observations}
            assert (current_state["lat"], current_state["lon"]) in observed_positions
        finally:
            _clear_override()

    def test_gemini_source_response_also_carries_the_same_seeded_evidence(
        self, client, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        """Not just the fallback path -- a real (mocked) Gemini success
        response must carry the identical, unmodified evidence packet too."""
        db_session.add(Prediction(
            sid=sample_storm.sid, task="intensity",
            origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
            lead_hours=24, valid_ts=dt.datetime(2010, 6, 27, 12, tzinfo=dt.timezone.utc),
            model_id=sample_intensity_model.id, pred_wind_kt=92.4,
        ))
        db_session.flush()
        _override_client(MockGeminiClient(responses=[GOOD_JSON]))
        try:
            r = client.post("/api/v1/explain/forecast", json={"sid": sample_storm.sid})
            body = r.json()
            assert body["source"] == "gemini"
            assert body["evidence"]["track"]["forecasts"][0]["pred_lat"] == sample_prediction.pred_lat
            assert body["evidence"]["track"]["forecasts"][0]["pred_lon"] == sample_prediction.pred_lon
        finally:
            _clear_override()
