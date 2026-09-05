"""Evidence-builder tests against the REAL test database (same fixtures and
conventions `tests/conftest.py` already established for Phase 3 -- reused,
not duplicated), per task §6/§7: the packet is assembled from stored rows
only, and every field is traceable to one of them.
"""

from __future__ import annotations

import datetime as dt

from app.db.models import Prediction
from app.gemini.evidence_builder import build_evidence_packet


def _add_intensity_prediction(db_session, storm, model, *, lead_hours=24, pred_wind_kt=92.4,
                              true_wind_kt=90.0, wind_error_kt=2.4):
    p = Prediction(
        sid=storm.sid, task="intensity",
        origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
        lead_hours=lead_hours,
        valid_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc) + dt.timedelta(hours=lead_hours),
        model_id=model.id, pred_wind_kt=pred_wind_kt, true_wind_kt=true_wind_kt,
        wind_error_kt=wind_error_kt,
    )
    db_session.add(p)
    db_session.flush()
    return p


class TestEvidencePacketFromRealRows:
    def test_evidence_schema_version_is_set(self, db_session, sample_storm, sample_observations,
                                            sample_model, sample_intensity_model, sample_prediction):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.evidence_schema_version == "v1"

    def test_storm_fields_are_copied_verbatim(self, db_session, sample_storm, sample_observations,
                                              sample_model, sample_intensity_model, sample_prediction):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.storm.sid == sample_storm.sid
        assert evidence.storm.season == sample_storm.season
        assert evidence.storm.n_observations == sample_storm.n_observations

    def test_track_forecast_matches_the_stored_prediction_row(
        self, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.track is not None
        fc = evidence.track.forecasts[0]
        assert fc.lead_hours == sample_prediction.lead_hours
        assert fc.pred_lat == sample_prediction.pred_lat
        assert fc.pred_lon == sample_prediction.pred_lon
        assert fc.track_error_km == sample_prediction.track_error_km
        assert evidence.track.context.model_name == "track_cliper"
        assert evidence.track.context.display_name == "CLIPER-style Ridge"

    def test_intensity_forecast_matches_the_stored_prediction_row(
        self, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.intensity is not None
        fc = evidence.intensity.forecasts[0]
        assert fc.pred_wind_kt == 92.4
        assert fc.true_wind_kt == 90.0
        assert evidence.intensity.context.model_name == "intensity_lightgbm"
        assert evidence.intensity.context.display_name == "LightGBM"

    def test_missing_intensity_prediction_leaves_intensity_none(
        self, db_session, sample_storm, sample_observations, sample_model, sample_prediction,
    ):
        """No intensity Prediction row was added -- the packet must say so
        structurally (None), never fabricate one."""
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.intensity is None
        assert any("intensity" in lim for lim in evidence.known_limitations)

    def test_classification_is_none_and_documented_as_a_limitation(
        self, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.classification is None
        assert any("classification" in lim for lim in evidence.known_limitations)

    def test_current_state_uses_the_latest_observation_at_or_before_origin(
        self, db_session, sample_storm, sample_observations, sample_model,
        sample_intensity_model, sample_prediction,
    ):
        _add_intensity_prediction(db_session, sample_storm, sample_intensity_model)
        evidence = build_evidence_packet(db_session, sample_storm.sid)
        assert evidence.current_state is not None
        assert evidence.current_state.timestamp <= evidence.track.origin_ts

    def test_unknown_storm_raises(self, db_session):
        import pytest
        with pytest.raises(ValueError):
            build_evidence_packet(db_session, "NOPE")
