"""Database-level coordinate validation.

Per the Phase 3 task: "The implementation must correctly handle: latitude
[-90, 90], longitude [-180, 180]. Do not silently swap lat/lon." These
tests prove the CHECK constraints reject invalid data at the database
layer, not merely in application code that could be bypassed.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import Observation


def test_observation_rejects_latitude_out_of_range(db_session, sample_storm):
    bad = Observation(
        sid=sample_storm.sid, ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
        step_index=0, lat=91.0, lon=0.0, geom="SRID=4326;POINT(0 91)",
        is_synoptic=True, is_observed=True,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_observation_rejects_longitude_out_of_range(db_session, sample_storm):
    bad = Observation(
        sid=sample_storm.sid, ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
        step_index=0, lat=0.0, lon=181.0, geom="SRID=4326;POINT(181 0)",
        is_synoptic=True, is_observed=True,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_observation_accepts_boundary_values(db_session, sample_storm):
    ok = Observation(
        sid=sample_storm.sid, ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
        step_index=0, lat=90.0, lon=-180.0, geom="SRID=4326;POINT(-180 90)",
        is_synoptic=True, is_observed=True,
    )
    db_session.add(ok)
    db_session.flush()  # must not raise


def test_observation_unique_sid_ts_enforced(db_session, sample_storm):
    kwargs = dict(sid=sample_storm.sid, ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
                  step_index=0, lat=1.0, lon=1.0, geom="SRID=4326;POINT(1 1)",
                  is_synoptic=True, is_observed=True)
    db_session.add(Observation(**kwargs))
    db_session.flush()
    db_session.add(Observation(**{**kwargs, "step_index": 1}))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prediction_unique_constraint_enforced(db_session, sample_storm, sample_model):
    from app.db.models import Prediction
    kwargs = dict(
        sid=sample_storm.sid, task="track",
        origin_ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc), lead_hours=6,
        valid_ts=dt.datetime(2010, 6, 26, 6, tzinfo=dt.timezone.utc),
        model_id=sample_model.id, pred_lat=1.0, pred_lon=1.0,
    )
    db_session.add(Prediction(**kwargs))
    db_session.flush()
    db_session.add(Prediction(**kwargs))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prediction_rejects_out_of_range_pred_lat(db_session, sample_storm, sample_model):
    from app.db.models import Prediction
    bad = Prediction(
        sid=sample_storm.sid, task="track",
        origin_ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc), lead_hours=6,
        valid_ts=dt.datetime(2010, 6, 26, 6, tzinfo=dt.timezone.utc),
        model_id=sample_model.id, pred_lat=95.0, pred_lon=1.0,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prediction_rejects_nonpositive_lead_hours(db_session, sample_storm, sample_model):
    from app.db.models import Prediction
    bad = Prediction(
        sid=sample_storm.sid, task="track",
        origin_ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc), lead_hours=0,
        valid_ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
        model_id=sample_model.id,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()
