"""Pydantic schema validation -- defense-in-depth alongside the DB CHECK
constraints in test_db_constraints.py. Coordinates must never silently
pass through out of range, and lat/lon must never be swappable."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionOut
from app.schemas.storm import ObservationOut


def _obs_kwargs(**overrides):
    base = dict(
        ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc), lat=16.9, lon=-86.1,
        wind_kt=40.0, pressure_hpa=1004.0, category=0, nature=None,
        storm_speed_kt=None, storm_dir_deg=None, dist2land_km=None,
        is_synoptic=True, is_observed=True,
    )
    base.update(overrides)
    return base


class TestObservationCoordinateValidation:
    def test_valid_coordinates_accepted(self):
        ObservationOut(**_obs_kwargs())  # must not raise

    def test_latitude_above_90_rejected(self):
        with pytest.raises(ValidationError, match="lat"):
            ObservationOut(**_obs_kwargs(lat=91.0))

    def test_latitude_below_negative_90_rejected(self):
        with pytest.raises(ValidationError, match="lat"):
            ObservationOut(**_obs_kwargs(lat=-91.0))

    def test_longitude_above_180_rejected(self):
        with pytest.raises(ValidationError, match="lon"):
            ObservationOut(**_obs_kwargs(lon=181.0))

    def test_longitude_below_negative_180_rejected(self):
        with pytest.raises(ValidationError, match="lon"):
            ObservationOut(**_obs_kwargs(lon=-181.0))

    def test_boundary_values_accepted(self):
        ObservationOut(**_obs_kwargs(lat=90.0, lon=180.0))
        ObservationOut(**_obs_kwargs(lat=-90.0, lon=-180.0))

    def test_lat_lon_not_silently_swappable(self):
        """A lat/lon swap for a storm at lat~17, lon~-120 would put
        lat=-120 -- outside [-90,90] -- so the validator must catch a
        transposed pair rather than accept it silently. (A swap where both
        axes happen to still be in range for the other field cannot be
        caught by range validation alone; this case, where the swap
        produces an out-of-range value, is the one range validation is
        actually responsible for.)"""
        with pytest.raises(ValidationError):
            ObservationOut(**_obs_kwargs(lat=-120.0, lon=17.0))


def _pred_kwargs(**overrides):
    base = dict(
        task="track", origin_ts=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
        lead_hours=6, valid_ts=dt.datetime(2010, 6, 26, 6, tzinfo=dt.timezone.utc),
        model_name="track_cliper", model_version="v1",
        pred_lat=17.1, pred_lon=-87.1, pred_wind_kt=None, pred_pressure_hpa=None,
        error_radius_km=30.0, true_lat=17.2, true_lon=-87.2, true_wind_kt=45.0,
        track_error_km=6.0, wind_error_kt=None,
    )
    base.update(overrides)
    return base


class TestPredictionCoordinateValidation:
    def test_valid_prediction_accepted(self):
        PredictionOut(**_pred_kwargs())

    def test_pred_lat_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PredictionOut(**_pred_kwargs(pred_lat=200.0))

    def test_true_lon_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PredictionOut(**_pred_kwargs(true_lon=-200.0))

    def test_null_coordinates_accepted_for_intensity_predictions(self):
        """Intensity models predict no position -- pred_lat/pred_lon are
        legitimately null, and must not be forced through range validation."""
        PredictionOut(**_pred_kwargs(
            task="intensity", pred_lat=None, pred_lon=None,
            pred_wind_kt=44.0, track_error_km=None, wind_error_kt=-1.0,
        ))

    def test_model_identity_always_present(self):
        p = PredictionOut(**_pred_kwargs())
        assert p.model_name and p.model_version
        assert p.data_kind == "model_prediction"
        assert "not an operational forecast" in p.disclaimer.lower()
