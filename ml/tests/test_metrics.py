"""Tests for ml/geostrom_ml/evaluation/metrics.py on hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from ml.geostrom_ml.evaluation.metrics import (
    bias, intensity_metrics, mae, rmse, skill_vs_baseline, track_point_metrics,
)


class TestBasicMetrics:
    def test_mae_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16) / 2))

    def test_bias_positive_means_overforecast(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 14.0])
        assert bias(y_true, y_pred) == pytest.approx(3.0)

    def test_bias_negative_means_underforecast(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([8.0, 6.0])
        assert bias(y_true, y_pred) == pytest.approx(-3.0)

    def test_perfect_prediction_zeros_everything(self):
        y = np.array([5.0, 15.0, 25.0])
        assert mae(y, y) == 0.0
        assert rmse(y, y) == 0.0
        assert bias(y, y) == 0.0


class TestSkillVsBaseline:
    def test_model_better_than_baseline_is_positive(self):
        s = skill_vs_baseline(model_mae=5.0, baseline_mae=10.0)
        assert s == pytest.approx(50.0)

    def test_model_tied_with_baseline_is_zero(self):
        s = skill_vs_baseline(model_mae=10.0, baseline_mae=10.0)
        assert s == pytest.approx(0.0)

    def test_model_worse_than_baseline_is_negative_not_clipped(self):
        s = skill_vs_baseline(model_mae=15.0, baseline_mae=10.0)
        assert s == pytest.approx(-50.0)


class TestIntensityMetrics:
    def test_returns_expected_keys(self):
        y = np.array([10.0, 20.0])
        out = intensity_metrics(y, y + 1)
        assert set(out) == {"n", "mae_kt", "rmse_kt", "bias_kt"}
        assert out["n"] == 2


class TestTrackPointMetrics:
    def test_zero_error_when_prediction_is_exact(self):
        ref_lat = np.array([10.0])
        ref_lon = np.array([20.0])
        true_lat = np.array([10.5])
        true_lon = np.array([20.5])
        pred_dlat = true_lat - ref_lat
        pred_dlon = true_lon - ref_lon
        m = track_point_metrics(ref_lat, ref_lon, true_lat, true_lon, pred_dlat, pred_dlon)
        assert m["mean_track_error_km"] == pytest.approx(0.0, abs=1e-6)
        assert m["median_track_error_km"] == pytest.approx(0.0, abs=1e-6)

    def test_error_is_positive_when_wrong(self):
        ref_lat = np.array([10.0])
        ref_lon = np.array([20.0])
        true_lat = np.array([10.5])
        true_lon = np.array([20.5])
        pred_dlat = np.array([0.0])   # model predicts no motion at all
        pred_dlon = np.array([0.0])
        m = track_point_metrics(ref_lat, ref_lon, true_lat, true_lon, pred_dlat, pred_dlon)
        assert m["mean_track_error_km"] > 0

    def test_antimeridian_prediction_not_penalised_incorrectly(self):
        """A storm just west of the antimeridian, actual and predicted both
        drift slightly east across it -- error must stay small (few km),
        not spuriously huge from a raw-degree miscalculation."""
        ref_lat = np.array([15.0])
        ref_lon = np.array([179.7])
        true_lat = np.array([15.0])
        true_lon = np.array([-179.9])   # actual crossed the dateline
        pred_dlat = np.array([0.0])
        pred_dlon = np.array([0.5])     # model correctly predicts crossing too
        m = track_point_metrics(ref_lat, ref_lon, true_lat, true_lon, pred_dlat, pred_dlon)
        assert m["mean_track_error_km"] < 50.0

    def test_along_track_dominant_for_speed_error(self):
        """Actual and predicted are on the same bearing from origin, so the
        error should be almost entirely along-track, ~zero cross-track."""
        ref_lat, ref_lon = np.array([0.0]), np.array([0.0])
        true_lat, true_lon = np.array([0.0]), np.array([2.0])
        pred_dlat, pred_dlon = np.array([0.0]), np.array([3.0])  # overshoot
        m = track_point_metrics(ref_lat, ref_lon, true_lat, true_lon, pred_dlat, pred_dlon)
        assert abs(m["mean_abs_cross_track_km"]) < 1.0
        assert m["mean_abs_along_track_km"] > 50.0
