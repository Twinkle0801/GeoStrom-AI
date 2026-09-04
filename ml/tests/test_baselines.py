"""Tests for baseline model output shape and basic sanity, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.features.engineering import (
    HORIZONS_H, build_per_timestep_features, build_sequence_windows,
)
from ml.geostrom_ml.models.intensity_baselines import (
    LightGBMIntensity, PersistenceIntensity, RidgeIntensity, target_col,
)
from ml.geostrom_ml.models.track_baselines import (
    CliperTrack, LightGBMTrack, PersistenceTrack, dlat_col, dlon_col,
)


@pytest.fixture
def synthetic_windows(two_synthetic_storms):
    """A larger synthetic dataset (two storms with enough steps to yield
    several windows each) used to fit and predict with real learned models
    without depending on downloaded IBTrACS data."""
    rng = np.random.default_rng(1)
    n = 30
    times = pd.date_range("1995-06-01T00:00:00", periods=n, freq="6h")
    storms = []
    for i in range(6):
        lat = 12.0 + i + 0.12 * np.arange(n) + rng.normal(0, 0.01, n)
        lon = -30.0 - i * 2 - 0.18 * np.arange(n)
        wind = np.clip(25 + rng.normal(0, 1, n).cumsum() * 0.5 + 0.3 * np.arange(n), 15, 140)
        pres = 1010 - 0.6 * wind
        storms.append(pd.DataFrame({
            "SID": f"1995{150+i:03d}N{10+i:02d}0{30+i:02d}", "SEASON": 1995, "NUMBER": i,
            "BASIN": "NA", "SUBBASIN": "MM", "NAME": f"S{i}",
            "ISO_TIME": times, "NATURE": "TS",
            "LAT": lat, "LON": lon,
            "USA_WIND": wind, "USA_PRES": pres,
            "USA_SSHS": 0, "USA_STATUS": "TS",
            "TRACK_TYPE": "main", "IFLAG": "O_____________",
            "STORM_SPEED": 10.0, "STORM_DIR": 300.0,
            "DIST2LAND": 400.0, "LANDFALL": 999.0,
        }))
    df = pd.concat(storms, ignore_index=True)
    feat = build_per_timestep_features(df)
    windows = build_sequence_windows(feat)
    return windows


@pytest.fixture
def train_test_windows(synthetic_windows):
    sids = sorted(synthetic_windows["sid"].unique())
    train_sids, test_sids = sids[:4], sids[4:]
    train = synthetic_windows[synthetic_windows["sid"].isin(train_sids)].reset_index(drop=True)
    test = synthetic_windows[synthetic_windows["sid"].isin(test_sids)].reset_index(drop=True)
    return train, test


class TestPersistenceIntensity:
    def test_predicts_exactly_ref_wind(self, train_test_windows):
        train, test = train_test_windows
        model = PersistenceIntensity()
        model.fit(train)
        preds = model.predict(test)
        for h in HORIZONS_H:
            np.testing.assert_allclose(preds[target_col(h)], test["ref_wind"].to_numpy())

    def test_output_shape(self, train_test_windows):
        _, test = train_test_windows
        preds = PersistenceIntensity().predict(test)
        for h in HORIZONS_H:
            assert len(preds[target_col(h)]) == len(test)


class TestRidgeIntensity:
    def test_fit_predict_shape(self, train_test_windows):
        train, test = train_test_windows
        model = RidgeIntensity()
        model.fit(train)
        preds = model.predict(test)
        for h in HORIZONS_H:
            assert preds[target_col(h)].shape == (len(test),)
            assert np.all(np.isfinite(preds[target_col(h)]))

    def test_no_nan_when_train_has_no_nan(self, train_test_windows):
        train, test = train_test_windows
        model = RidgeIntensity()
        model.fit(train)
        preds = model.predict(test)
        assert not np.any(np.isnan(preds[target_col(24)]))


class TestLightGBMIntensity:
    def test_fit_predict_shape(self, train_test_windows):
        train, test = train_test_windows
        model = LightGBMIntensity(n_estimators=20)
        model.fit(train)
        preds = model.predict(test)
        for h in HORIZONS_H:
            assert preds[target_col(h)].shape == (len(test),)


class TestPersistenceTrack:
    def test_zero_speed_gives_zero_displacement(self):
        df = pd.DataFrame({
            "ref_lat": [10.0], "ref_lon": [20.0],
            "x__storm_speed_kt__lag0": [0.0],
            "x__storm_dir_sin__lag0": [0.0], "x__storm_dir_cos__lag0": [1.0],
        })
        model = PersistenceTrack()
        preds = model.predict(df)
        for h in HORIZONS_H:
            assert preds[dlat_col(h)][0] == pytest.approx(0.0, abs=1e-6)
            assert preds[dlon_col(h)][0] == pytest.approx(0.0, abs=1e-6)

    def test_output_shape(self, train_test_windows):
        _, test = train_test_windows
        preds = PersistenceTrack().predict(test)
        for h in HORIZONS_H:
            assert preds[dlat_col(h)].shape == (len(test),)
            assert preds[dlon_col(h)].shape == (len(test),)

    def test_displacement_grows_with_horizon_for_constant_motion(self):
        df = pd.DataFrame({
            "ref_lat": [10.0], "ref_lon": [20.0],
            "x__storm_speed_kt__lag0": [15.0],
            "x__storm_dir_sin__lag0": [1.0], "x__storm_dir_cos__lag0": [0.0],
        })
        preds = PersistenceTrack().predict(df)
        dist_6 = abs(preds[dlon_col(6)][0])
        dist_24 = abs(preds[dlon_col(24)][0])
        assert dist_24 > dist_6


class TestCliperTrack:
    def test_fit_predict_shape(self, train_test_windows):
        train, test = train_test_windows
        model = CliperTrack()
        model.fit(train)
        preds = model.predict(test)
        for h in HORIZONS_H:
            assert preds[dlat_col(h)].shape == (len(test),)
            assert np.all(np.isfinite(preds[dlat_col(h)]))
            assert np.all(np.isfinite(preds[dlon_col(h)]))


class TestLightGBMTrack:
    def test_fit_predict_shape(self, train_test_windows):
        train, test = train_test_windows
        model = LightGBMTrack(n_estimators=20)
        model.fit(train)
        preds = model.predict(test)
        for h in HORIZONS_H:
            assert preds[dlat_col(h)].shape == (len(test),)
            assert preds[dlon_col(h)].shape == (len(test),)


class TestModelTaskLabels:
    def test_intensity_models_tagged_correctly(self):
        assert PersistenceIntensity().task == "intensity"
        assert RidgeIntensity().task == "intensity"
        assert LightGBMIntensity().task == "intensity"

    def test_track_models_tagged_correctly(self):
        assert PersistenceTrack().task == "track"
        assert CliperTrack().task == "track"
        assert LightGBMTrack().task == "track"
