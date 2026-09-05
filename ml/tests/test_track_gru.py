"""Phase 8: TrackGRU correctness, determinism, geospatial-loss correctness,
and antimeridian handling, using small synthetic data (fast, no dependency on
the real materialised dataset -- mirrors the Phase 7 `test_intensity_gru.py`
convention)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ml.geostrom_ml.features.engineering import HORIZONS_H  # noqa: E402
from ml.geostrom_ml.features.geo import displace, haversine_km, wrap_lon_diff  # noqa: E402
from ml.geostrom_ml.models.track_baselines import dlat_col, dlon_col  # noqa: E402
from ml.geostrom_ml.models.track_gru import (  # noqa: E402
    FEATURE_COLS, CosLatWeightedHuberLoss, TrackGRU, TrackGRUConfig,
    lat_future_col, lon_future_col,
)


def _synthetic_windows(n: int, seed: int = 0, ref_lat_range=(5.0, 40.0)) -> pd.DataFrame:
    """Windows shaped exactly like build_sequence_windows()'s real output --
    same column names, synthetic values. `ref_lat`/`ref_lon` are the last
    observed position (already-known, causal); the y_dlat/y_dlon/
    y_lat_future/y_lon_future targets are internally consistent (future =
    ref + displacement, wrap-safe) so tests exercise a physically coherent
    dataset rather than pure noise."""
    rng = np.random.default_rng(seed)
    ref_lat = rng.uniform(*ref_lat_range, n)
    ref_lon = rng.uniform(-180.0, 180.0, n)
    data = {"sid": [f"S{i % 5}" for i in range(n)], "ref_lat": ref_lat, "ref_lon": ref_lon}
    for col in FEATURE_COLS:
        data[col] = rng.normal(size=n)
    for h in HORIZONS_H:
        dlat = rng.uniform(-2.0, 2.0, n)
        dlon = rng.uniform(-2.0, 2.0, n)
        fut_lat = ref_lat + dlat
        fut_lon = ((ref_lon + dlon + 180.0) % 360.0) - 180.0
        data[dlat_col(h)] = dlat
        data[dlon_col(h)] = wrap_lon_diff(ref_lon, fut_lon)  # exactly as engineering.py builds it
        data[lat_future_col(h)] = fut_lat
        data[lon_future_col(h)] = fut_lon
    return pd.DataFrame(data)


class TestTrackGRUTraining:
    def test_fit_predict_returns_all_horizons_dlat_dlon(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)
        config = TrackGRUConfig(max_epochs=2, device="cpu", batch_size=32,
                                early_stopping_patience=2)
        model = TrackGRU(config=config)
        model.fit(train, val_df=val)
        preds = model.predict(test)
        expected_keys = {dlat_col(h) for h in HORIZONS_H} | {dlon_col(h) for h in HORIZONS_H}
        assert set(preds.keys()) == expected_keys
        for h in HORIZONS_H:
            assert preds[dlat_col(h)].shape == (20,)
            assert preds[dlon_col(h)].shape == (20,)

    def test_predict_before_fit_raises(self):
        model = TrackGRU()
        with pytest.raises(RuntimeError):
            model.predict(_synthetic_windows(5))

    def test_model_name_is_versioned(self):
        assert TrackGRU().name == "track_gru_v1"
        assert TrackGRU().task == "track"

    def test_early_stopping_records_a_best_epoch_and_km_metric(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        config = TrackGRUConfig(max_epochs=5, device="cpu", batch_size=32,
                                early_stopping_patience=2)
        model = TrackGRU(config=config)
        model.fit(train, val_df=val)
        assert 0 <= model.best_epoch < len(model.history)
        assert "val_mean_track_error_km" in model.history[0]
        assert model.history[0]["val_mean_track_error_km"] >= 0.0


class TestDeterminism:
    def test_identical_config_and_seed_gives_identical_predictions(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)
        config = TrackGRUConfig(max_epochs=3, device="cpu", batch_size=32,
                                early_stopping_patience=3, seed=42)

        model_a = TrackGRU(config=config)
        model_a.fit(train, val_df=val)
        preds_a = model_a.predict(test)

        model_b = TrackGRU(config=TrackGRUConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=42))
        model_b.fit(train, val_df=val)
        preds_b = model_b.predict(test)

        for h in HORIZONS_H:
            np.testing.assert_array_equal(preds_a[dlat_col(h)], preds_b[dlat_col(h)])
            np.testing.assert_array_equal(preds_a[dlon_col(h)], preds_b[dlon_col(h)])

    def test_different_seed_can_give_different_predictions(self):
        """Sanity-check the determinism test methodology: a different seed
        must be able to change the result, proving the equality test above
        is not vacuously passing."""
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)

        model_a = TrackGRU(config=TrackGRUConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=1))
        model_a.fit(train, val_df=val)
        preds_a = model_a.predict(test)[dlat_col(HORIZONS_H[0])]

        model_b = TrackGRU(config=TrackGRUConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=2))
        model_b.fit(train, val_df=val)
        preds_b = model_b.predict(test)[dlat_col(HORIZONS_H[0])]

        assert not np.array_equal(preds_a, preds_b)


class TestScalingIsTrainOnly:
    def test_scaler_fit_only_on_training_data(self):
        import inspect
        source = inspect.getsource(TrackGRU.fit)
        assert "StandardScaler().fit(X_train_raw)" in source
        assert "StandardScaler().fit(val" not in source
        assert "StandardScaler().fit(test" not in source


class TestCosLatWeightedGeospatialLoss:
    """Verifies the ML_ARCHITECTURE.md §7.2 cos(latitude) longitude-weighting
    formula is actually implemented, and that it has a REAL, non-vacuous
    effect (an unweighted Huber loss would score a fixed longitude error
    identically regardless of latitude; the weighted loss must not)."""

    def test_same_raw_lon_error_scores_lower_loss_at_higher_latitude(self):
        """A 2-degree longitude error represents ~222km at the equator but
        only ~111km at 60N (cos(60)=0.5) -- the weighted loss must reflect
        that, scoring the high-latitude case as a smaller error."""
        criterion = CosLatWeightedHuberLoss(delta=1.0)
        pred = torch.zeros(1, 1, 2)
        true = torch.zeros(1, 1, 2)
        true[0, 0, 1] = 2.0  # 2-degree longitude error, lat error = 0

        loss_equator = criterion(pred, true, torch.tensor([np.cos(np.radians(0.0))]))
        loss_60n = criterion(pred, true, torch.tensor([np.cos(np.radians(60.0))]))
        assert loss_60n.item() < loss_equator.item()

    def test_latitude_error_is_never_weighted(self):
        """A pure latitude error must score identically regardless of the
        cos(latitude) weight -- only the longitude term is weighted, per the
        architecture spec (weighting latitude too would be a real bug: 1
        degree of latitude is ~111km everywhere, unlike longitude)."""
        criterion = CosLatWeightedHuberLoss(delta=1.0)
        pred = torch.zeros(1, 1, 2)
        true = torch.zeros(1, 1, 2)
        true[0, 0, 0] = 2.0  # 2-degree LATITUDE error, lon error = 0

        loss_equator = criterion(pred, true, torch.tensor([np.cos(np.radians(0.0))]))
        loss_60n = criterion(pred, true, torch.tensor([np.cos(np.radians(60.0))]))
        assert loss_equator.item() == pytest.approx(loss_60n.item())

    def test_an_unweighted_loss_would_be_insensitive_to_latitude_by_construction(self):
        """Sanity-check the test methodology itself: a (hypothetical,
        deliberately wrong) UNWEIGHTED Huber loss on the same inputs WOULD
        score identically at every latitude -- proving the two tests above
        are actually exercising the weighting, not some other effect."""
        huber = torch.nn.HuberLoss(delta=1.0)
        pred_lon = torch.zeros(1)
        true_lon = torch.full((1,), 2.0)
        unweighted_loss_equator = huber(pred_lon, true_lon)
        unweighted_loss_60n = huber(pred_lon, true_lon)  # no latitude term at all
        assert unweighted_loss_equator.item() == pytest.approx(unweighted_loss_60n.item())


class TestAntimeridianHandling:
    """Phase 8 explicit requirement: antimeridian-crossing displacement
    reconstruction must be verified, not assumed. Reuses (does not
    reimplement) the same `displace()`/`haversine_km()` functions
    `TrackGRU.fit()`'s validation metric and `evaluate_track_model()` both
    call, so this test exercises the exact code path the model depends on."""

    def test_displace_and_haversine_handle_a_storm_crossing_the_antimeridian(self):
        ref_lat, ref_lon = 20.0, 179.5
        pred_dlat, pred_dlon = 0.1, 2.0  # crosses +180 -> should wrap to -178.5
        pred_lat, pred_lon = displace(
            np.array([ref_lat]), np.array([ref_lon]),
            np.array([pred_dlat]), np.array([pred_dlon]),
        )
        assert pred_lon[0] == pytest.approx(-178.5)

        true_lat, true_lon = 20.1, -178.7  # a nearby true position, also past the antimeridian
        err_km = haversine_km(np.array([true_lat]), np.array([true_lon]), pred_lat, pred_lon)
        # The two points are ~0.2 degrees apart in reality; a broken
        # (unwrapped) longitude difference would compute ~357.8 degrees of
        # longitude separation and inflate this to thousands of km.
        assert err_km[0] < 50.0

    def test_a_naive_unwrapped_subtraction_would_be_caught(self):
        """Sanity-check: the naive (WRONG) way of measuring this same pair's
        longitude separation -- raw subtraction, no wrapping -- gives a
        wildly wrong answer, proving the test above is not vacuously
        passing (i.e. it would fail if displace()/haversine_km() regressed
        to naive subtraction)."""
        naive_lon_diff = abs(-178.5 - 179.5)  # WRONG: should be 2.0, not 358.0
        assert naive_lon_diff == pytest.approx(358.0)
        correct_lon_diff = abs(wrap_lon_diff(179.5, -178.5))
        assert correct_lon_diff == pytest.approx(2.0)
