"""Phase 7: IntensityGRU correctness, determinism, and the sequence-reshape
transform, using small synthetic data (fast, no dependency on the real
materialised dataset -- mirrors the existing Phase 2 test convention of
`ml/tests/conftest.py`'s synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ml.geostrom_ml.features.engineering import (  # noqa: E402
    HORIZONS_H, L_STEPS, PER_TIMESTEP_FEATURES, flattened_feature_columns,
)
from ml.geostrom_ml.models.intensity_gru import (  # noqa: E402
    FEATURE_COLS, GRUIntensityConfig, IntensityGRU, delta_target_col,
    reshape_to_sequence, set_deterministic, target_col,
)


def _synthetic_windows(n: int, seed: int = 0) -> pd.DataFrame:
    """Windows shaped exactly like build_sequence_windows()'s real output --
    same column names, synthetic values."""
    rng = np.random.default_rng(seed)
    data = {"sid": [f"S{i % 5}" for i in range(n)], "ref_wind": rng.uniform(20, 100, n)}
    for col in FEATURE_COLS:
        data[col] = rng.normal(size=n)
    for h in HORIZONS_H:
        data[target_col(h)] = rng.uniform(20, 100, n)
        data[delta_target_col(h)] = rng.uniform(-20, 20, n)
    return pd.DataFrame(data)


class TestReshapeToSequence:
    def test_output_shape(self):
        df = _synthetic_windows(10)
        seq = reshape_to_sequence(df, L=L_STEPS)
        assert seq.shape == (10, L_STEPS, len(PER_TIMESTEP_FEATURES))

    def test_lag0_maps_to_last_sequence_step(self):
        """lag0 = t_ref = most recent -> must land at sequence index L-1."""
        df = _synthetic_windows(1)
        feat = PER_TIMESTEP_FEATURES[0]
        df[f"x__{feat}__lag0"] = 999.0
        seq = reshape_to_sequence(df, L=L_STEPS)
        assert seq[0, L_STEPS - 1, 0] == pytest.approx(999.0)

    def test_oldest_lag_maps_to_first_sequence_step(self):
        """lag(L-1) = oldest -> must land at sequence index 0."""
        df = _synthetic_windows(1)
        feat = PER_TIMESTEP_FEATURES[0]
        df[f"x__{feat}__lag{L_STEPS - 1}"] = -999.0
        seq = reshape_to_sequence(df, L=L_STEPS)
        assert seq[0, 0, 0] == pytest.approx(-999.0)

    def test_feature_order_matches_per_timestep_features(self):
        df = _synthetic_windows(1)
        for i, feat in enumerate(PER_TIMESTEP_FEATURES):
            df[f"x__{feat}__lag3"] = float(i * 10)
        seq = reshape_to_sequence(df, L=L_STEPS)
        seq_idx = L_STEPS - 1 - 3
        for i in range(len(PER_TIMESTEP_FEATURES)):
            assert seq[0, seq_idx, i] == pytest.approx(i * 10.0)


class TestIntensityGRUTraining:
    def test_fit_predict_absolute_returns_all_horizons(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)
        config = GRUIntensityConfig(max_epochs=2, device="cpu", batch_size=32,
                                    early_stopping_patience=2)
        model = IntensityGRU(target_mode="absolute", config=config)
        model.fit(train, val_df=val)
        preds = model.predict(test)
        assert set(preds.keys()) == {target_col(h) for h in HORIZONS_H}
        for h in HORIZONS_H:
            assert preds[target_col(h)].shape == (20,)

    def test_delta_model_predict_returns_absolute_scale(self):
        """predict() must reconstruct absolute wind (ref_wind + delta) for
        the delta-target variant, per docs/ML_ARCHITECTURE.md §6.4 -- not
        raw deltas."""
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)
        config = GRUIntensityConfig(max_epochs=2, device="cpu", batch_size=32,
                                    early_stopping_patience=2)
        model = IntensityGRU(target_mode="delta", config=config)
        model.fit(train, val_df=val)
        abs_preds = model.predict(test)
        delta_preds = model.predict_delta(test)
        h = HORIZONS_H[0]
        # abs = ref_wind + delta, reconstructed -- not equal to the raw delta itself
        reconstructed = test["ref_wind"].to_numpy() + delta_preds[delta_target_col(h)]
        assert np.allclose(abs_preds[target_col(h)], reconstructed)

    def test_predict_before_fit_raises(self):
        model = IntensityGRU(target_mode="absolute")
        with pytest.raises(RuntimeError):
            model.predict(_synthetic_windows(5))

    def test_invalid_target_mode_raises(self):
        with pytest.raises(ValueError):
            IntensityGRU(target_mode="nonsense")

    def test_model_names_are_versioned_and_distinct(self):
        assert IntensityGRU(target_mode="absolute").name == "intensity_gru_v1"
        assert IntensityGRU(target_mode="delta").name == "intensity_gru_delta_v1"

    def test_early_stopping_records_a_best_epoch(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        config = GRUIntensityConfig(max_epochs=5, device="cpu", batch_size=32,
                                    early_stopping_patience=2)
        model = IntensityGRU(target_mode="absolute", config=config)
        model.fit(train, val_df=val)
        assert 0 <= model.best_epoch < len(model.history)
        assert "val_mae_kt" in model.history[0]


class TestDeterminism:
    def test_set_deterministic_does_not_raise(self):
        set_deterministic(42)

    def test_identical_config_and_seed_gives_identical_predictions(self):
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)
        config = GRUIntensityConfig(max_epochs=3, device="cpu", batch_size=32,
                                    early_stopping_patience=3, seed=42)

        model_a = IntensityGRU(target_mode="absolute", config=config)
        model_a.fit(train, val_df=val)
        preds_a = model_a.predict(test)

        model_b = IntensityGRU(target_mode="absolute", config=GRUIntensityConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=42))
        model_b.fit(train, val_df=val)
        preds_b = model_b.predict(test)

        for h in HORIZONS_H:
            np.testing.assert_array_equal(preds_a[target_col(h)], preds_b[target_col(h)])

    def test_different_seed_can_give_different_predictions(self):
        """Sanity-check the determinism test methodology: a different seed
        must be able to change the result, proving the equality test above
        is not vacuously passing."""
        train = _synthetic_windows(120, seed=1)
        val = _synthetic_windows(30, seed=2)
        test = _synthetic_windows(20, seed=3)

        model_a = IntensityGRU(target_mode="absolute", config=GRUIntensityConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=1))
        model_a.fit(train, val_df=val)
        preds_a = model_a.predict(test)[target_col(HORIZONS_H[0])]

        model_b = IntensityGRU(target_mode="absolute", config=GRUIntensityConfig(
            max_epochs=3, device="cpu", batch_size=32, early_stopping_patience=3, seed=2))
        model_b.fit(train, val_df=val)
        preds_b = model_b.predict(test)[target_col(HORIZONS_H[0])]

        assert not np.array_equal(preds_a, preds_b)


class TestScalingIsTrainOnly:
    def test_scaler_fit_only_on_training_data(self):
        """Structural proof: fit()'s scaler-fitting call only ever sees
        train_df -- val/test are transformed with the already-fitted
        scaler, never refit."""
        import inspect
        source = inspect.getsource(IntensityGRU.fit)
        # the only "StandardScaler().fit(" call in fit() must operate on a
        # training-derived variable, not val_df/test_df
        assert "StandardScaler().fit(X_train_raw)" in source
        assert "StandardScaler().fit(val" not in source
        assert "StandardScaler().fit(test" not in source
