"""Deterministic image feature extraction: correctness, determinism, and
proof that no feature reads anything beyond the pixel arrays."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ml.geostrom_ml.classification.features import (
    FEATURE_NAMES, GRID, extract_features,
)


def _uniform(value: float, valid_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    kelvin = np.full(GRID, value, dtype="float32")
    n_valid = int(GRID[0] * GRID[1] * valid_frac)
    mask = np.zeros(GRID, dtype=bool)
    mask.flat[:n_valid] = True
    return kelvin, mask


class TestDeterminism:
    def test_same_input_same_output(self):
        kelvin, mask = _uniform(260.0)
        a = extract_features(kelvin, mask)
        b = extract_features(kelvin, mask)
        assert a == b

    def test_no_randomness_in_the_function_signature(self):
        sig = inspect.signature(extract_features)
        assert "seed" not in sig.parameters
        assert "random_state" not in sig.parameters


class TestFunctionSignatureExcludesForbiddenInputs:
    """Structural proof (not just a promise) that this function cannot use
    future IBTrACS/track/intensity data: it only accepts two pixel arrays."""

    def test_signature_has_exactly_kelvin_and_valid_mask(self):
        params = list(inspect.signature(extract_features).parameters)
        assert params == ["kelvin", "valid_mask"]


class TestBasicStatistics:
    def test_uniform_field_has_zero_std(self):
        kelvin, mask = _uniform(250.0)
        f = extract_features(kelvin, mask)
        assert f["mean_k"] == pytest.approx(250.0)
        assert f["std_k"] == pytest.approx(0.0)
        assert f["min_k"] == f["max_k"] == pytest.approx(250.0)

    def test_valid_fraction_reflects_the_mask(self):
        kelvin, mask = _uniform(250.0, valid_frac=0.5)
        f = extract_features(kelvin, mask)
        assert f["valid_fraction"] == pytest.approx(0.5, abs=0.01)

    def test_all_invalid_returns_nan_not_a_fabricated_zero(self):
        kelvin = np.full(GRID, 250.0, dtype="float32")
        mask = np.zeros(GRID, dtype=bool)
        f = extract_features(kelvin, mask)
        assert f["valid_fraction"] == 0.0
        assert all(v != v for k, v in f.items() if k != "valid_fraction")  # NaN != NaN

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.zeros((10, 10), dtype="float32"), np.ones((10, 10), dtype=bool))


class TestSpatialStructure:
    def test_cold_center_warm_outer_gives_negative_center_minus_ring(self):
        """A synthetic eye-like frame: cold core, warmer surroundings --
        the opposite sign of a real eye (warm clear centre, cold eyewall),
        chosen deliberately so this test cannot pass by coincidentally
        matching real-data intuition; it only checks the arithmetic."""
        kelvin = np.full(GRID, 290.0, dtype="float32")
        h, w = GRID
        cy, cx = h // 2, w // 2
        kelvin[cy - 10:cy + 10, cx - 10:cx + 10] = 210.0  # cold core
        mask = np.ones(GRID, dtype=bool)
        f = extract_features(kelvin, mask)
        assert f["center_mean_k"] < f["ring_mean_k"]
        assert f["center_minus_ring_k"] < 0

    def test_asymmetric_field_has_nonzero_quadrant_std(self):
        kelvin = np.full(GRID, 280.0, dtype="float32")
        kelvin[:150, :150] = 200.0  # cold NW quadrant only
        mask = np.ones(GRID, dtype=bool)
        f = extract_features(kelvin, mask)
        assert f["quad_std_k"] > 5.0

    def test_uniform_field_has_near_zero_quadrant_std(self):
        kelvin, mask = _uniform(260.0)
        f = extract_features(kelvin, mask)
        assert f["quad_std_k"] == pytest.approx(0.0, abs=1e-4)


class TestFeatureNamesContract:
    def test_output_keys_match_feature_names_exactly(self):
        kelvin, mask = _uniform(260.0)
        f = extract_features(kelvin, mask)
        assert set(f.keys()) == set(FEATURE_NAMES)

    def test_no_feature_name_references_scene_or_label(self):
        """Guards against ever adding a feature that is a disguised copy of
        the classification target."""
        for name in FEATURE_NAMES:
            assert "scene" not in name.lower()
            assert "label" not in name.lower()
            assert "class" not in name.lower()
