"""Phase 6 augmentation: determinism, physical defensibility (no flips
implemented -- see the module docstring's chirality argument)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ml.geostrom_ml.classification.deep import augmentation
from ml.geostrom_ml.classification.deep.augmentation import (
    MAX_ROTATION_DEG, augment_train_image, rotate_reflect, sample_rotation_angle,
)


class TestNoFlipsImplemented:
    """Structural proof, not just documentation: this module provides no
    flip/mirror function at all."""

    def test_module_defines_no_flip_function(self):
        names = [n for n, _ in inspect.getmembers(augmentation, inspect.isfunction)]
        assert not any("flip" in n.lower() or "mirror" in n.lower() for n in names)


class TestDeterminism:
    def test_same_seed_same_sample_index_gives_same_angle(self):
        a = sample_rotation_angle(sample_index=5, seed=42)
        b = sample_rotation_angle(sample_index=5, seed=42)
        assert a == b

    def test_different_sample_index_gives_different_angle(self):
        a = sample_rotation_angle(sample_index=0, seed=42)
        b = sample_rotation_angle(sample_index=1, seed=42)
        assert a != b

    def test_different_seed_gives_different_angle(self):
        a = sample_rotation_angle(sample_index=0, seed=42)
        b = sample_rotation_angle(sample_index=0, seed=99)
        assert a != b

    def test_angle_within_documented_bound(self):
        for i in range(50):
            angle = sample_rotation_angle(sample_index=i, seed=42)
            assert -MAX_ROTATION_DEG <= angle <= MAX_ROTATION_DEG


class TestRotation:
    def test_zero_rotation_is_identity(self):
        img = np.random.default_rng(0).uniform(200, 300, size=(50, 50)).astype("float32")
        out = rotate_reflect(img, 0.0)
        assert np.allclose(out, img, atol=1e-3)

    def test_rotation_preserves_dtype(self):
        img = np.full((50, 50), 250.0, dtype="float32")
        out = rotate_reflect(img, 10.0)
        assert out.dtype == np.float32

    def test_uniform_field_stays_uniform_after_rotation(self):
        """No corner-fill artefact should introduce a fabricated
        temperature into an otherwise-uniform frame."""
        img = np.full((100, 100), 260.0, dtype="float32")
        out = rotate_reflect(img, 12.0)
        assert np.allclose(out, 260.0, atol=0.5)


class TestAugmentTrainImage:
    def test_kelvin_and_mask_stay_aligned_in_shape(self):
        kelvin = np.random.default_rng(0).uniform(200, 300, size=(60, 60)).astype("float32")
        mask = np.ones((60, 60), dtype=bool)
        out_k, out_m = augment_train_image(kelvin, mask, sample_index=3, seed=42)
        assert out_k.shape == kelvin.shape == out_m.shape

    def test_reproducible_across_calls(self):
        kelvin = np.random.default_rng(0).uniform(200, 300, size=(60, 60)).astype("float32")
        mask = np.ones((60, 60), dtype=bool)
        a_k, a_m = augment_train_image(kelvin, mask, sample_index=7, seed=42)
        b_k, b_m = augment_train_image(kelvin, mask, sample_index=7, seed=42)
        assert np.array_equal(a_k, b_k)
        assert np.array_equal(a_m, b_m)
