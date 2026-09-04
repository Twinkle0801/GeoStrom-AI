"""Phase 6 dataset: resize, invalid-pixel fill, and SceneImageDataset --
none of this requires torch to be installed (see module docstring in
`ml/geostrom_ml/classification/deep/dataset.py` for why)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.classification.deep.config import MODEL_INPUT_SIZE, NATIVE_GRID
from ml.geostrom_ml.classification.deep.dataset import (
    SceneImageDataset, fill_invalid_with_image_mean, resize_deterministic,
)
from ml.geostrom_ml.satellite.imagery import SatelliteZarrStore


@pytest.fixture
def synthetic_zarr(tmp_path):
    store = SatelliteZarrStore(tmp_path / "images.zarr").create(4, overwrite=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        kelvin = rng.uniform(220, 300, size=NATIVE_GRID).astype("float32")
        mask = np.ones(NATIVE_GRID, dtype=bool)
        if i == 3:
            mask[:10, :10] = False  # a few invalid pixels in one frame
        store.write_frame(i, kelvin, mask)
    return tmp_path / "images.zarr"


@pytest.fixture
def synthetic_index() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["s0", "s1", "s2", "s3"],
        "storm_id": ["A", "A", "B", "C"],
        "split": ["train", "train", "val", "test"],
        "zarr_index": [0, 1, 2, 3],
        "final_class": ["CDO", "Shear", "Eye", "CurvedBand"],
        "qc_status": ["included"] * 4,
    })


class TestResize:
    def test_output_shape_is_model_input_size(self):
        img = np.random.default_rng(0).uniform(200, 300, size=NATIVE_GRID).astype("float32")
        out = resize_deterministic(img, MODEL_INPUT_SIZE)
        assert out.shape == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    def test_deterministic_same_input_same_output(self):
        img = np.random.default_rng(1).uniform(200, 300, size=NATIVE_GRID).astype("float32")
        a = resize_deterministic(img)
        b = resize_deterministic(img)
        assert np.array_equal(a, b)

    def test_uniform_field_stays_uniform_after_resize(self):
        img = np.full(NATIVE_GRID, 260.0, dtype="float32")
        out = resize_deterministic(img)
        assert np.allclose(out, 260.0, atol=0.01)


class TestInvalidPixelFill:
    def test_all_valid_frame_is_unchanged(self):
        kelvin = np.full((5, 5), 250.0, dtype="float32")
        mask = np.ones((5, 5), dtype=bool)
        out = fill_invalid_with_image_mean(kelvin, mask)
        assert np.array_equal(out, kelvin)

    def test_invalid_pixels_filled_with_the_valid_mean_not_zero(self):
        kelvin = np.array([[100.0, 300.0], [300.0, 300.0]], dtype="float32")
        mask = np.array([[False, True], [True, True]])
        out = fill_invalid_with_image_mean(kelvin, mask)
        assert out[0, 0] == pytest.approx(300.0)  # mean of the 3 valid=300 pixels
        assert out[0, 0] != 0.0

    def test_fill_value_never_fabricates_an_out_of_range_temperature(self):
        kelvin = np.full((4, 4), 220.0, dtype="float32")
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 0] = False
        out = fill_invalid_with_image_mean(kelvin, mask)
        assert 150.0 <= out[0, 0] <= 350.0  # IRWIN physical range, Phase 1/4 locked

    def test_degenerate_all_invalid_frame_does_not_crash(self):
        kelvin = np.full((3, 3), 250.0, dtype="float32")
        mask = np.zeros((3, 3), dtype=bool)
        out = fill_invalid_with_image_mean(kelvin, mask)
        assert out.shape == (3, 3)


@pytest.fixture
def zarr_with_invalid_train_pixels(tmp_path):
    """Regression fixture for the real NaN-bleeding bug found in Phase 6
    development: a TRAIN-split frame with invalid (NaN-backed) pixels,
    exactly the condition that triggered NaN training loss before the
    fill-before-augment ordering fix (see dataset.py's module docstring)."""
    store = SatelliteZarrStore(tmp_path / "images.zarr").create(1, overwrite=True)
    rng = np.random.default_rng(0)
    kelvin = rng.uniform(220, 300, size=NATIVE_GRID).astype("float32")
    mask = np.ones(NATIVE_GRID, dtype=bool)
    mask[100:120, 100:120] = False  # a real-sized invalid patch, not just 1 pixel
    store.write_frame(0, kelvin, mask)
    index = pd.DataFrame({
        "sample_id": ["train0"], "storm_id": ["A"], "split": ["train"],
        "zarr_index": [0], "final_class": ["CDO"], "qc_status": ["included"],
    })
    return index, tmp_path / "images.zarr"


class TestNoNaNBleedRegression:
    """Regression test for the real bug: invalid pixels must be filled
    BEFORE rotation augmentation, or bilinear interpolation spreads NaN
    into neighbouring pixels (found via a real training run producing NaN
    loss from epoch 0 -- see dataset.py's module docstring)."""

    def test_augmented_frame_with_invalid_pixels_has_no_nan(self, zarr_with_invalid_train_pixels):
        index, zarr_path = zarr_with_invalid_train_pixels
        ds = SceneImageDataset(index, zarr_path, split="train", augment=True, seed=42)
        for i in range(len(ds)):
            image, _, _ = ds[i]
            assert np.isfinite(image).all(), "NaN/Inf leaked through augmentation+resize"

    def test_normalized_augmented_frame_with_invalid_pixels_has_no_nan(self, zarr_with_invalid_train_pixels):
        """Same check with normalization applied -- the exact path the real
        training script uses."""
        index, zarr_path = zarr_with_invalid_train_pixels
        ds = SceneImageDataset(index, zarr_path, split="train", augment=True, seed=42,
                               train_mean=260.0, train_std=20.0)
        for i in range(len(ds)):
            image, _, _ = ds[i]
            assert np.isfinite(image).all()


class TestSceneImageDataset:
    def test_length_matches_included_rows_in_the_requested_split(self, synthetic_index, synthetic_zarr):
        ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        assert len(ds) == 2  # s0, s1

    def test_excludes_other_splits(self, synthetic_index, synthetic_zarr):
        train_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        val_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="val")
        test_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="test")
        assert len(train_ds) + len(val_ds) + len(test_ds) == 4

    def test_getitem_returns_image_label_sample_id(self, synthetic_index, synthetic_zarr):
        ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        image, label_idx, sample_id = ds[0]
        assert image.shape == (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        assert isinstance(label_idx, int)
        assert sample_id in ("s0", "s1")

    def test_excludes_excluded_rows(self, synthetic_index, synthetic_zarr):
        idx = synthetic_index.copy()
        idx.loc[0, "qc_status"] = "excluded"
        ds = SceneImageDataset(idx, synthetic_zarr, split="train")
        assert len(ds) == 1

    def test_normalization_applied_when_stats_given(self, synthetic_index, synthetic_zarr):
        ds_raw = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        ds_norm = SceneImageDataset(synthetic_index, synthetic_zarr, split="train",
                                    train_mean=260.0, train_std=10.0)
        raw_img, _, _ = ds_raw[0]
        norm_img, _, _ = ds_norm[0]
        assert not np.allclose(raw_img, norm_img)
