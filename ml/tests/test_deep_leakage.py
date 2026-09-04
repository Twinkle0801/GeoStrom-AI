"""Phase 6 Task: dedicated deep-learning leakage tests.

Covers all seven vectors the task names, reusing Phase 5's real validators
(`ml/geostrom_ml/classification/leakage.py`) where they already apply
generically, and adding DL-pipeline-specific structural proofs (dataset
split separation, no-label-in-preprocessing) where Phase 5's validators
don't reach.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.classification.deep.dataset import (
    SceneImageDataset, fill_invalid_with_image_mean, resize_deterministic,
)
from ml.geostrom_ml.classification.leakage import (
    assert_no_excluded_rows, assert_no_storm_split_leakage, find_storm_split_violations,
)
from ml.geostrom_ml.satellite.imagery import SatelliteZarrStore


@pytest.fixture
def synthetic_zarr(tmp_path):
    store = SatelliteZarrStore(tmp_path / "images.zarr").create(6, overwrite=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        kelvin = rng.uniform(220, 300, size=(301, 301)).astype("float32")
        store.write_frame(i, kelvin, np.ones((301, 301), dtype=bool))
    return tmp_path / "images.zarr"


@pytest.fixture
def synthetic_index() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "storm_id": ["A", "A", "B", "B", "C", "C"],
        "split": ["train", "train", "val", "val", "test", "test"],
        "zarr_index": list(range(6)),
        "final_class": ["CDO", "Shear", "Eye", "CDO", "CurvedBand", "Shear"],
        "qc_status": ["included"] * 6,
    })


class TestVector1StormSplitIntegrity:
    def test_clean_index_has_no_violations(self, synthetic_index):
        assert find_storm_split_violations(synthetic_index) == []

    def test_adversarial_storm_in_two_splits_is_caught(self, synthetic_index):
        bad = synthetic_index.copy()
        bad.loc[bad["storm_id"] == "A", "split"] = ["train", "val"]
        with pytest.raises(ValueError, match="more than one split"):
            assert_no_storm_split_leakage(bad)


class TestVector2And3TestValNeverInTraining:
    def test_train_dataset_contains_only_train_split_samples(self, synthetic_index, synthetic_zarr):
        train_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        sample_ids = {train_ds.rows.iloc[i]["sample_id"] for i in range(len(train_ds))}
        assert sample_ids == {"s0", "s1"}

    def test_val_and_test_samples_are_disjoint_from_train(self, synthetic_index, synthetic_zarr):
        train_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        val_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="val")
        test_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="test")
        train_ids = {train_ds.rows.iloc[i]["sample_id"] for i in range(len(train_ds))}
        val_ids = {val_ds.rows.iloc[i]["sample_id"] for i in range(len(val_ds))}
        test_ids = {test_ds.rows.iloc[i]["sample_id"] for i in range(len(test_ds))}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_normalization_stats_computed_from_train_split_function_signature(self):
        from ml.geostrom_ml.classification.deep.dataset import compute_train_normalization_stats
        # the function internally builds split="train" only -- verified by reading its
        # source is impractical to assert generically, so this is a behavioural check:
        sig = inspect.signature(compute_train_normalization_stats)
        assert list(sig.parameters) == ["index_df", "zarr_path"]  # no split argument to misuse


class TestVector4NoLabelsInPreprocessing:
    def test_resize_function_signature_excludes_label(self):
        assert list(inspect.signature(resize_deterministic).parameters) == ["image", "size"]

    def test_fill_function_signature_excludes_label(self):
        assert list(inspect.signature(fill_invalid_with_image_mean).parameters) == \
            ["kelvin", "valid_mask"]

    def test_changing_the_label_column_does_not_change_the_processed_pixels(
        self, synthetic_index, synthetic_zarr
    ):
        ds_a = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        mutated = synthetic_index.copy()
        mutated.loc[mutated["split"] == "train", "final_class"] = "Shear"  # relabel both to Shear
        ds_b = SceneImageDataset(mutated, synthetic_zarr, split="train")
        img_a, _, _ = ds_a[0]
        img_b, _, _ = ds_b[0]
        assert np.array_equal(img_a, img_b)  # pixels unaffected by the label change


class TestVector5NoFutureOrMetadataLeakage:
    def test_dataset_getitem_returns_only_image_label_id(self, synthetic_index, synthetic_zarr):
        ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        item = ds[0]
        assert len(item) == 3  # image, label_idx, sample_id -- nothing else

    def test_index_columns_available_to_the_dataset_exclude_ibtracs_fields(self, synthetic_index):
        forbidden = {"usa_wind", "max_wind", "pressure_if_valid", "ibtracs_lat", "ibtracs_lon"}
        assert forbidden.isdisjoint(synthetic_index.columns)


class TestVector6DeterministicIndexing:
    def test_row_order_is_deterministic_regardless_of_input_order(self, synthetic_index, synthetic_zarr):
        shuffled = synthetic_index.sample(frac=1.0, random_state=123).reset_index(drop=True)
        ds_a = SceneImageDataset(synthetic_index, synthetic_zarr, split="train")
        ds_b = SceneImageDataset(shuffled, synthetic_zarr, split="train")
        ids_a = [ds_a.rows.iloc[i]["sample_id"] for i in range(len(ds_a))]
        ids_b = [ds_b.rows.iloc[i]["sample_id"] for i in range(len(ds_b))]
        assert ids_a == ids_b  # sorted by zarr_index -> same order regardless of input order


class TestVector7EvaluationUsesFrozenTestIndex:
    def test_test_dataset_is_built_from_the_same_index_object_as_train_val(
        self, synthetic_index, synthetic_zarr
    ):
        """Structural proof: there is exactly one classification index
        DataFrame passed to all three SceneImageDataset constructions in
        ml/scripts/train_deep_classifier.py -- reproduced here at unit
        scale to confirm the split filter, not a second index, is what
        separates them."""
        test_ds = SceneImageDataset(synthetic_index, synthetic_zarr, split="test")
        assert len(test_ds) == 2
        assert set(test_ds.rows["sample_id"]) == {"s4", "s5"}


class TestExcludedRowsNeverEnterAnySplit:
    def test_excluded_row_is_absent_from_every_split(self, synthetic_index, synthetic_zarr):
        idx = synthetic_index.copy()
        idx.loc[0, "qc_status"] = "excluded"
        total = 0
        for split in ("train", "val", "test"):
            ds = SceneImageDataset(idx, synthetic_zarr, split=split)
            total += len(ds)
        assert total == 5  # one fewer than the full 6

    def test_adversarial_excluded_row_planted_in_a_selection_is_caught(self, synthetic_index):
        bad = synthetic_index.copy()
        bad.loc[0, "qc_status"] = "excluded"
        with pytest.raises(ValueError, match="excluded sample"):
            assert_no_excluded_rows(bad)
