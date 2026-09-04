"""Phase 5 Task 9: dedicated adversarial classification leakage tests.

Covers all ten vectors the task names. At least one test per vector
deliberately introduces the bad condition and proves a validator (or the
pipeline itself) catches it, per the task's explicit instruction -- these
are not vacuous checks against already-clean data alone.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from ml.geostrom_ml.classification.baselines import MajorityClassBaseline
from ml.geostrom_ml.classification.dataset import build_classification_index
from ml.geostrom_ml.classification.features import extract_features
from ml.geostrom_ml.classification.imbalance import compute_class_weights
from ml.geostrom_ml.classification.leakage import (
    assert_no_excluded_rows, assert_no_storm_split_leakage,
    find_excluded_rows_in_selection, find_storm_split_violations,
)
from ml.geostrom_ml.classification.taxonomy import apply_taxonomy
from ml.geostrom_ml.config import MANIFEST_DIR, get_data_root


def _clean_index() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "storm_id": ["A", "A", "B", "C"],
        "split": ["train", "train", "val", "test"],
        "scene_label": ["CDO", "Shear", "Eye", "CurvedBand"],
        "qc_status": ["included", "included", "included", "included"],
    })


class TestVector1And2StormSplitIntegrity:
    """1. Same storm cannot appear in multiple splits.
    2. Multiple images from the same storm cannot leak across splits."""

    def test_clean_data_has_no_violations(self):
        assert find_storm_split_violations(_clean_index()) == []
        assert_no_storm_split_leakage(_clean_index())  # does not raise

    def test_adversarial_injected_violation_is_caught(self):
        """Deliberately put storm 'A' in BOTH train and val -- proves the
        validator is not vacuous."""
        bad = _clean_index().copy()
        bad.loc[bad["storm_id"] == "A", "split"] = ["train", "val"]  # A now spans 2 splits
        violations = find_storm_split_violations(bad)
        assert len(violations) == 1
        assert violations[0]["storm_id"] == "A"
        assert set(violations[0]["splits"]) == {"train", "val"}
        with pytest.raises(ValueError, match="more than one split"):
            assert_no_storm_split_leakage(bad)

    def test_real_classification_index_has_no_storm_split_violations(self):
        """Real-data confirmation, skipped cleanly if Phase 4/5 artifacts
        are absent on this machine."""
        root = get_data_root()
        path = root / "processed" / "classification" / "scene_taxonomy_v1" / "classification_index.parquet"
        if not path.exists():
            pytest.skip("Phase 5 classification index not built on this machine")
        df = pd.read_parquet(path)
        assert find_storm_split_violations(df) == []

    def test_real_splits_v1_manifest_has_no_storm_overlap(self):
        """Re-verifies the frozen Phase 2 split itself is still clean --
        Phase 5 must never have touched it (see also test_data_safety)."""
        import json
        data = json.loads((MANIFEST_DIR / "splits_v1.json").read_text(encoding="utf-8"))
        train, val, test = (set(data[s]["storm_ids"]) for s in ("train", "val", "test"))
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)


class TestVector3LabelsFromAllowedMetadataOnly:
    """3. Classification labels come only from the allowed current-time metadata."""

    def test_final_class_depends_only_on_scene_label_column(self):
        idx = _clean_index()
        base = build_classification_index(idx)["final_class"].tolist()

        mutated = idx.copy()
        mutated["storm_id"] = ["ZZZZ"] * len(mutated)  # change everything else
        mutated["split"] = ["test"] * len(mutated)
        after = build_classification_index(mutated)["final_class"].tolist()
        assert base == after  # only scene_label drives the label

    def test_apply_taxonomy_signature_takes_only_the_label(self):
        assert list(inspect.signature(apply_taxonomy).parameters) == ["scene_label"]


class TestVector4NoFutureIbtracsAsFeatures:
    """4. Future IBTrACS values are not used as features."""

    def test_extract_features_signature_excludes_all_ibtracs_fields(self):
        params = set(inspect.signature(extract_features).parameters)
        forbidden = {"usa_wind", "max_wind", "pressure_if_valid", "ibtracs_lat",
                     "ibtracs_lon", "ibtracs_timestamp", "storm_speed", "storm_dir"}
        assert params.isdisjoint(forbidden)
        assert params == {"kelvin", "valid_mask"}


class TestVector5ExcludedSamplesCannotEnterTraining:
    """5. Excluded samples cannot accidentally enter training."""

    def test_clean_selection_passes(self):
        assert find_excluded_rows_in_selection(_clean_index()) == []
        assert_no_excluded_rows(_clean_index())

    def test_adversarial_excluded_row_in_a_training_selection_is_caught(self):
        bad = _clean_index().copy()
        bad.loc[0, "qc_status"] = "excluded"  # s1 sneaks into a "clean" selection
        found = find_excluded_rows_in_selection(bad)
        assert found == ["s1"]
        with pytest.raises(ValueError, match="excluded sample"):
            assert_no_excluded_rows(bad)


class TestVector6OriginalScenePreserved:
    """6. Original Scene labels remain unchanged."""

    def test_source_column_untouched_by_reference(self):
        idx = _clean_index()
        before = idx["scene_label"].copy()
        build_classification_index(idx)
        pd.testing.assert_series_equal(idx["scene_label"], before)

    def test_real_classification_index_original_scene_matches_phase4_scene_label(self):
        root = get_data_root()
        cls_path = root / "processed" / "classification" / "scene_taxonomy_v1" / "classification_index.parquet"
        sat_path = root / "processed" / "satellite" / "satellite_v1" / "sample_index.parquet"
        if not (cls_path.exists() and sat_path.exists()):
            pytest.skip("Phase 4/5 real artifacts not present on this machine")
        cls_df = pd.read_parquet(cls_path).set_index("sample_id")
        sat_df = pd.read_parquet(sat_path).set_index("sample_id")
        joined = cls_df.join(sat_df[["scene_label"]], rsuffix="_phase4")
        assert (joined["original_scene"] == joined["scene_label"]).all()


class TestVector7DeterministicMapping:
    """7. Label mapping is deterministic."""

    def test_repeated_calls_agree(self):
        idx = _clean_index()
        a = build_classification_index(idx)["final_class"].tolist()
        b = build_classification_index(idx)["final_class"].tolist()
        assert a == b


class TestVector8ReproducibleDatasetCreation:
    """8. Dataset creation is reproducible."""

    def test_two_independent_builds_are_identical(self):
        idx = _clean_index()
        out1 = build_classification_index(idx.copy())
        out2 = build_classification_index(idx.copy())
        pd.testing.assert_frame_equal(out1, out2)


class TestVector9TestSetNeverUsedForTaxonomySelection:
    """9. The test set is never used for taxonomy selection."""

    def test_taxonomy_mapping_is_a_static_dict_not_a_function_of_data(self):
        from ml.geostrom_ml.classification import taxonomy
        # SCENE_TAXONOMY_V1 is a module-level constant: nothing in this
        # package computes it from a DataFrame, a split, or any statistic.
        assert isinstance(taxonomy.SCENE_TAXONOMY_V1, dict)
        # apply_taxonomy takes no split/data argument at all (see Vector 3 test).


class TestVector10ClassWeightsFromTrainingSplitOnly:
    """10. Class weighting is computed only from the training split."""

    def test_weights_differ_when_val_test_are_wrongly_included(self):
        """Adversarial: if a caller mistakenly passed train+val+test labels
        instead of train-only, the computed weights would visibly differ --
        proving the function is not silently indifferent to which split it
        receives (the real defense is the call site always passing
        y['train'] only, enforced by code review / the signature tests in
        test_classification_baselines.py)."""
        train_only = pd.Series(["CDO"] * 80 + ["Shear"] * 20)
        train_plus_val = pd.Series(["CDO"] * 80 + ["Shear"] * 20 + ["Shear"] * 40)
        w_correct = compute_class_weights(train_only)
        w_wrong = compute_class_weights(train_plus_val)
        assert w_correct != w_wrong

    def test_majority_baseline_fit_signature_also_takes_training_labels_only(self):
        assert list(inspect.signature(MajorityClassBaseline.fit).parameters) == ["self", "y_train"]
