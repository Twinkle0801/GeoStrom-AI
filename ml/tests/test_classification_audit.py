"""Scene-label audit: sample-level AND storm-level statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.classification.audit import MIN_STORMS_FOR_GENERALIZATION, build_scene_audit


@pytest.fixture
def synthetic_index() -> pd.DataFrame:
    # Class "Rare" has many SAMPLES (5) but only ONE storm -- the exact
    # trap the task warns against ("do not claim adequate representation
    # merely because it has many image samples from one or two storms").
    rows = []
    for i in range(5):
        rows.append({"sample_id": f"r{i}", "storm_id": "STORM_X", "season": 2001,
                     "split": "train", "scene_label": "Rare",
                     "satellite_timestamp": pd.Timestamp("2001-01-01") + pd.Timedelta(hours=i)})
    for storm, split in [("A", "train"), ("B", "train"), ("C", "val"), ("D", "test")]:
        rows.append({"sample_id": f"c_{storm}", "storm_id": storm, "season": 2002,
                     "split": split, "scene_label": "Common",
                     "satellite_timestamp": pd.Timestamp("2002-06-01")})
    return pd.DataFrame(rows)


class TestSampleVsStormLevel:
    def test_rare_class_has_high_sample_count_but_one_storm(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert audit["class_counts"]["Rare"] == 5
        assert audit["unique_storms_per_class"]["Rare"] == 1

    def test_rare_class_is_flagged_below_generalization_threshold(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert "Rare" in audit["classes_below_generalization_storm_threshold"]
        assert "Common" not in audit["classes_below_generalization_storm_threshold"]

    def test_threshold_is_explicit_and_documented(self):
        assert MIN_STORMS_FOR_GENERALIZATION >= 1


class TestSplitCoverage:
    def test_rare_class_present_in_only_one_split_is_flagged(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert "Rare" in audit["classes_present_in_only_one_split"]

    def test_zero_test_and_val_detection(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert "Rare" in audit["classes_with_zero_test_samples"]
        assert "Rare" in audit["classes_with_zero_val_samples"]
        assert "Common" not in audit["classes_with_zero_test_samples"]


class TestBasicCounts:
    def test_totals(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert audit["total_samples"] == 9
        assert audit["total_storms"] == 5
        assert audit["total_seasons"] == 2

    def test_no_missing_labels_or_duplicates_in_clean_fixture(self, synthetic_index):
        audit = build_scene_audit(synthetic_index)
        assert audit["missing_labels"] == 0
        assert audit["duplicate_sample_id_label_pairs"] == 0

    def test_missing_label_is_counted(self, synthetic_index):
        df = synthetic_index.copy()
        df.loc[0, "scene_label"] = None
        audit = build_scene_audit(df)
        assert audit["missing_labels"] == 1
