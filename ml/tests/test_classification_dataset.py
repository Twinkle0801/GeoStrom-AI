"""Classification dataset index: schema, traceability, no silent drops."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.classification.dataset import (
    CLASSIFICATION_COLUMNS, build_classification_index, split_summary,
)
from ml.geostrom_ml.classification.taxonomy import LABEL_VERSION


@pytest.fixture
def synthetic_sample_index() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        "storm_id": ["A", "A", "B", "B", "C"],
        "satellite_timestamp": pd.to_datetime(
            ["2001-01-01", "2001-01-02", "2001-01-03", "2001-01-04", "2001-01-05"]),
        "season": [2001, 2001, 2001, 2001, 2001],
        "split": ["train", "train", "val", "val", "test"],
        "zarr_index": [0, 1, 2, 3, 4],
        "scene_label": ["CDO", "Land", "Eye", "EmbCenter", "UnknownFutureLabel"],
        "qc_status": ["ok"] * 5,
    })


class TestSchema:
    def test_output_has_exactly_the_documented_columns(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        assert list(out.columns) == CLASSIFICATION_COLUMNS

    def test_no_rows_are_dropped(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        assert len(out) == len(synthetic_sample_index)


class TestTraceability:
    def test_original_scene_is_preserved_verbatim(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        assert out["original_scene"].tolist() == synthetic_sample_index["scene_label"].tolist()

    def test_source_dataframe_scene_label_column_is_untouched(self, synthetic_sample_index):
        original = synthetic_sample_index["scene_label"].copy()
        build_classification_index(synthetic_sample_index)
        pd.testing.assert_series_equal(synthetic_sample_index["scene_label"], original)

    def test_label_version_is_stamped_on_every_row(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        assert (out["label_version"] == LABEL_VERSION).all()


class TestQCStatusAndExclusion:
    def test_included_rows_have_no_exclusion_reason(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        included = out[out["qc_status"] == "included"]
        assert included["exclusion_reason"].isna().all()

    def test_excluded_rows_always_have_a_reason(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        excluded = out[out["qc_status"] == "excluded"]
        assert excluded["exclusion_reason"].notna().all()
        assert len(excluded) == 3  # Land, EmbCenter, UnknownFutureLabel

    def test_unresolved_label_is_excluded_not_silently_classified(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        row = out[out["sample_id"] == "s5"].iloc[0]
        assert row["qc_status"] == "excluded"
        assert row["exclusion_reason"] == "unresolved_mapping"
        assert row["final_class"] is None


class TestSplitSummary:
    def test_counts_only_included_rows(self, synthetic_sample_index):
        out = build_classification_index(synthetic_sample_index)
        summary = split_summary(out)
        assert summary["total_included"] == 2  # s1 (CDO/train), s3 (Eye/val)
        assert summary["total_excluded"] == 3
        assert summary["samples_by_split"] == {"train": 1, "val": 1}
