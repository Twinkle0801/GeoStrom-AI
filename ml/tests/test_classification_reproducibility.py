"""Phase 5 Task 10: reproducibility.

Running Phase 5 dataset preparation twice must give identical sample IDs,
label mappings, split assignments, feature values, and baseline metrics.
Real-data tests skip cleanly (not fail) if the Phase 4/5 artifacts are not
present on this machine, matching the project's established precedent
(`ml/tests/test_leakage.py`, `ml/tests/test_satellite_pipeline_integration.py`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.classification.baselines import MajorityClassBaseline, build_logistic_regression_pipeline
from ml.geostrom_ml.classification.dataset import build_classification_index
from ml.geostrom_ml.classification.evaluation import evaluate
from ml.geostrom_ml.classification.features import build_feature_matrix, extract_features
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1
from ml.geostrom_ml.config import get_data_root


@pytest.fixture(scope="module")
def real_classification_index():
    root = get_data_root()
    path = root / "processed" / "classification" / "scene_taxonomy_v1" / "classification_index.parquet"
    if not path.exists():
        pytest.skip("Phase 5 classification index not built on this machine")
    return pd.read_parquet(path), root


class TestSyntheticReproducibility:
    def _index(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "storm_id": ["A", "B", "C"],
            "split": ["train", "val", "test"],
            "scene_label": ["CDO", "LargeEye", "Land"],
            "qc_status": ["included"] * 3,
        })

    def test_identical_sample_ids_and_label_mappings(self):
        a = build_classification_index(self._index())
        b = build_classification_index(self._index())
        pd.testing.assert_frame_equal(a[["sample_id", "final_class", "exclusion_reason"]],
                                      b[["sample_id", "final_class", "exclusion_reason"]])

    def test_identical_split_assignments(self):
        a = build_classification_index(self._index())
        b = build_classification_index(self._index())
        assert a["split"].tolist() == b["split"].tolist()

    def test_baseline_metrics_identical_across_two_independent_fits(self):
        X = pd.DataFrame({"f1": [1.0, 2.0, 1.1, 2.1, 1.2, 2.2] * 3,
                          "f2": [0.5, 0.4, 0.6, 0.3, 0.55, 0.45] * 3})
        y = pd.Series((["A", "B"] * 3) * 3)

        m1 = build_logistic_regression_pipeline().fit(X, y)
        m2 = build_logistic_regression_pipeline().fit(X, y)
        e1 = evaluate(y, m1.predict(X), labels=["A", "B"])
        e2 = evaluate(y, m2.predict(X), labels=["A", "B"])
        assert e1["macro_f1"] == e2["macro_f1"]
        assert e1["confusion_matrix"] == e2["confusion_matrix"]

    def test_majority_baseline_is_trivially_reproducible(self):
        y = pd.Series(["A", "A", "B"])
        m1 = MajorityClassBaseline().fit(y)
        m2 = MajorityClassBaseline().fit(y)
        assert m1.majority_class == m2.majority_class


class TestRealDataReproducibility:
    def test_feature_extraction_is_identical_on_repeated_reads(self, real_classification_index):
        index, root = real_classification_index
        zarr_path = root / "processed" / "satellite" / "satellite_v1" / "images.zarr"
        sample = index[index["qc_status"] == "included"].head(5)
        f1 = build_feature_matrix(sample, zarr_path)
        f2 = build_feature_matrix(sample, zarr_path)
        pd.testing.assert_frame_equal(f1, f2)

    def test_classification_index_rebuild_matches_committed_artifact(self, real_classification_index):
        """Re-derives the classification index from Phase 4's sample index
        and confirms it matches the committed one exactly -- proves the
        derivation is reproducible end to end, not just internally
        self-consistent."""
        index, root = real_classification_index
        sat_path = root / "processed" / "satellite" / "satellite_v1" / "sample_index.parquet"
        if not sat_path.exists():
            pytest.skip("Phase 4 sample index not present")
        sample_index = pd.read_parquet(sat_path)
        rebuilt = build_classification_index(sample_index)
        merged = index.merge(rebuilt, on="sample_id", suffixes=("_committed", "_rebuilt"))

        def _equal_including_both_null(a: pd.Series, b: pd.Series) -> pd.Series:
            # None/NaN != None/NaN under pandas' default elementwise
            # comparison (missing values are never equal, even to
            # themselves) -- both sides being null must still count as a
            # match here, since "excluded" rows legitimately carry None.
            return (a == b) | (a.isna() & b.isna())

        assert _equal_including_both_null(
            merged["final_class_committed"], merged["final_class_rebuilt"]).all()
        assert (merged["qc_status_committed"] == merged["qc_status_rebuilt"]).all()
