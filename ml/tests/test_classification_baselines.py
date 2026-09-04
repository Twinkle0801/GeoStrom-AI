"""Non-deep-learning baselines: majority-class, logistic regression,
imbalance handling -- all with synthetic, fast, deterministic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.classification.baselines import (
    MajorityClassBaseline, RANDOM_SEED, build_logistic_regression_pipeline,
)
from ml.geostrom_ml.classification.imbalance import compute_class_weights


class TestMajorityClassBaseline:
    def test_predicts_the_most_frequent_training_class(self):
        y_train = pd.Series(["A", "A", "A", "B", "C"])
        baseline = MajorityClassBaseline().fit(y_train)
        assert baseline.majority_class == "A"

    def test_predict_returns_n_copies_of_the_majority_class(self):
        y_train = pd.Series(["A", "A", "B"])
        baseline = MajorityClassBaseline().fit(y_train)
        preds = baseline.predict(5)
        assert preds == ["A"] * 5

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MajorityClassBaseline().predict(3)

    def test_majority_is_computed_from_training_labels_only(self):
        """The baseline object never receives val/test labels at all --
        structural proof it cannot use them to pick its answer."""
        import inspect
        sig = inspect.signature(MajorityClassBaseline.fit)
        assert list(sig.parameters) == ["self", "y_train"]


class TestClassWeights:
    def test_balanced_weights_favour_the_rarer_class(self):
        y_train = pd.Series(["A"] * 90 + ["B"] * 10)
        weights = compute_class_weights(y_train)
        assert weights["B"] > weights["A"]

    def test_weights_computed_only_from_the_given_series(self):
        """No split parameter exists -- the function cannot silently reach
        into val/test data; the caller must pass training labels only."""
        import inspect
        assert list(inspect.signature(compute_class_weights).parameters) == ["train_labels"]

    def test_equal_class_counts_give_equal_weights(self):
        y_train = pd.Series(["A", "A", "B", "B"])
        weights = compute_class_weights(y_train)
        assert weights["A"] == pytest.approx(weights["B"])


class TestLogisticRegressionReproducibility:
    def test_seed_is_fixed_and_documented(self):
        assert RANDOM_SEED == 42

    def test_identical_fit_gives_identical_predictions(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["f1", "f2", "f3", "f4"])
        y = pd.Series((["A"] * 30) + (["B"] * 30))

        m1 = build_logistic_regression_pipeline(class_weight="balanced").fit(X, y)
        m2 = build_logistic_regression_pipeline(class_weight="balanced").fit(X, y)
        assert list(m1.predict(X)) == list(m2.predict(X))

    def test_pipeline_imputes_missing_values_without_error(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f1", "f2", "f3"])
        X.iloc[0, 0] = float("nan")
        y = pd.Series((["A"] * 20) + (["B"] * 20))
        model = build_logistic_regression_pipeline().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
