"""Evaluation metrics: macro-F1 as primary metric, absent-class handling."""

from __future__ import annotations

from ml.geostrom_ml.classification.evaluation import evaluate, macro_f1_over_present_classes


class TestPerfectPredictions:
    def test_perfect_predictions_score_1_everywhere(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        m = evaluate(y_true, y_pred, labels=["A", "B"])
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["weighted_f1"] == 1.0


class TestAbsentClassHandling:
    def test_class_absent_from_y_true_is_reported_not_hidden(self):
        y_true = ["A", "A", "B"]
        y_pred = ["A", "A", "B"]
        m = evaluate(y_true, y_pred, labels=["A", "B", "C"])
        assert "C" in m["classes_absent_from_this_split"]
        assert m["per_class"]["C"]["support"] == 0
        assert m["n_classes"] == 3

    def test_macro_f1_over_present_classes_excludes_absent_class(self):
        y_true = ["A", "A", "B"]
        y_pred = ["A", "A", "A"]  # B misclassified
        f1_full = evaluate(y_true, y_pred, labels=["A", "B", "C"])["macro_f1"]
        f1_present, present = macro_f1_over_present_classes(y_true, y_pred)
        assert present == ["A", "B"]
        assert f1_present != f1_full  # different denominators, both reported


class TestMajorityBaselineIsPoorOnMacroF1:
    def test_predicting_only_the_majority_class_scores_low_macro_f1(self):
        y_true = ["A"] * 8 + ["B"] * 1 + ["C"] * 1
        y_pred = ["A"] * 10
        m = evaluate(y_true, y_pred, labels=["A", "B", "C"])
        # accuracy is high (80%) but macro-F1 must expose the failure on B/C
        assert m["accuracy"] == 0.8
        assert m["macro_f1"] < 0.4


class TestConfusionMatrix:
    def test_confusion_matrix_shape_matches_label_count(self):
        m = evaluate(["A", "B"], ["A", "A"], labels=["A", "B"])
        assert len(m["confusion_matrix"]) == 2
        assert len(m["confusion_matrix"][0]) == 2

    def test_confusion_matrix_diagonal_for_perfect_predictions(self):
        m = evaluate(["A", "B", "A"], ["A", "B", "A"], labels=["A", "B"])
        cm = m["confusion_matrix"]
        assert cm[0][1] == 0
        assert cm[1][0] == 0


class TestPerClassMetricsPresent:
    def test_per_class_dict_has_precision_recall_f1_support(self):
        m = evaluate(["A", "B"], ["A", "B"], labels=["A", "B"])
        for label in ("A", "B"):
            for key in ("precision", "recall", "f1", "support"):
                assert key in m["per_class"][label]
