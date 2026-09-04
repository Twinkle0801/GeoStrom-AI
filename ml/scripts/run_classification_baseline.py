"""Phase 5 Task 7-8: non-deep-learning classification baselines + evaluation.

Trains on the TRAIN split only, selects nothing on the test split (the val
split is used only to report its own metrics -- no hyperparameter search is
performed in this phase, so there is no selection step to leak; this
limitation, and why, is stated explicitly in the output and the Phase 5
doc), and reports final metrics on train/val/test with MACRO-F1 as the
primary metric.

Usage:
    python ml/scripts/run_classification_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import MANIFEST_DIR, REPORT_DIR, get_data_root  # noqa: E402
from ml.geostrom_ml.classification.baselines import (  # noqa: E402
    MajorityClassBaseline, build_lightgbm_model, build_logistic_regression_pipeline,
)
from ml.geostrom_ml.classification.evaluation import evaluate, macro_f1_over_present_classes  # noqa: E402
from ml.geostrom_ml.classification.features import FEATURE_NAMES, build_feature_matrix  # noqa: E402
from ml.geostrom_ml.classification.imbalance import compute_class_weights  # noqa: E402
from ml.geostrom_ml.classification.leakage import (  # noqa: E402
    assert_no_excluded_rows, assert_no_storm_split_leakage,
)
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1, LABEL_VERSION  # noqa: E402
from ml.geostrom_ml.satellite.schema import DATASET_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "phase5_baseline_results.json")
    ap.add_argument("--figures-dir", type=Path, default=REPORT_DIR / "figures")
    args = ap.parse_args()

    import pandas as pd

    root = get_data_root()
    idx_path = root / "processed" / "classification" / LABEL_VERSION / "classification_index.parquet"
    zarr_path = root / "processed" / "satellite" / args.dataset_version / "images.zarr"
    if not idx_path.exists():
        print(f"No classification index at {idx_path}. Run build_classification_dataset.py first.",
              file=sys.stderr)
        return 1

    index = pd.read_parquet(idx_path)
    assert_no_storm_split_leakage(index)  # defense-in-depth, see leakage.py
    included = index[index["qc_status"] == "included"].copy()
    assert_no_excluded_rows(included)
    print(f"Included samples: {len(included)}  (excluded: {len(index) - len(included)})")

    print("Extracting deterministic image features from the canonical Zarr store...")
    feat_df = build_feature_matrix(included, zarr_path)
    data = included.merge(feat_df, on="sample_id", how="inner")
    assert len(data) == len(included), "feature extraction dropped or duplicated rows"

    splits = {s: data[data["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for name, d in splits.items():
        print(f"{name}: {len(d)} samples, {d['storm_id'].nunique()} storms, "
              f"classes present: {sorted(d['final_class'].unique().tolist())}")

    X = {s: d[FEATURE_NAMES] for s, d in splits.items()}
    y = {s: d["final_class"] for s, d in splits.items()}

    class_weights = compute_class_weights(y["train"])
    print(f"Training-split class weights: {class_weights}")

    results: dict = {
        "label_version": LABEL_VERSION,
        "final_classes": FINAL_CLASSES_V1,
        "class_weights_train_only": class_weights,
        "split_sizes": {s: {"n_samples": int(len(d)), "n_storms": int(d["storm_id"].nunique())}
                        for s, d in splits.items()},
        "note_on_val_selection": (
            "No hyperparameter search was performed in this phase (fixed, "
            "documented baseline configurations only), so there is no "
            "selection decision that could leak from val or test. The val "
            "split's own metrics are reported for completeness and to "
            "surface its own small-sample limitations, not used to choose "
            "between models."
        ),
        "models": {},
    }

    # ---- 1. Majority-class baseline ---------------------------------------
    maj = MajorityClassBaseline().fit(y["train"])
    results["models"]["majority_class"] = _evaluate_model_predictions(
        {s: maj.predict(len(y[s])) for s in splits}, y, FINAL_CLASSES_V1)
    print(f"Majority-class baseline: predicts '{maj.majority_class}' always. "
          f"Test macro-F1: {results['models']['majority_class']['test']['macro_f1']:.4f}")

    # ---- 2. Logistic regression on deterministic image features -----------
    lr = build_logistic_regression_pipeline(class_weight="balanced")
    lr.fit(X["train"], y["train"])
    lr_preds = {s: lr.predict(X[s]).tolist() for s in splits}
    results["models"]["logistic_regression"] = _evaluate_model_predictions(lr_preds, y, FINAL_CLASSES_V1)
    print(f"Logistic regression: test macro-F1: "
          f"{results['models']['logistic_regression']['test']['macro_f1']:.4f}")

    # ---- 3. LightGBM on the same features (tree-based, already a dep) -----
    try:
        gbm = build_lightgbm_model(class_weight=class_weights)
        gbm.fit(X["train"], y["train"])
        gbm_preds = {s: gbm.predict(X[s]).tolist() for s in splits}
        results["models"]["lightgbm"] = _evaluate_model_predictions(gbm_preds, y, FINAL_CLASSES_V1)
        print(f"LightGBM: test macro-F1: {results['models']['lightgbm']['test']['macro_f1']:.4f}")
    except ImportError:
        print("lightgbm not available -- skipped (majority-class and logistic-regression still reported).")

    # ---- primary metric summary -------------------------------------------
    best = max(
        (name for name in results["models"] if name != "majority_class"),
        key=lambda n: results["models"][n]["test"]["macro_f1"],
        default=None,
    )
    results["primary_metric"] = "macro_f1"
    results["best_model_by_test_macro_f1"] = best

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written: {args.out}")

    try:
        _render_confusion_matrix(results, args.figures_dir)
    except ImportError:
        print("matplotlib not available -- skipped confusion-matrix figure.")

    return 0


def _evaluate_model_predictions(preds: dict, y: dict, labels: list[str]) -> dict:
    out = {}
    for split_name in ("train", "val", "test"):
        m = evaluate(y[split_name], preds[split_name], labels)
        present_f1, present_labels = macro_f1_over_present_classes(y[split_name], preds[split_name])
        m["macro_f1_over_present_classes_only"] = present_f1
        m["classes_present_in_this_split"] = present_labels
        out[split_name] = m
    return out


def _render_confusion_matrix(results: dict, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    best = results.get("best_model_by_test_macro_f1")
    if not best:
        return
    test_result = results["models"][best]["test"]
    cm = np.array(test_result["confusion_matrix"])
    labels = test_result["confusion_matrix_labels"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix -- {best} (test split, n={test_result['n_samples']})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_confusion_matrix.png", dpi=110)
    plt.close(fig)
    print(f"Confusion matrix figure written: {figures_dir / 'phase5_confusion_matrix.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
