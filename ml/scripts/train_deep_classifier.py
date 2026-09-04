"""Phase 6: train + evaluate one deep-learning classifier on the frozen
Phase 5 taxonomy/split.

Never touches the test split during training or model selection -- the
test split is loaded but evaluated exactly once, after early-stopped
training on train/val is complete.

Usage:
    python ml/scripts/train_deep_classifier.py --model small_cnn
    python ml/scripts/train_deep_classifier.py --model resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR, get_data_root  # noqa: E402
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1, LABEL_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["small_cnn", "resnet18"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not installed. See ml/requirements-deep-learning.txt", file=sys.stderr)
        return 1

    import pandas as pd

    from ml.geostrom_ml.classification.deep.config import RESNET18_CONFIG, SMALL_CNN_CONFIG
    from ml.geostrom_ml.classification.deep.dataset import (
        SceneImageDataset, compute_train_normalization_stats,
    )
    from ml.geostrom_ml.classification.deep.losses import build_class_weighted_loss
    from ml.geostrom_ml.classification.deep.models import build_model, count_trainable_parameters
    from ml.geostrom_ml.classification.deep.training import (
        evaluate_on_split, resolve_device, save_checkpoint, set_deterministic, train_model,
    )
    from ml.geostrom_ml.classification.leakage import (
        assert_no_excluded_rows, assert_no_storm_split_leakage,
    )

    config = RESNET18_CONFIG if args.model == "resnet18" else SMALL_CNN_CONFIG
    device = resolve_device(config.device)
    # MUST happen before any stochastic operation -- model weight initialisation
    # (build_model, below) consumes the global torch RNG stream. Seeding only
    # inside train_model() (as train_model does again, harmlessly, for anyone
    # calling it directly) was too late: two "identical" runs produced
    # different initial weights and therefore different results -- a real
    # reproducibility bug found while verifying Task 10, fixed here.
    set_deterministic(config.seed)
    print(f"Model: {args.model}  Device: {device}")
    print(f"Config: {config.as_dict()}")

    root = get_data_root()
    idx_path = root / "processed" / "classification" / LABEL_VERSION / "classification_index.parquet"
    zarr_path = root / "processed" / "satellite" / "satellite_v1" / "images.zarr"
    if not idx_path.exists():
        print(f"No classification index at {idx_path}. Run Phase 5's "
              f"build_classification_dataset.py first.", file=sys.stderr)
        return 1

    index = pd.read_parquet(idx_path)
    assert_no_storm_split_leakage(index)
    included = index[index["qc_status"] == "included"].copy()
    assert_no_excluded_rows(included)

    print("Computing normalization stats from the TRAINING split only...")
    train_mean, train_std = compute_train_normalization_stats(index, zarr_path)
    print(f"  train_mean={train_mean:.3f} K  train_std={train_std:.3f} K")

    train_ds = SceneImageDataset(index, zarr_path, "train", train_mean=train_mean,
                                  train_std=train_std, augment=True, seed=config.seed)
    val_ds = SceneImageDataset(index, zarr_path, "val", train_mean=train_mean,
                               train_std=train_std, augment=False, seed=config.seed)
    test_ds = SceneImageDataset(index, zarr_path, "test", train_mean=train_mean,
                                train_std=train_std, augment=False, seed=config.seed)
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_labels = included[included["split"] == "train"]["final_class"]
    criterion = build_class_weighted_loss(train_labels, label_smoothing=config.label_smoothing,
                                          device=device)

    model_kwargs = {"freeze_until_layer": config.resnet_freeze_until_layer} if args.model == "resnet18" else {}
    model = build_model(args.model, **model_kwargs)
    trainable, total = count_trainable_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    print("Training (early-stopping on val macro-F1, test split untouched)...")
    model, history = train_model(model, train_ds, val_ds, config, criterion)
    print(f"Best epoch: {history.best_epoch}  best val macro-F1: {history.best_val_macro_f1:.4f}  "
          f"stopped_early={history.stopped_early}  wall_clock={history.wall_clock_seconds:.1f}s")

    print("Evaluating on TEST split (single evaluation, post model-selection)...")
    test_metrics = evaluate_on_split(model, test_ds, config)
    train_metrics_final = evaluate_on_split(model, train_ds, config)
    val_metrics_final = evaluate_on_split(model, val_ds, config)
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}  accuracy: {test_metrics['accuracy']:.4f}")

    checkpoint_dir = args.checkpoint_dir or (root / "processed" / "classification" / LABEL_VERSION
                                             / "checkpoints")
    ckpt_path = checkpoint_dir / f"{args.model}_best.pt"
    save_checkpoint(model, config, history, ckpt_path)
    print(f"Checkpoint written: {ckpt_path}")

    results = {
        "model": args.model,
        "label_version": LABEL_VERSION,
        "final_classes": FINAL_CLASSES_V1,
        "config": config.as_dict(),
        "device": device,
        "n_trainable_parameters": trainable,
        "n_total_parameters": total,
        "normalization": {"train_mean_k": train_mean, "train_std_k": train_std},
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "training_history": history.epochs,
        "best_epoch": history.best_epoch,
        "best_val_macro_f1": history.best_val_macro_f1,
        "stopped_early": history.stopped_early,
        "wall_clock_seconds": history.wall_clock_seconds,
        "final_train_metrics": train_metrics_final,
        "final_val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "checkpoint_path_relative_to_data_root": str(ckpt_path.relative_to(root)).replace("\\", "/"),
    }
    out = args.out or (REPORT_DIR / f"phase6_{args.model}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
