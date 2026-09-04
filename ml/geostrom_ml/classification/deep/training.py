"""Phase 6 training loop: deterministic, early-stopped on validation
Macro-F1, test set never touched until the final, single evaluation call.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.geostrom_ml.classification.deep.config import TrainingConfig
from ml.geostrom_ml.classification.deep.dataset import IDX_TO_CLASS, SceneImageDataset
from ml.geostrom_ml.classification.evaluation import evaluate
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1


def set_deterministic(seed: int) -> None:
    """Best-effort full determinism. Documented limitation: CUDA convolution
    algorithms are not bit-exact across GPU driver/cuDNN versions even with
    these flags set (a well-known, unavoidable PyTorch/cuDNN property, not
    a bug in this code) -- see docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md
    "Reproducibility" section for what IS and is NOT guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _TorchSceneDataset(Dataset):
    """Thin torch.utils.data.Dataset wrapper around SceneImageDataset."""

    def __init__(self, scene_dataset: SceneImageDataset):
        self.ds = scene_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        image, label_idx, sample_id = self.ds[i]
        return torch.from_numpy(image), label_idx, sample_id


def make_loader(scene_dataset: SceneImageDataset, batch_size: int, shuffle: bool,
                seed: int, num_workers: int = 0) -> DataLoader:
    torch_ds = _TorchSceneDataset(scene_dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(torch_ds, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=num_workers, drop_last=False)


@dataclass
class TrainingHistory:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_f1: float = -1.0
    stopped_early: bool = False
    wall_clock_seconds: float = 0.0


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str,
                  train: bool) -> tuple[float, list[int], list[int]]:
    model.train(mode=train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _sample_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            n += images.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return total_loss / max(n, 1), y_true, y_pred


def train_model(
    model: nn.Module,
    train_ds: SceneImageDataset,
    val_ds: SceneImageDataset,
    config: TrainingConfig,
    criterion,
) -> tuple[nn.Module, TrainingHistory]:
    """Trains `model`, early-stopping on validation Macro-F1. Returns the
    BEST-validation-Macro-F1 epoch's weights (not necessarily the last
    epoch) and the full per-epoch history for overfitting analysis."""
    set_deterministic(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)

    train_loader = make_loader(train_ds, config.batch_size, shuffle=True,
                               seed=config.seed, num_workers=config.num_workers)
    val_loader = make_loader(val_ds, config.batch_size, shuffle=False,
                             seed=config.seed, num_workers=config.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
                if config.scheduler == "cosine" else None)

    history = TrainingHistory()
    best_state = None
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(config.num_epochs):
        train_loss, train_y, train_pred = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_y, val_pred = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        if scheduler is not None:
            scheduler.step()

        train_labels = [IDX_TO_CLASS[i] for i in train_y]
        train_preds_labels = [IDX_TO_CLASS[i] for i in train_pred]
        val_labels = [IDX_TO_CLASS[i] for i in val_y]
        val_preds_labels = [IDX_TO_CLASS[i] for i in val_pred]

        train_metrics = evaluate(train_labels, train_preds_labels, FINAL_CLASSES_V1)
        val_metrics = evaluate(val_labels, val_preds_labels, FINAL_CLASSES_V1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_macro_f1": train_metrics["macro_f1"], "val_macro_f1": val_metrics["macro_f1"],
        }
        history.epochs.append(record)

        improved = val_metrics["macro_f1"] > history.best_val_macro_f1 + config.early_stopping_min_delta
        if improved:
            history.best_val_macro_f1 = val_metrics["macro_f1"]
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            history.stopped_early = True
            break

    history.wall_clock_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_on_split(model: nn.Module, dataset: SceneImageDataset, config: TrainingConfig) -> dict:
    """The ONE evaluation call meant to be used on the test split -- call
    this exactly once, after training/model-selection is fully finished."""
    device = resolve_device(config.device)
    model = model.to(device).eval()
    loader = make_loader(dataset, config.batch_size, shuffle=False, seed=config.seed)
    y_true_idx, y_pred_idx = [], []
    with torch.no_grad():
        for images, labels, _sample_ids in loader:
            logits = model(images.to(device))
            y_true_idx.extend(labels.tolist())
            y_pred_idx.extend(logits.argmax(dim=1).cpu().tolist())
    y_true = [IDX_TO_CLASS[i] for i in y_true_idx]
    y_pred = [IDX_TO_CLASS[i] for i in y_pred_idx]
    return evaluate(y_true, y_pred, FINAL_CLASSES_V1)


def save_checkpoint(model: nn.Module, config: TrainingConfig, history: TrainingHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.as_dict(),
        "best_epoch": history.best_epoch,
        "best_val_macro_f1": history.best_val_macro_f1,
    }, path)
