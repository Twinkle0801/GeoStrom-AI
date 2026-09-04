"""Phase 6 training loop: a tiny, fast, fully-synthetic smoke run through
the real train_model()/evaluate_on_split() code path -- not the real
353-sample dataset (that's exercised by the real training scripts /
integration test), but the same functions, same determinism machinery.
Requires torch -- skips cleanly if unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ml.geostrom_ml.classification.deep.config import TrainingConfig  # noqa: E402
from ml.geostrom_ml.classification.deep.dataset import SceneImageDataset  # noqa: E402
from ml.geostrom_ml.classification.deep.losses import build_class_weighted_loss  # noqa: E402
from ml.geostrom_ml.classification.deep.models import SmallCNN  # noqa: E402
from ml.geostrom_ml.classification.deep.training import (  # noqa: E402
    evaluate_on_split, resolve_device, set_deterministic, train_model,
)
from ml.geostrom_ml.satellite.imagery import SatelliteZarrStore  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    n = 16
    store = SatelliteZarrStore(tmp_path / "images.zarr").create(n, overwrite=True)
    rng = np.random.default_rng(0)
    classes = ["CDO", "CurvedBand", "Eye", "Shear"]
    labels = [classes[i % 4] for i in range(n)]
    for i in range(n):
        # give each class a distinguishable mean so the model has *something*
        # learnable, without this being a real-data claim of any kind
        base = 220.0 + 15.0 * classes.index(labels[i])
        kelvin = rng.normal(base, 3.0, size=(301, 301)).astype("float32")
        store.write_frame(i, kelvin, np.ones((301, 301), dtype=bool))

    splits = (["train"] * 8) + (["val"] * 4) + (["test"] * 4)
    index = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "storm_id": [f"storm{i // 4}" for i in range(n)],  # 4 storms, disjoint per split-block
        "split": splits,
        "zarr_index": list(range(n)),
        "final_class": labels,
        "qc_status": ["included"] * n,
    })
    return index, tmp_path / "images.zarr"


class TestSetDeterministic:
    def test_does_not_raise(self):
        set_deterministic(42)  # smoke check only


class TestTinyTrainingRun:
    def test_train_model_runs_and_returns_history(self, tiny_dataset):
        index, zarr_path = tiny_dataset
        config = TrainingConfig(model_name="small_cnn", num_epochs=2,
                                batch_size=4, device="cpu", early_stopping_patience=10)
        train_ds = SceneImageDataset(index, zarr_path, "train", augment=True, seed=config.seed)
        val_ds = SceneImageDataset(index, zarr_path, "val", augment=False, seed=config.seed)

        train_labels = index[index["split"] == "train"]["final_class"]
        criterion = build_class_weighted_loss(train_labels, device="cpu")

        model = SmallCNN()
        model, history = train_model(model, train_ds, val_ds, config, criterion)

        assert len(history.epochs) == 2
        assert history.best_epoch >= 0
        for record in history.epochs:
            assert "train_loss" in record and "val_loss" in record
            assert "train_macro_f1" in record and "val_macro_f1" in record

    def test_evaluate_on_split_never_trains(self, tiny_dataset):
        """The evaluation function must leave the model's weights
        unchanged -- proof it does not accidentally call .backward()."""
        index, zarr_path = tiny_dataset
        config = TrainingConfig(model_name="small_cnn", device="cpu")
        test_ds = SceneImageDataset(index, zarr_path, "test", augment=False, seed=config.seed)
        model = SmallCNN()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        evaluate_on_split(model, test_ds, config)
        after = model.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k])

    def test_resolve_device_falls_back_to_cpu_when_requested(self):
        assert resolve_device("cpu") == "cpu"

    def test_early_stopping_triggers_with_zero_patience_and_no_improvement(self, tiny_dataset):
        index, zarr_path = tiny_dataset
        config = TrainingConfig(model_name="small_cnn", num_epochs=20, batch_size=4,
                                device="cpu", early_stopping_patience=1,
                                early_stopping_min_delta=1.0)  # impossible-to-hit improvement
        train_ds = SceneImageDataset(index, zarr_path, "train", augment=True, seed=config.seed)
        val_ds = SceneImageDataset(index, zarr_path, "val", augment=False, seed=config.seed)
        train_labels = index[index["split"] == "train"]["final_class"]
        criterion = build_class_weighted_loss(train_labels, device="cpu")
        model = SmallCNN()
        _, history = train_model(model, train_ds, val_ds, config, criterion)
        assert history.stopped_early
        assert len(history.epochs) < 20
