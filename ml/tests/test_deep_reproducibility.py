"""Phase 6 Task 10: reproducibility of the deep-learning pipeline.

Two independent training runs with the same config/seed on the same tiny
synthetic dataset must produce identical sample IDs, split assignments,
label mappings, and (CPU-only, see the documented exception below)
identical metrics.
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
from ml.geostrom_ml.classification.deep.training import evaluate_on_split, train_model  # noqa: E402
from ml.geostrom_ml.satellite.imagery import SatelliteZarrStore  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    n = 16
    store = SatelliteZarrStore(tmp_path / "images.zarr").create(n, overwrite=True)
    rng = np.random.default_rng(0)
    classes = ["CDO", "CurvedBand", "Eye", "Shear"]
    labels = [classes[i % 4] for i in range(n)]
    for i in range(n):
        base = 220.0 + 15.0 * classes.index(labels[i])
        kelvin = rng.normal(base, 3.0, size=(301, 301)).astype("float32")
        store.write_frame(i, kelvin, np.ones((301, 301), dtype=bool))
    splits = (["train"] * 8) + (["val"] * 4) + (["test"] * 4)
    index = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "storm_id": [f"storm{i // 4}" for i in range(n)],
        "split": splits,
        "zarr_index": list(range(n)),
        "final_class": labels,
        "qc_status": ["included"] * n,
    })
    return index, tmp_path / "images.zarr"


def _run(index, zarr_path, seed: int):
    config = TrainingConfig(model_name="small_cnn", num_epochs=3, batch_size=4,
                            device="cpu", seed=seed, early_stopping_patience=10)
    # Seed BEFORE constructing the model -- model weight init consumes the
    # global torch RNG stream. Seeding only inside train_model() (which also
    # calls set_deterministic, redundantly-but-harmlessly, for callers that
    # forget this) was too late and was a real bug: two "identical" runs of
    # the real 353-sample training script produced different initial
    # weights and therefore different results (best_epoch 0 vs 1, val
    # macro-F1 0.1618 vs 0.2675) until fixed. Reproduced here at unit scale
    # via set_deterministic() before SmallCNN(), matching the real fix in
    # ml/scripts/train_deep_classifier.py.
    from ml.geostrom_ml.classification.deep.training import set_deterministic
    set_deterministic(seed)

    train_ds = SceneImageDataset(index, zarr_path, "train", augment=True, seed=config.seed)
    val_ds = SceneImageDataset(index, zarr_path, "val", augment=False, seed=config.seed)
    test_ds = SceneImageDataset(index, zarr_path, "test", augment=False, seed=config.seed)
    train_labels = index[index["split"] == "train"]["final_class"]
    criterion = build_class_weighted_loss(train_labels, device="cpu")
    model = SmallCNN()
    model, history = train_model(model, train_ds, val_ds, config, criterion)
    test_metrics = evaluate_on_split(model, test_ds, config)
    return train_ds, val_ds, test_ds, history, test_metrics


class TestModelInitializationReproducibility:
    """Direct, sensitive regression test for the real bug above: this
    checks the root cause (weight values right after construction) rather
    than a downstream metric that a trivial toy dataset might be
    insensitive to."""

    def test_same_seed_gives_identical_initial_weights(self):
        from ml.geostrom_ml.classification.deep.training import set_deterministic

        set_deterministic(42)
        model_a = SmallCNN()
        set_deterministic(42)
        model_b = SmallCNN()
        for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert name_a == name_b
            assert torch.equal(p_a, p_b), f"parameter {name_a} differs between identically-seeded inits"

    def test_different_seed_gives_different_initial_weights(self):
        """Sanity-check the test methodology: unseeded/differently-seeded
        construction really does produce different weights, proving the
        equality test above is not vacuous."""
        from ml.geostrom_ml.classification.deep.training import set_deterministic

        set_deterministic(1)
        model_a = SmallCNN()
        set_deterministic(2)
        model_b = SmallCNN()
        first_param_a = next(model_a.parameters())
        first_param_b = next(model_b.parameters())
        assert not torch.equal(first_param_a, first_param_b)


class TestDatasetReproducibility:
    def test_sample_ids_identical_across_two_builds(self, tiny_dataset):
        index, zarr_path = tiny_dataset
        a_train, a_val, a_test, _, _ = _run(index, zarr_path, seed=42)
        b_train, b_val, b_test, _, _ = _run(index, zarr_path, seed=42)
        for a, b in ((a_train, b_train), (a_val, b_val), (a_test, b_test)):
            assert a.rows["sample_id"].tolist() == b.rows["sample_id"].tolist()

    def test_label_mapping_identical_across_two_builds(self, tiny_dataset):
        index, zarr_path = tiny_dataset
        a_train, *_ = _run(index, zarr_path, seed=42)
        b_train, *_ = _run(index, zarr_path, seed=42)
        assert a_train.rows["final_class"].tolist() == b_train.rows["final_class"].tolist()


class TestTrainingReproducibilityCPU:
    """CPU-only determinism is the guarantee this project makes (see
    training.py::set_deterministic's documented limitation about CUDA
    convolution algorithms not being bit-exact across driver/cuDNN
    versions -- a well-known PyTorch property, not a bug here)."""

    def test_identical_config_and_seed_gives_identical_test_metrics_on_cpu(self, tiny_dataset):
        index, zarr_path = tiny_dataset
        _, _, _, history_a, metrics_a = _run(index, zarr_path, seed=42)
        _, _, _, history_b, metrics_b = _run(index, zarr_path, seed=42)
        assert metrics_a["macro_f1"] == metrics_b["macro_f1"]
        assert metrics_a["confusion_matrix"] == metrics_b["confusion_matrix"]
        assert history_a.best_val_macro_f1 == history_b.best_val_macro_f1

    def test_different_seed_can_give_different_training_history(self, tiny_dataset):
        """Sanity-check the reproducibility test methodology itself: if
        results were identical regardless of seed, the equality tests above
        could be vacuously passing (e.g. because the model never actually
        trains). A different seed changes augmentation and initialisation,
        so SOME difference is expected -- proving the pipeline is genuinely
        seed-sensitive, not accidentally constant."""
        index, zarr_path = tiny_dataset
        _, _, _, history_a, _ = _run(index, zarr_path, seed=42)
        _, _, _, history_b, _ = _run(index, zarr_path, seed=123)
        train_losses_a = [e["train_loss"] for e in history_a.epochs]
        train_losses_b = [e["train_loss"] for e in history_b.epochs]
        assert train_losses_a != train_losses_b
