"""Phase 6 training configuration -- every knob stated and justified.

With only 353 training samples across 7 storms, this configuration is
deliberately conservative: few epochs, aggressive early stopping, weight
decay, and (for the transfer-learning model) a mostly-frozen backbone.
None of these choices are tuned against the validation or test split's
performance -- they are fixed before any evaluation is inspected, per the
task's explicit "do not use the test set during training" and "do not
interpret high training accuracy as evidence of successful classification"
instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RANDOM_SEED = 42  # matches ml/geostrom_ml/classification/baselines.py::RANDOM_SEED

# Native HURSAT-B1 grid (Phase 4, locked -- see docs/PHASE_4_SATELLITE_PIPELINE.md §9).
NATIVE_GRID = (301, 301)
# Model input size. Chosen, not invented: docs/ML_ARCHITECTURE.md §5.2/§9 locked 224x224 as
# the MVP CNN input (matches ImageNet-pretrained backbones). This resize happens ONLY inside
# the data-loading pipeline (see dataset.py); the canonical Zarr store stays 301x301,
# untouched. See docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md for the full justification.
MODEL_INPUT_SIZE = 224


@dataclass
class TrainingConfig:
    model_name: str  # "small_cnn" | "resnet18"
    seed: int = RANDOM_SEED
    batch_size: int = 16          # 353 train samples -> ~22 batches/epoch; small on purpose
    num_epochs: int = 40          # upper bound; early stopping is expected to trigger first
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4    # L2 regularisation -- necessary at this sample size
    optimizer: str = "adamw"
    scheduler: str = "cosine"     # cosine decay over num_epochs, no restarts
    early_stopping_metric: str = "val_macro_f1"  # NOT val_loss: matches the primary metric
    early_stopping_patience: int = 8   # epochs without val_macro_f1 improvement
    early_stopping_min_delta: float = 0.0
    checkpoint_metric: str = "val_macro_f1"  # save the best-val-macro-F1 epoch only
    num_workers: int = 0          # Windows spawn-based multiprocessing is expensive at this
                                   # dataset size (docs/ML_ARCHITECTURE.md §9) -- not worth it
                                   # for 353 samples; also keeps determinism simpler
    device: str = "auto"          # "auto" -> cuda if available, else cpu (see compute-safety
                                   # check in ml/scripts/dl_smoke_test.py)
    label_smoothing: float = 0.05  # Phase 5's imbalance.py::PHASE_6_IMBALANCE_STRATEGY
                                    # explicitly named this as an "additional measure"
    class_weighted_loss: bool = True  # weights computed from TRAIN split only, see losses.py

    # ResNet18-specific: freeze all backbone layers except the last residual block + head.
    # Justification: 353 images cannot fine-tune 11M parameters without overfitting
    # immediately; freezing preserves the ImageNet-pretrained low/mid-level filters (edges,
    # textures) which transfer even to IR imagery, and only adapts the task-specific head.
    resnet_freeze_until_layer: str = "layer4"  # layers before this stay frozen

    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "extra"} | self.extra


SMALL_CNN_CONFIG = TrainingConfig(model_name="small_cnn", num_epochs=40, learning_rate=1e-3)
RESNET18_CONFIG = TrainingConfig(
    model_name="resnet18", num_epochs=25, learning_rate=3e-4, batch_size=16,
)
