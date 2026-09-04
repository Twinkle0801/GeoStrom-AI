"""Phase 6: PyTorch Dataset over Phase 5's classification index + Phase 4's
canonical Zarr imagery.

============================================================================
INVALID-PIXEL HANDLING (documented, not silent -- per the Phase 6 task)
============================================================================

A HURSAT-B1 frame's `valid_mask` (Phase 4) marks pixels outside the
physically valid IRWIN range (150-350 K) as invalid -- typically a small
fraction (Phase 4 measured mean 0.3%, max 4.7% per frame). These pixels
have NO reliable brightness-temperature reading; feeding them to a CNN as
their raw (masked-to-NaN) value would propagate NaN through every
convolution touching them.

Chosen fill: **the frame's own valid-pixel mean**, computed per-image
after normalization. This is a deliberate choice among the alternatives,
stated explicitly:
  - NOT zero-fill: 0.0 in the normalized (post-standardisation) space does
    not correspond to a real physical value and would look like an
    artificially extreme cold (or, depending on normalisation sign, hot)
    reading to the network -- a fabricated temperature.
  - NOT the dataset-wide mean: would leak per-split statistics were it
    computed across splits; the per-IMAGE mean uses only information
    already present in that same image.
  - The per-image mean is the least informative value achievable without
    inventing data: it contributes ~zero gradient signal at invalid pixel
    locations (they look "neutral" relative to that image's own
    distribution) while remaining a physically plausible temperature for
    that scene.

This mirrors the same "never fabricate, always document" principle Phase 4
applied to IRWIN QC (`ml/geostrom_ml/satellite/hursat.py`) and to the
statistical-feature baseline's NaN handling (Phase 5
`ml/geostrom_ml/classification/features.py`).

**Ordering requirement (found via a real bug during Phase 6 development,
fixed, and documented here so it cannot regress silently): invalid-pixel
fill MUST happen before any interpolation step (rotation augmentation,
resize).** The canonical Zarr's `irwin_k` array stores NaN, not a finite
sentinel, at invalid pixels (`ml/geostrom_ml/satellite/imagery.py`).
Bilinear/order>0 interpolation of a NaN-containing array spreads NaN into
every neighbouring pixel the interpolation kernel touches -- rotating (or
resizing) BEFORE filling turned a handful of invalid pixels per frame into
thousands of NaN pixels after interpolation, which silently produced a NaN
training loss from epoch 0 (caught by inspecting `training_history` in
`ml/reports/phase6_small_cnn_results.json`, not by a passing test -- the
existing unit tests used all-valid synthetic frames and could not have
caught this; a regression test using a frame with invalid pixels was added
after the fact, see `ml/tests/test_deep_dataset.py`). `__getitem__` below
therefore always fills before it augments or resizes.

============================================================================
RESIZE (301x301 -> 224x224, data-loading pipeline ONLY)
============================================================================

The canonical Zarr store stays 301x301 (Phase 4 locked decision, untouched
by Phase 6). Resize to 224 (docs/ML_ARCHITECTURE.md's locked MVP CNN input
size, matching ImageNet-pretrained backbones) happens here, in the Dataset,
using deterministic bilinear interpolation (`scipy.ndimage.zoom`, no
randomness) -- applied identically to every sample regardless of split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.geostrom_ml.classification.deep.augmentation import augment_train_image
from ml.geostrom_ml.classification.deep.config import MODEL_INPUT_SIZE, NATIVE_GRID
from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(FINAL_CLASSES_V1)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


def resize_deterministic(image: np.ndarray, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Deterministic bilinear resize -- no randomness, always the same
    output for the same input, regardless of split or call order."""
    from scipy.ndimage import zoom

    factor = size / image.shape[0]
    resized = zoom(image, factor, order=1)
    # zoom's output size can be off by one pixel due to floating rounding;
    # crop/pad to the exact target deterministically.
    out = np.zeros((size, size), dtype=image.dtype)
    h, w = min(size, resized.shape[0]), min(size, resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    return out


def fill_invalid_with_image_mean(kelvin: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """See module docstring, 'INVALID-PIXEL HANDLING'."""
    valid = valid_mask.astype(bool)
    if not valid.any():
        return np.zeros_like(kelvin)  # degenerate case: no valid pixel at all in this frame
    fill_value = float(kelvin[valid].mean())
    out = kelvin.copy()
    out[~valid] = fill_value
    return out


class SceneImageDataset:
    """PyTorch-Dataset-compatible (implements __len__/__getitem__) loader
    over one split's classification-index rows.

    Deliberately does NOT subclass `torch.utils.data.Dataset` at module
    import time so this file (and its resize/fill helpers) can be imported
    and unit-tested even in environments without torch installed; the
    subclassing happens in `training.py::to_torch_dataset`, right where
    torch is actually needed.
    """

    def __init__(self, index_df: pd.DataFrame, zarr_path: Path, split: str, *,
                 train_mean: float | None = None, train_std: float | None = None,
                 augment: bool = False, seed: int = 42):
        rows = index_df[(index_df["qc_status"] == "included") & (index_df["split"] == split)]
        self.rows = rows.sort_values("zarr_index").reset_index(drop=True)
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.augment = augment
        self.seed = seed
        self.train_mean = train_mean
        self.train_std = train_std
        self._root = None

    def __len__(self) -> int:
        return len(self.rows)

    def _ensure_open(self):
        if self._root is None:
            import zarr
            self._root = zarr.open_group(store=zarr.storage.LocalStore(str(self.zarr_path)), mode="r")
        return self._root

    def raw_frame(self, i: int) -> tuple[np.ndarray, np.ndarray, str, str]:
        """Native 301x301 frame + mask, unprocessed -- for tests/inspection."""
        root = self._ensure_open()
        row = self.rows.iloc[i]
        zi = int(row["zarr_index"])
        kelvin = np.asarray(root["irwin_k"][zi])
        mask = np.asarray(root["valid_mask"][zi])
        return kelvin, mask, row["sample_id"], row["final_class"]

    def __getitem__(self, i: int):
        kelvin, mask, sample_id, final_class = self.raw_frame(i)
        assert kelvin.shape == NATIVE_GRID

        # Fill invalid (NaN, per the canonical Zarr's own convention -- see
        # ml/geostrom_ml/satellite/imagery.py) pixels BEFORE any interpolation
        # (augmentation's rotation, or resize). Interpolating a NaN-containing
        # array (bilinear or any other order>0 interpolation) spreads NaN into
        # every neighbouring pixel it touches -- filling first, on the native
        # grid, guarantees rotation/resize only ever see finite values.
        filled = fill_invalid_with_image_mean(kelvin, mask)

        if self.augment:
            filled, _ = augment_train_image(filled, mask, i, self.seed)

        resized = resize_deterministic(filled, MODEL_INPUT_SIZE)

        if self.train_mean is not None and self.train_std is not None:
            normalized = (resized - self.train_mean) / self.train_std
        else:
            normalized = resized

        label_idx = CLASS_TO_IDX[final_class]
        return normalized.astype("float32")[None, :, :], label_idx, sample_id


def compute_train_normalization_stats(index_df: pd.DataFrame, zarr_path: Path) -> tuple[float, float]:
    """Mean/std of valid pixels across the TRAINING split only -- the
    normalisation stats applied identically to train/val/test (never
    computed from val or test, which would leak their distribution into
    preprocessing)."""
    ds = SceneImageDataset(index_df, zarr_path, split="train", augment=False)
    all_valid = []
    for i in range(len(ds)):
        kelvin, mask, _, _ = ds.raw_frame(i)
        all_valid.append(kelvin[mask.astype(bool)])
    pooled = np.concatenate(all_valid)
    return float(pooled.mean()), float(pooled.std())
