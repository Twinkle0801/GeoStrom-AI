"""Canonical satellite imagery storage: physical values in Zarr.

Design decision (documented here, not silently assumed):

  RESOLUTION -- the canonical store keeps HURSAT-B1's native 301x301 grid
  unchanged. docs/DATA_STRATEGY.md's Phase 0/1 planning sketch assumed a
  224x224 resize "for storage efficiency", but that was explicitly marked an
  ASSUMPTION pending Phase 4. The Phase 4 task instructs: "Do NOT apply
  arbitrary brightness-temperature normalization that changes the physical
  interpretation" and "preserve raw/physical values in the canonical
  dataset". Any resampling (crop or interpolation) to 224x224 either alters
  pixel values (interpolation) or discards field-of-view asymmetrically
  (crop) -- both are transformations a downstream model phase can apply
  deterministically FROM the canonical 301x301 grid, but neither belongs in
  the canonicalization step itself. Storage cost at 301x301 remains modest:
  see docs/PHASE_4_SATELLITE_PIPELINE.md §9 for measured per-frame bytes.

  VALUES -- the canonical array (`irwin_k`) stores brightness temperature in
  Kelvin, float32, physically masked (invalid pixels are NaN, governed by a
  SEPARATE boolean `valid_mask` array -- never a magic sentinel hidden in
  the value channel). A companion `irwin_u8` array stores a DETERMINISTIC,
  DOCUMENTED, fully invertible-to-1-Kelvin-resolution uint8 quantization of
  the same physically valid range (150-350 K -> 0-255), for compact storage;
  it is a convenience copy, never the source of truth. `dequantize_irwin`
  exactly reproduces the transform for any future consumer.

  CHUNKING -- (32, 301, 301) per array. 32 frames x 301x301x4 bytes
  (float32) = ~11.6 MB/chunk -- large enough to avoid a small-files problem
  at MVP scale (tens of thousands of frames -> low hundreds of chunks), small
  enough to read a single training batch without pulling in unrelated
  frames. Documented here per the Phase 4 task's explicit instruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from ml.geostrom_ml.satellite.schema import IRWIN_VALID_RANGE_K

CHUNK_FRAMES = 32
GRID = (301, 301)
IRWIN_MIN_K, IRWIN_MAX_K = IRWIN_VALID_RANGE_K
QUANT_LEVELS = 256


def quantize_irwin(kelvin: np.ndarray) -> np.ndarray:
    """Deterministic float32 Kelvin -> uint8 quantization over [150, 350] K.

    Values outside the physically valid range (already NaN by the time they
    reach here, per `hursat.read_irwin`) map to 0 -- the sentinel is
    disambiguated by `valid_mask`, never trusted on its own.
    """
    clipped = np.clip(kelvin, IRWIN_MIN_K, IRWIN_MAX_K)
    scaled = (clipped - IRWIN_MIN_K) / (IRWIN_MAX_K - IRWIN_MIN_K) * (QUANT_LEVELS - 1)
    q = np.nan_to_num(scaled, nan=0.0)
    return np.rint(q).astype("uint8")


def dequantize_irwin(u8: np.ndarray) -> np.ndarray:
    """Exact inverse of `quantize_irwin`'s linear mapping (up to rounding)."""
    return (u8.astype("float32") / (QUANT_LEVELS - 1)) * (IRWIN_MAX_K - IRWIN_MIN_K) + IRWIN_MIN_K


class SatelliteZarrStore:
    """Canonical imagery store: irwin_k (float32 K), irwin_u8 (uint8), valid_mask (bool)."""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self._root = None  # cached open group; avoids a re-open per frame in write_frame

    def create(self, n_samples: int, *, overwrite: bool = False) -> "SatelliteZarrStore":
        store = zarr.storage.LocalStore(str(self.store_path))
        root = zarr.open_group(store=store, mode="a")
        shape = (n_samples, *GRID)
        chunks = (min(CHUNK_FRAMES, max(n_samples, 1)), *GRID)
        root.create_array(name="irwin_k", shape=shape, chunks=chunks, dtype="float32",
                           fill_value=np.nan, overwrite=overwrite)
        root.create_array(name="irwin_u8", shape=shape, chunks=chunks, dtype="uint8",
                           fill_value=0, overwrite=overwrite)
        root.create_array(name="valid_mask", shape=shape, chunks=chunks, dtype="bool",
                           fill_value=False, overwrite=overwrite)
        root.attrs["grid"] = list(GRID)
        root.attrs["irwin_k_units"] = "Kelvin"
        root.attrs["irwin_valid_range_k"] = list(IRWIN_VALID_RANGE_K)
        root.attrs["quantization"] = (
            "irwin_u8 = round((clip(irwin_k, 150, 350) - 150) / 200 * 255); "
            "dequantize via ml.geostrom_ml.satellite.imagery.dequantize_irwin"
        )
        root.attrs["n_samples"] = n_samples
        self._root = root  # keep the handle open for the write_frame loop that follows
        return self

    def write_frame(self, index: int, kelvin: np.ndarray, valid_mask: np.ndarray) -> None:
        """Write one frame. Re-uses the group handle opened by `create()` (or
        the first call's own open, for standalone use) rather than re-opening
        the Zarr store on every call -- with thousands of frames, a per-call
        re-open dominates runtime on Windows filesystems."""
        if self._root is None:
            store = zarr.storage.LocalStore(str(self.store_path))
            self._root = zarr.open_group(store=store, mode="a")
        masked = np.where(valid_mask, kelvin, np.nan).astype("float32")
        self._root["irwin_k"][index] = masked
        self._root["irwin_u8"][index] = quantize_irwin(masked)
        self._root["valid_mask"][index] = valid_mask.astype("bool")

    def open_readonly(self):
        store = zarr.storage.LocalStore(str(self.store_path))
        return zarr.open_group(store=store, mode="r")
