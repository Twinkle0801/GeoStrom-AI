"""Quantization round-trip and Zarr canonical-store round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from ml.geostrom_ml.satellite.imagery import (
    CHUNK_FRAMES,
    GRID,
    IRWIN_MAX_K,
    IRWIN_MIN_K,
    SatelliteZarrStore,
    dequantize_irwin,
    quantize_irwin,
)


class TestQuantization:
    def test_round_trip_error_is_within_one_quantization_step(self):
        rng = np.random.default_rng(0)
        kelvin = rng.uniform(IRWIN_MIN_K, IRWIN_MAX_K, size=(301, 301)).astype("float32")
        q = quantize_irwin(kelvin)
        back = dequantize_irwin(q)
        step = (IRWIN_MAX_K - IRWIN_MIN_K) / 255.0
        assert np.abs(back - kelvin).max() <= step / 2 + 1e-3

    def test_endpoints_map_exactly(self):
        arr = np.array([[IRWIN_MIN_K, IRWIN_MAX_K]], dtype="float32")
        q = quantize_irwin(arr)
        assert q[0, 0] == 0
        assert q[0, 1] == 255

    def test_out_of_range_values_are_clipped_not_wrapped(self):
        arr = np.array([[100.0, 400.0]], dtype="float32")  # below/above physical range
        q = quantize_irwin(arr)
        assert q[0, 0] == 0    # clipped to IRWIN_MIN_K
        assert q[0, 1] == 255  # clipped to IRWIN_MAX_K

    def test_nan_maps_to_zero_sentinel(self):
        arr = np.array([[np.nan]], dtype="float32")
        q = quantize_irwin(arr)
        assert q[0, 0] == 0

    def test_quantization_is_deterministic(self):
        rng = np.random.default_rng(1)
        kelvin = rng.uniform(IRWIN_MIN_K, IRWIN_MAX_K, size=(50, 50)).astype("float32")
        assert np.array_equal(quantize_irwin(kelvin), quantize_irwin(kelvin))


class TestSatelliteZarrStore:
    def test_write_and_read_back_a_frame(self, tmp_path):
        store = SatelliteZarrStore(tmp_path / "images.zarr").create(3, overwrite=True)
        kelvin = np.full(GRID, 220.0, dtype="float32")
        valid = np.ones(GRID, dtype="bool")
        valid[0, 0] = False
        store.write_frame(1, kelvin, valid)

        root = store.open_readonly()
        assert root["irwin_k"].shape == (3, *GRID)
        readback = root["irwin_k"][1]
        assert readback[5, 5] == pytest.approx(220.0)
        assert np.isnan(readback[0, 0])  # invalid pixel masked, not the raw sentinel
        assert bool(root["valid_mask"][1][0, 0]) is False
        assert bool(root["valid_mask"][1][5, 5]) is True

    def test_frames_not_written_default_to_the_declared_fill_values(self, tmp_path):
        store = SatelliteZarrStore(tmp_path / "images.zarr").create(2, overwrite=True)
        root = store.open_readonly()
        assert np.isnan(root["irwin_k"][0]).all()
        assert not root["valid_mask"][0].any()

    def test_store_metadata_documents_the_quantization_transform(self, tmp_path):
        store = SatelliteZarrStore(tmp_path / "images.zarr").create(1, overwrite=True)
        root = store.open_readonly()
        assert root.attrs["irwin_k_units"] == "Kelvin"
        assert "quantization" in root.attrs
        assert root.attrs["irwin_valid_range_k"] == [150.0, 350.0]

    def test_chunking_uses_the_documented_frame_count(self, tmp_path):
        store = SatelliteZarrStore(tmp_path / "images.zarr").create(100, overwrite=True)
        root = store.open_readonly()
        assert root["irwin_k"].chunks == (CHUNK_FRAMES, *GRID)
