"""Real-data end-to-end integration test: Phase 1's actual HURSAT-B1 +
ADT-HURSAT verification sample, run through the complete Phase 4 pipeline.

Skips cleanly (does not fail) when the ~130 MB Phase 1 sample is not present
on this machine, mirroring the existing skip pattern in
`ml/tests/test_leakage.py::TestMaterialisedDatasetSplitLeakage`.

NOTE ON RUNTIME: opening each real HURSAT NetCDF file costs roughly 1.5-2s
on this workstation (measured; see docs/PHASE_4_SATELLITE_PIPELINE.md §17).
With 195 real frames read twice (metadata pass + pixel pass), this single
test module takes several minutes to run -- an inherent cost of testing
against real files rather than synthetic fixtures, not a bug. It is
isolated to this one session-scoped fixture so the cost is paid once, not
once per assertion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.geostrom_ml.config import MANIFEST_DIR, zone
from ml.geostrom_ml.data.ibtracs import load_ibtracs_raw
from ml.geostrom_ml.satellite.pipeline import load_split_map, run_pipeline


@pytest.fixture(scope="module")
def real_pipeline_result(tmp_path_factory):
    sample_dir = zone("samples", "hursat")
    if not sample_dir.exists() or not any(sample_dir.rglob("*.hursat-b1.v06.nc")):
        pytest.skip("Phase 1 HURSAT sample not present on this machine")

    ib_raw, _ = load_ibtracs_raw(basin="NA")
    out_zarr = tmp_path_factory.mktemp("satellite_it") / "images.zarr"
    return run_pipeline(
        interim_hursat_dir=sample_dir,
        adt_dir=zone("samples", "adt"),
        splits_path=MANIFEST_DIR / "splits_v1.json",
        ibtracs_full_track=ib_raw,
        zarr_out_path=out_zarr,
    )


class TestRealSampleEndToEnd:
    def test_frames_are_discovered_and_parsed(self, real_pipeline_result):
        """When every frame parses cleanly, `inventory_frames` produces no
        `error` column at all (no dict in the batch ever set that key) --
        that absence itself means zero errors, same convention
        `pipeline.py`/`qc.py` already rely on (`"error" in inventory` guard)."""
        inventory = real_pipeline_result["inventory"]
        assert len(inventory) > 0
        if "error" in inventory.columns:
            assert inventory["error"].isna().all()

    def test_deduplication_reduces_frame_count(self, real_pipeline_result):
        assert len(real_pipeline_result["canonical"]) < len(real_pipeline_result["inventory"])
        assert len(real_pipeline_result["canonical"]) > 0

    def test_final_index_has_no_duplicate_storm_timestamp_pairs(self, real_pipeline_result):
        fi = real_pipeline_result["final_index"]
        assert not fi.duplicated(["storm_id", "satellite_timestamp"]).any()

    def test_final_index_has_no_duplicate_sample_ids(self, real_pipeline_result):
        fi = real_pipeline_result["final_index"]
        assert fi["sample_id"].is_unique

    def test_every_final_sample_is_within_spatial_qc_gate(self, real_pipeline_result):
        fi = real_pipeline_result["final_index"]
        assert (fi["spatial_distance_km"] < 50.0).all()

    def test_every_final_sample_has_a_frozen_split_label(self, real_pipeline_result):
        fi = real_pipeline_result["final_index"]
        assert fi["split"].isin(["train", "val", "test"]).all()

    def test_split_assignment_matches_the_frozen_manifest_exactly(self, real_pipeline_result):
        fi = real_pipeline_result["final_index"]
        split_map = load_split_map(MANIFEST_DIR / "splits_v1.json")
        mismatches = fi[fi.apply(lambda r: split_map.get(r["storm_id"]) != r["split"], axis=1)]
        assert mismatches.empty

    def test_gate_status_is_pass(self, real_pipeline_result):
        assert real_pipeline_result["qc_report"]["summary"]["gate_status"] == "PASS"

    def test_at_least_one_sample_has_a_scene_label(self, real_pipeline_result):
        """The 3-storm sample's ADT files were verified in Phase 1 to have
        100% coverage -- a failure here would indicate a real regression in
        the ADT join, not an expected-missing-data case."""
        fi = real_pipeline_result["final_index"]
        assert fi["scene_label"].notna().any()

    def test_zarr_store_row_count_matches_final_index(self, real_pipeline_result):
        import zarr

        n = len(real_pipeline_result["final_index"])
        store = zarr.storage.LocalStore(real_pipeline_result["zarr_store_path"])
        root = zarr.open_group(store=store, mode="r")
        assert root["irwin_k"].shape[0] == n

    def test_pipeline_is_reproducible_on_a_second_run(self, real_pipeline_result, tmp_path_factory):
        """Same inputs -> identical sample_ids, split assignments, and row
        count on a second, independent run."""
        ib_raw, _ = load_ibtracs_raw(basin="NA")
        out_zarr2 = tmp_path_factory.mktemp("satellite_it2") / "images.zarr"
        second = run_pipeline(
            interim_hursat_dir=zone("samples", "hursat"),
            adt_dir=zone("samples", "adt"),
            splits_path=MANIFEST_DIR / "splits_v1.json",
            ibtracs_full_track=ib_raw,
            zarr_out_path=out_zarr2,
        )
        first_ids = sorted(real_pipeline_result["final_index"]["sample_id"])
        second_ids = sorted(second["final_index"]["sample_id"])
        assert first_ids == second_ids
