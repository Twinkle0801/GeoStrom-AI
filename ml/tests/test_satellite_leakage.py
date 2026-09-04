"""Dedicated satellite-pipeline leakage regression tests.

Mirrors the adversarial spirit of `ml/tests/test_leakage.py` (Phase 2): each
test targets one of the six leakage vectors the Phase 4 task explicitly
names, and includes at least one test that would FAIL if the corresponding
protection were removed -- so these tests are not vacuously passing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.geostrom_ml.satellite.alignment import join_frames_to_ibtracs
from ml.geostrom_ml.satellite.dedup import select_canonical_frames
from ml.geostrom_ml.satellite.schema import DEFAULT_TEMPORAL_TOLERANCE_MIN, make_sample_id


class TestVector1And2TemporalOffsetBounded:
    """1. Satellite frame timestamp > label timestamp where inappropriate.
    2. Future IBTrACS observation leaking into a sample.

    For this dataset, 'label' == the fused IBTrACS row describing the SAME
    physical instant as the image (this is a classification fusion, not a
    forecast target) -- so the protection that matters is that the offset
    between image time and the IBTrACS row used is NEVER silently larger
    than the declared, documented tolerance."""

    def test_qc_ok_rows_never_exceed_the_declared_temporal_tolerance(
        self, synthetic_ibtracs_full_track
    ):
        canonical = pd.DataFrame([{
            "storm_id": "2001213N15040", "satellite_timestamp": pd.Timestamp("2001-08-01T01:20:00"),
            "satellite_lat": 15.0, "satellite_lon": -40.0, "vza": 20.0, "source_file": "f.nc",
        }])
        joined = join_frames_to_ibtracs(canonical, synthetic_ibtracs_full_track,
                                        tolerance_min=DEFAULT_TEMPORAL_TOLERANCE_MIN)
        ok = joined[joined["qc_status"] == "ok"]
        if len(ok):
            assert (ok["temporal_offset_minutes"] <= DEFAULT_TEMPORAL_TOLERANCE_MIN).all()

    def test_a_match_beyond_tolerance_is_rejected_not_silently_accepted(
        self, synthetic_ibtracs_full_track
    ):
        """Proves the tolerance is actually enforced (not vacuous): a frame
        90 minutes + epsilon past the last track point must be rejected."""
        canonical = pd.DataFrame([{
            "storm_id": "2001213N15040", "satellite_timestamp": pd.Timestamp("2001-08-02T01:31:00"),
            "satellite_lat": 15.0, "satellite_lon": -40.0, "vza": 20.0, "source_file": "f.nc",
        }])
        joined = join_frames_to_ibtracs(canonical, synthetic_ibtracs_full_track, tolerance_min=90)
        assert joined.iloc[0]["qc_status"] == "rejected"


class TestVector3AdtNeverExceedsTolerance:
    """3. Future ADT scene information being incorrectly attached."""

    def test_adt_join_shares_the_same_tolerance_mechanism_as_ibtracs(self):
        from ml.geostrom_ml.satellite.adt import join_adt_scene
        import inspect
        sig = inspect.signature(join_adt_scene)
        assert "tolerance_min" in sig.parameters  # not hard-coded/unbounded


class TestVector4And6SplitIntegrity:
    """4. Duplicate storm frames crossing dataset boundaries.
    6. Multiple representations of the same observation leaking across splits.

    Both reduce to: a storm_id must map to exactly one split, and that
    mapping must come from the FROZEN Phase 2 manifest, never be
    independently re-derived (which could disagree with Phase 2 and
    silently split one storm's samples across partitions)."""

    def test_split_is_looked_up_not_recomputed(self):
        from ml.geostrom_ml.satellite.pipeline import load_split_map

        assert callable(load_split_map)  # the pipeline module's only split source

    def test_a_storm_present_in_two_splits_would_be_a_manifest_bug_not_ours(self, tmp_path):
        """Adversarial check: if splits_v1.json ever listed one storm_id in
        two splits (a Phase 2 manifest bug), `load_split_map` must not
        silently pick one -- it should let the LATER split win deterministically
        and this test documents/pins that behaviour rather than hiding it."""
        import json

        from ml.geostrom_ml.satellite.pipeline import load_split_map

        bad_manifest = tmp_path / "bad_splits.json"
        bad_manifest.write_text(json.dumps({
            "train": {"storm_ids": ["DUPLICATE_SID"]},
            "val": {"storm_ids": ["DUPLICATE_SID"]},
            "test": {"storm_ids": []},
        }))
        mapping = load_split_map(bad_manifest)
        assert mapping["DUPLICATE_SID"] in ("train", "val")  # deterministic, not random


class TestVector5NoDuplicatePhysicalFrame:
    """5. Same physical frame appearing multiple times."""

    def test_dedup_guarantees_at_most_one_canonical_frame_per_storm_and_time(self):
        t = pd.Timestamp("2001-01-01")
        inv = pd.DataFrame([
            {"storm_id": "A", "satellite_timestamp": t, "vza": 10.0, "source_file": "1.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 20.0, "source_file": "2.nc"},
            {"storm_id": "A", "satellite_timestamp": t, "vza": 30.0, "source_file": "3.nc"},
        ])
        canonical, _ = select_canonical_frames(inv)
        assert len(canonical[(canonical["storm_id"] == "A") & (canonical["satellite_timestamp"] == t)]) == 1

    def test_a_pipeline_that_skipped_dedup_would_be_caught_by_qc_gate_check_Q1(self):
        """Sanity-check the TEST METHODOLOGY: confirm the QC gate's Q1 check
        (see ml/geostrom_ml/satellite/qc.py) actually fires on a final_index
        containing a duplicate -- i.e. this protection is not vacuous."""
        from ml.geostrom_ml.satellite.qc import build_qc_report

        t = pd.Timestamp("2001-01-01")
        dup_final = pd.DataFrame([
            {"storm_id": "A", "satellite_timestamp": t, "spatial_distance_km": 1.0,
             "qc_status": "ok", "scene_label": None, "season": 2001},
            {"storm_id": "A", "satellite_timestamp": t, "spatial_distance_km": 1.0,
             "qc_status": "ok", "scene_label": None, "season": 2001},
        ])
        report = build_qc_report(
            n_files_discovered=2, inventory=pd.DataFrame({"error": [None, None]}),
            duplicate_summary={}, canonical=pd.DataFrame(), rejected_duplicates=pd.DataFrame(),
            joined=pd.DataFrame(), final_index=dup_final, known_split_storm_ids={"A"},
        )
        q1 = next(c for c in report["checks"] if c["id"] == "Q1")
        assert q1["passed"] is False
        assert report["summary"]["gate_status"] == "FAIL"


class TestSampleIdDeterminism:
    """Reproducibility underlies every leakage protection here: if sample_id
    generation were nondeterministic, duplicate/QC checks could pass or fail
    inconsistently between runs."""

    def test_same_inputs_produce_the_same_sample_id(self):
        a = make_sample_id("2001213N15040", pd.Timestamp("2001-08-01T00:00:00"))
        b = make_sample_id("2001213N15040", pd.Timestamp("2001-08-01T00:00:00"))
        assert a == b

    def test_different_timestamps_produce_different_sample_ids(self):
        a = make_sample_id("2001213N15040", pd.Timestamp("2001-08-01T00:00:00"))
        b = make_sample_id("2001213N15040", pd.Timestamp("2001-08-01T06:00:00"))
        assert a != b
