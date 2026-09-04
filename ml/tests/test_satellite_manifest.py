"""Dataset manifest: content, and the "no absolute machine-specific paths" rule."""

from __future__ import annotations

import pandas as pd

from ml.geostrom_ml.satellite.manifest import build_manifest
from ml.geostrom_ml.satellite.schema import SAMPLE_COLUMNS


def _final_index():
    return pd.DataFrame([{c: None for c in SAMPLE_COLUMNS}]).assign(
        scene_label="Eye", storm_id="2001213N15040")


def _qc_report():
    return {"counts": {"14_final_fused_samples": 1}, "summary": {"gate_status": "PASS"}}


class TestManifestContent:
    def test_required_top_level_fields_present(self, tmp_path):
        m = build_manifest(
            data_root=tmp_path, final_index=_final_index(), qc_report=_qc_report(),
            basin="NA", seasons_covered=[2001], storms_requested=["2001213N15040"],
            zarr_path=tmp_path / "processed" / "satellite" / "v1" / "images.zarr",
            parquet_path=tmp_path / "processed" / "satellite" / "v1" / "sample_index.parquet",
            splits_path_repo_relative="ml/manifests/splits_v1.json",
        )
        for key in ("dataset_version", "preprocessing_version", "generated_utc",
                    "source_datasets", "selection", "counts", "qc_thresholds",
                    "scene_labeled_samples", "gate_status", "storage", "split_source"):
            assert key in m

    def test_source_datasets_never_treat_adt_as_intensity_ground_truth(self, tmp_path):
        m = build_manifest(
            data_root=tmp_path, final_index=_final_index(), qc_report=_qc_report(),
            basin="NA", seasons_covered=[2001], storms_requested=[],
            zarr_path=tmp_path / "x.zarr", parquet_path=tmp_path / "x.parquet",
            splits_path_repo_relative="ml/manifests/splits_v1.json",
        )
        adt_entry = next(s for s in m["source_datasets"] if s["name"] == "ADT-HURSAT")
        assert "NEVER" in adt_entry["role"] or "never" in adt_entry["role"]

    def test_storage_paths_are_relative_to_data_root_not_absolute(self, tmp_path):
        m = build_manifest(
            data_root=tmp_path, final_index=_final_index(), qc_report=_qc_report(),
            basin="NA", seasons_covered=[2001], storms_requested=[],
            zarr_path=tmp_path / "processed" / "satellite" / "v1" / "images.zarr",
            parquet_path=tmp_path / "processed" / "satellite" / "v1" / "sample_index.parquet",
            splits_path_repo_relative="ml/manifests/splits_v1.json",
        )
        zarr_rel = m["storage"]["zarr_path_relative_to_data_root"]
        parquet_rel = m["storage"]["sample_index_path_relative_to_data_root"]
        assert not zarr_rel.startswith(str(tmp_path))
        assert ":" not in zarr_rel  # no Windows drive letter
        assert not parquet_rel.startswith(str(tmp_path))
        assert zarr_rel == "processed/satellite/v1/images.zarr"

    def test_split_source_reuses_phase2_manifest_not_a_new_split(self, tmp_path):
        m = build_manifest(
            data_root=tmp_path, final_index=_final_index(), qc_report=_qc_report(),
            basin="NA", seasons_covered=[2001], storms_requested=[],
            zarr_path=tmp_path / "x.zarr", parquet_path=tmp_path / "x.parquet",
            splits_path_repo_relative="ml/manifests/splits_v1.json",
        )
        assert m["split_source"]["manifest"] == "ml/manifests/splits_v1.json"
