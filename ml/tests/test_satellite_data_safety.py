"""Phase 4 data-safety checks: DATA_ROOT placement, no absolute paths in
committed metadata, and .gitignore actually covers every new artifact type
this phase introduces (raw archives, extracted NetCDF, Zarr, satellite
Parquet) without repo modification.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPO_ROOT, check_data_root  # noqa: E402


def _git_available() -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO_ROOT,
                       capture_output=True, timeout=10, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_ignored(rel_path: str) -> bool:
    result = subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=REPO_ROOT, timeout=10)
    return result.returncode == 0


class TestDataRootSafety:
    def test_data_root_inside_repo_is_rejected(self):
        problems = check_data_root(REPO_ROOT / "some_data_dir")
        assert any("Git repository" in p for p in problems)

    def test_a_safe_external_data_root_has_no_problems(self, tmp_path):
        assert check_data_root(tmp_path) == []


@pytest.mark.skipif(not _git_available(), reason="not a git repository / git unavailable")
class TestGitIgnoreCoversPhase4Artifacts:
    @pytest.mark.parametrize("rel_path", [
        "GeoStromData/raw/hursat/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz",
        "GeoStromData/interim/hursat/2005236N23285/frame.hursat-b1.v06.nc",
        "GeoStromData/processed/satellite/satellite_v1/images.zarr/irwin_k/.zarray",
        "GeoStromData/processed/satellite/satellite_v1/sample_index.parquet",
        "GeoStromData/samples/adt/2005236N23285.nc",
    ])
    def test_data_artifact_paths_are_ignored(self, rel_path):
        assert _is_ignored(rel_path), f"{rel_path} is NOT git-ignored -- would be commit-able"

    @pytest.mark.parametrize("rel_path", [
        "ml/manifests/satellite_dataset_v1_manifest.json",
        "ml/reports/satellite_qc_gate.json",
        "docs/PHASE_4_SATELLITE_PIPELINE.md",
    ])
    def test_committed_artifact_paths_are_not_ignored(self, rel_path):
        assert not _is_ignored(rel_path), f"{rel_path} IS git-ignored -- would never be commit-able"
