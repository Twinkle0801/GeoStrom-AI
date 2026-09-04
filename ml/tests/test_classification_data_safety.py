"""Phase 5 data safety: new classification artifacts are covered by the
existing (unmodified) .gitignore; committed docs/manifests/reports are not."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPO_ROOT  # noqa: E402


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


@pytest.mark.skipif(not _git_available(), reason="not a git repository / git unavailable")
class TestGitIgnoreCoversPhase5Artifacts:
    @pytest.mark.parametrize("rel_path", [
        "GeoStromData/processed/classification/scene_taxonomy_v1/classification_index.parquet",
    ])
    def test_data_artifact_paths_are_ignored(self, rel_path):
        assert _is_ignored(rel_path), f"{rel_path} is NOT git-ignored -- would be commit-able"

    @pytest.mark.parametrize("rel_path", [
        "ml/manifests/classification_dataset_v1_manifest.json",
        "ml/reports/phase5_scene_audit.json",
        "ml/reports/phase5_baseline_results.json",
        "ml/reports/figures/phase5_confusion_matrix.png",
        "docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md",
    ])
    def test_committed_artifact_paths_are_not_ignored(self, rel_path):
        assert not _is_ignored(rel_path), f"{rel_path} IS git-ignored -- would never be commit-able"
