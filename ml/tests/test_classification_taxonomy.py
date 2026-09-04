"""scene_taxonomy_v1: mapping correctness, exclusion reasons, determinism."""

from __future__ import annotations

from ml.geostrom_ml.classification.taxonomy import (
    EXCLUSION_INSUFFICIENT_SUPPORT,
    EXCLUSION_LAND_CONTAMINATED,
    EXCLUSION_UNRESOLVED_MAPPING,
    FINAL_CLASSES_V1,
    LABEL_VERSION,
    SCENE_TAXONOMY_V1,
    apply_taxonomy,
)


class TestMapping:
    def test_all_eight_real_labels_are_mapped(self):
        real_labels = {"CDO", "CurvedBand", "Land", "Shear", "Eye",
                       "EmbCenter", "LargeEye", "IrrCDO"}
        assert real_labels == set(SCENE_TAXONOMY_V1.keys())

    def test_cdo_family_merges_into_cdo(self):
        assert apply_taxonomy("CDO") == ("CDO", None)
        assert apply_taxonomy("IrrCDO") == ("CDO", None)

    def test_eye_family_merges_into_eye(self):
        assert apply_taxonomy("Eye") == ("Eye", None)
        assert apply_taxonomy("LargeEye") == ("Eye", None)

    def test_curvedband_and_shear_are_unchanged(self):
        assert apply_taxonomy("CurvedBand") == ("CurvedBand", None)
        assert apply_taxonomy("Shear") == ("Shear", None)

    def test_land_is_excluded_as_land_contaminated(self):
        final, reason = apply_taxonomy("Land")
        assert final is None
        assert reason == EXCLUSION_LAND_CONTAMINATED

    def test_embcenter_is_excluded_as_insufficient_support(self):
        final, reason = apply_taxonomy("EmbCenter")
        assert final is None
        assert reason == EXCLUSION_INSUFFICIENT_SUPPORT

    def test_unrecognised_label_is_unresolved_not_silently_dropped(self):
        final, reason = apply_taxonomy("SomeFutureLabelNeverSeenBefore")
        assert final is None
        assert reason == EXCLUSION_UNRESOLVED_MAPPING

    def test_missing_label_is_unresolved(self):
        assert apply_taxonomy(None) == (None, EXCLUSION_UNRESOLVED_MAPPING)
        assert apply_taxonomy(float("nan")) == (None, EXCLUSION_UNRESOLVED_MAPPING)


class TestFinalClasses:
    def test_final_classes_match_the_non_excluded_mapping_targets(self):
        mapped_targets = {v for v in SCENE_TAXONOMY_V1.values() if v is not None}
        assert mapped_targets == set(FINAL_CLASSES_V1)

    def test_four_final_classes(self):
        assert len(FINAL_CLASSES_V1) == 4


class TestVersioning:
    def test_label_version_is_the_documented_string(self):
        assert LABEL_VERSION == "scene_taxonomy_v1"

    def test_determinism_same_input_same_output(self):
        for label in SCENE_TAXONOMY_V1:
            assert apply_taxonomy(label) == apply_taxonomy(label)
