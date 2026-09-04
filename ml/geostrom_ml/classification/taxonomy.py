"""The Phase 5 MVP classification taxonomy: `scene_taxonomy_v1`.

============================================================================
SOURCE GROUND TRUTH (never invented, never modified)
============================================================================

The `scene_label` column in Phase 4's sample index is the genuine ADT-HURSAT
`Scene` field, retrieved verbatim from real NCEI ADT-HURSAT files
(`ml/geostrom_ml/satellite/adt.py::join_adt_scene`). This module NEVER
overwrites `scene_label` -- it only computes a derived `final_class` (or an
exclusion reason) alongside it, both persisted in the classification index
(`ml/geostrom_ml/classification/dataset.py`) so `original_scene` is always
traceable.

============================================================================
SEMANTIC EVIDENCE (cited, not guessed)
============================================================================

Two independent, pre-existing project sources agree on what these 8 labels
mean, before this phase was ever started:

1. `docs/PROJECT_REQUIREMENTS.md` Tier B (line ~101, a Phase 0 planning
   decision, later promoted to primary target by
   `docs/DATA_STRATEGY.md` decision #14 once Phase 1 confirmed the field
   exists): "Eye, Embedded Centre, Central Dense Overcast, Irregular CDO,
   Curved Band, Shear, Uniform" -- explicitly named as the ADT/Dvorak
   scene-type label set.
2. `ml/scripts/verify_adt.py` (Phase 1, verified against real ADT files):
   ```
   EYE_SCENE   = {0: "Eye", 1: "Pinhole Eye", 2: "Large Eye", 3: "No Eye"}
   CLOUD_SCENE = {0: "CDO", 1: "Embedded Center", 2: "Irregular CDO",
                  3: "Curved Band", 4: "Shear"}
   ```
   with the Scene field itself documented as "a combination of EyeScene and
   CloudScene integer codes" (`verify_adt.py`'s own report metadata,
   `scene_field.derivation`).

Every one of the 8 real observed `scene_label` values maps cleanly onto the
union of these two code tables:

| Observed label | Source table | Code | Meaning (per source) |
|---|---|---|---|
| CDO | CLOUD_SCENE | 0 | Central Dense Overcast |
| EmbCenter | CLOUD_SCENE | 1 | Embedded Center |
| IrrCDO | CLOUD_SCENE | 2 | Irregular CDO (shape variant of CDO) |
| CurvedBand | CLOUD_SCENE | 3 | Curved Band |
| Shear | CLOUD_SCENE | 4 | Shear |
| Eye | EYE_SCENE | 0 | Eye |
| LargeEye | EYE_SCENE | 2 | Large Eye (size variant of Eye) |
| Land | *(neither table)* | -- | see below |

`Land` appears in NEITHER code table and is absent from the Phase 0 Tier B
list. `verify_adt.py` itself treats it specially: it is counted separately
(`land_records`, `pct_land_records`) and explicitly EXCLUDED from the
"real" scene distribution it reports (`scene_distribution_excluding_land`).
This is documented codebase evidence, not a Phase 5 guess, that `Land`
functions as a geographic/QC override (the ADT algorithm reporting the
storm center sits over land, where Dvorak pattern analysis is not
meaningful) rather than a genuine storm-structure category. Confidence:
HIGH, evidence-based -- but no literal NCEI prose definition of `Land` was
found in this project's documentation, so this inference is stated with
that caveat rather than presented as directly quoted fact.

`EyeScene` code 1 ("Pinhole Eye") and code 3 ("No Eye") never occur as an
observed `scene_label` value in the real dataset (expected: "No Eye" implies
Scene falls back to CloudScene instead; "Pinhole Eye" simply never occurred
in this small 12-storm sample -- absence of evidence, not evidence of
absence, noted honestly rather than silently ignored).

============================================================================
DERIVED LABEL -- scene_taxonomy_v1
============================================================================

This is a DERIVED grouping of the genuine ADT Scene ground truth, not a new
ground truth and not a heuristic re-interpretation. The task's explicit
warning against `Developing/Organized/Mature/Weakening`-style labels
(`docs/PROJECT_REQUIREMENTS.md`'s "Tier C", explicitly "engineered, not
ground truth", "risk of circularity: the model learns our heuristic, not
nature") does NOT apply here: every merge below groups sub-codes the ADT
algorithm's OWN internal code tables already recognise as the same family
(EyeScene 0+2 = "Eye" family; CloudScene 0+2 = "CDO" family), never an
externally-invented organisational-quality axis.

  HEURISTIC ASSUMPTION being made: that a size/shape sub-variant (Large Eye
  vs Eye; Irregular vs regular CDO) is close enough to its parent category
  for an MVP classifier that, per the storm-level audit
  (`ml/reports/phase5_scene_audit.json`), cannot reliably distinguish either
  member of either pair on 3-6 storms alone. This is a data-support decision,
  not a meteorological claim that the size/shape distinction is
  unimportant -- documented so a future phase with more storms can split
  them back out.

  CIRCULARITY RISK: none identified. `final_class` is computed by a pure
  lookup table keyed on `scene_label` alone (see `SCENE_TAXONOMY_V1` below)
  -- it uses no image content, no model output, and no information the
  original ADT algorithm did not already produce.

Mapping (see `ml/reports/phase5_scene_audit.json` and
`docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md` for the full storm/sample
evidence behind each decision):

  CDO        <- {CDO, IrrCDO}          merge: IrrCDO is a CLOUD_SCENE shape
                                        variant of CDO; alone IrrCDO has only
                                        11 samples / 6 storms (0 in val).
  CurvedBand <- {CurvedBand}           kept as-is: best-represented class,
                                        present in 12/12 storms (100%).
  Shear      <- {Shear}                kept as-is: 83 samples / 10 storms.
  Eye        <- {Eye, LargeEye}        merge: LargeEye is an EYE_SCENE size
                                        variant of Eye; alone LargeEye has
                                        only 14 samples / 3 storms (0 in
                                        test). NOTE: even merged, the Eye
                                        family still has ZERO test-split
                                        samples -- the 3 frozen test storms
                                        simply never exhibit an eye. This is
                                        a genuine dataset limitation, not
                                        fixed by this merge; documented, not
                                        hidden (see the Phase 5 doc).
  EXCLUDED   <- EmbCenter              reason=insufficient_label_support:
                                        17 samples / 5 storms, only 1 sample
                                        in the entire test split -- too thin
                                        for any reliable per-class metric.
                                        Not merged into CDO or Eye because
                                        CLOUD_SCENE lists it as its own
                                        co-equal code (1), not a documented
                                        variant of either.
  EXCLUDED   <- Land                   reason=land_contaminated: not a
                                        storm-pattern category (see evidence
                                        above).
"""

from __future__ import annotations

LABEL_VERSION = "scene_taxonomy_v1"

# Exclusion reasons -- explicit, machine-readable, never a silent drop.
EXCLUSION_LAND_CONTAMINATED = "land_contaminated"
EXCLUSION_INSUFFICIENT_SUPPORT = "insufficient_label_support"
EXCLUSION_UNRESOLVED_MAPPING = "unresolved_mapping"
EXCLUSION_INVALID_QC = "invalid_qc"

# scene_label -> final_class. A value of None means "excluded"; the reason
# is looked up from EXCLUSION_REASONS below. This is the ENTIRE derivation
# -- a pure lookup table, no image content, no model output (see module
# docstring, "CIRCULARITY RISK").
SCENE_TAXONOMY_V1: dict[str, str | None] = {
    "CDO": "CDO",
    "IrrCDO": "CDO",
    "CurvedBand": "CurvedBand",
    "Shear": "Shear",
    "Eye": "Eye",
    "LargeEye": "Eye",
    "EmbCenter": None,
    "Land": None,
}

EXCLUSION_REASONS: dict[str, str] = {
    "EmbCenter": EXCLUSION_INSUFFICIENT_SUPPORT,
    "Land": EXCLUSION_LAND_CONTAMINATED,
}

# The final, orderable class list for this taxonomy version (used for
# stable confusion-matrix / report ordering everywhere downstream).
FINAL_CLASSES_V1: list[str] = ["CDO", "CurvedBand", "Eye", "Shear"]


def apply_taxonomy(scene_label: str | None) -> tuple[str | None, str | None]:
    """Map one original `scene_label` to (final_class, exclusion_reason).

    Exactly one of the two return values is non-None (unless the label is
    itself unrecognised, in which case both fields report
    `unresolved_mapping` so an unexpected future label is never silently
    dropped or silently classified).
    """
    if scene_label is None or (isinstance(scene_label, float) and scene_label != scene_label):
        return None, EXCLUSION_UNRESOLVED_MAPPING  # NaN/missing label
    if scene_label not in SCENE_TAXONOMY_V1:
        return None, EXCLUSION_UNRESOLVED_MAPPING
    final = SCENE_TAXONOMY_V1[scene_label]
    if final is None:
        return None, EXCLUSION_REASONS.get(scene_label, EXCLUSION_UNRESOLVED_MAPPING)
    return final, None
