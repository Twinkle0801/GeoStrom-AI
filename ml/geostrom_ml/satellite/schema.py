"""Canonical schema for the Phase 4 satellite sample index.

The column list below is the Phase 4 task's explicit minimum schema, plus a
small number of additions -- each justified inline -- needed to make the
index usable without re-deriving information that is expensive or ambiguous
to recompute:

  * ``split``       -- looked up from the frozen `ml/manifests/splits_v1.json`
                        storm-level split (Phase 2). Required so a satellite
                        DataLoader can honour the frozen split without joining
                        against the Phase 2 manifest at load time.
  * ``season``       -- IBTrACS SEASON of the storm. Cheap to store, avoids
                        re-parsing it out of `storm_id` for QC/report grouping.
  * ``zarr_index``    -- row index of this sample's image in the canonical
                        Zarr store (`processed/satellite/<version>/images.zarr`).
                        Without this the metadata index cannot locate its own
                        imagery.
  * ``preprocessing_version`` -- the code/config version that produced this
                        row (distinct from ``dataset_version``, which is the
                        dataset release identifier). Needed for reproducibility
                        audits per the Phase 4 task's explicit requirement.

No other fields are added. In particular, no meteorological features beyond
what identifies/QCs the fused observation are duplicated here -- feature
engineering for the eventual CNN belongs to a later phase.
"""

from __future__ import annotations

DATASET_VERSION = "satellite_v1"
PREPROCESSING_VERSION = "v1"

# Physically valid IRWIN brightness-temperature range, in Kelvin.
# LOCKED in Phase 1 / docs/DATA_STRATEGY.md decision #15 and used unchanged
# in ml/scripts/qc_gate.py::IRWIN_PHYSICAL_K. Anything outside this range is
# masked as invalid, never trusted -- this is a physical floor, not a
# fill-value equality check. Re-used verbatim here per the Phase 4 task's
# explicit instruction not to invent a new threshold.
IRWIN_VALID_RANGE_K = (150.0, 350.0)

# Great-circle separation above which a HURSAT<->IBTrACS join is rejected.
# LOCKED in Phase 1 (docs/DATA_STRATEGY.md pipeline check #1: "target < 50 km",
# measured max separation on the verification sample was 10.65 km). The Phase
# 4 task explicitly forbids loosening this threshold.
SPATIAL_QC_KM = 50.0

# Maximum |dt| between a HURSAT frame's nominal time and the nearest IBTrACS
# observation for a join to be attempted at all. Phase 1 established this as
# a hypothesis (+/-90 min) and measured HURSAT frames landing at |dt|=0 for
# 100% of the sample; ADT true scan times use the same tolerance per
# docs/DATA_STRATEGY.md check table. Configurable, not hard-coded, per the
# Phase 4 task's explicit instruction.
DEFAULT_TEMPORAL_TOLERANCE_MIN = 90

SAMPLE_COLUMNS: list[str] = [
    "sample_id",
    "storm_id",
    "satellite_timestamp",
    "ibtracs_timestamp",
    "adt_timestamp",
    "satellite_lat",
    "satellite_lon",
    "ibtracs_lat",
    "ibtracs_lon",
    "spatial_distance_km",
    "temporal_offset_minutes",
    "vza",
    "satellite_id",
    "source_file",
    "source_checksum",
    "channel",
    "image_height",
    "image_width",
    "scene_label",
    "usa_wind",
    "pressure_if_valid",
    "is_observed",
    "is_interpolated",
    "qc_status",
    "qc_reason",
    "dataset_version",
    # Justified additions -- see module docstring.
    "split",
    "season",
    "zarr_index",
    "preprocessing_version",
]

SAMPLE_DTYPES: dict[str, str] = {
    "sample_id": "string",
    "storm_id": "string",
    "satellite_timestamp": "datetime64[ns]",
    "ibtracs_timestamp": "datetime64[ns]",
    "adt_timestamp": "datetime64[ns]",
    "satellite_lat": "float64",
    "satellite_lon": "float64",
    "ibtracs_lat": "float64",
    "ibtracs_lon": "float64",
    "spatial_distance_km": "float64",
    "temporal_offset_minutes": "float64",
    "vza": "float64",
    "satellite_id": "string",
    "source_file": "string",
    "source_checksum": "string",
    "channel": "string",
    "image_height": "Int64",
    "image_width": "Int64",
    "scene_label": "string",
    "usa_wind": "float64",
    "pressure_if_valid": "float64",
    "is_observed": "boolean",
    "is_interpolated": "boolean",
    "qc_status": "string",
    "qc_reason": "string",
    "dataset_version": "string",
    "split": "string",
    "season": "Int64",
    "zarr_index": "Int64",
    "preprocessing_version": "string",
}


def make_sample_id(storm_id: str, satellite_timestamp) -> str:
    """Deterministic sample_id: `<storm_id>_<UTC timestamp, second precision>`.

    Deterministic (no randomness, no counters) so re-running the pipeline on
    unchanged inputs reproduces identical sample_ids -- required for the
    idempotency / reproducibility checks.
    """
    import pandas as pd

    ts = pd.Timestamp(satellite_timestamp)
    return f"{storm_id}_{ts.strftime('%Y%m%dT%H%M%SZ')}"
