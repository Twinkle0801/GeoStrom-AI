"""Evidence packet and Gemini response schemas.

Strongly typed per the task's explicit instruction ("avoid untyped
dictionaries wherever practical") and `docs/API_ARCHITECTURE.md` §7's
pre-existing evidence-packet design, which this module implements rather
than reinvents. The packet is versioned (`EVIDENCE_SCHEMA_VERSION`) so a
future schema change never silently reinterprets an older stored/cached
packet.

Every field here is either copied verbatim from a `Prediction`/`Storm`/
`Observation`/`ModelVersion` row already in the database (Phase 3's own
tables, never recomputed) or a small, deterministic derivation (a percentage
between two already-stored numbers). Nothing here is invented.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_SCHEMA_VERSION = "v1"


class StormEvidence(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sid: str
    name: str | None
    season: int
    basin: str
    start_time: dt.datetime
    end_time: dt.datetime
    n_observations: int


class CurrentStateEvidence(BaseModel):
    """The most recent OBSERVED state at or before the forecast origin time
    -- never a future/predicted value (this field must never be confused
    with `IntensityEvidence`/`TrackEvidence`'s `forecasts`, which are model
    output)."""

    timestamp: dt.datetime
    lat: float
    lon: float
    wind_kt: float | None
    pressure_hpa: float | None
    category: int | None
    storm_speed_kt: float | None
    storm_dir_deg: float | None
    dist2land_km: float | None


class HistoryPoint(BaseModel):
    timestamp: dt.datetime
    lat: float
    lon: float
    wind_kt: float | None


class IntensityForecastPoint(BaseModel):
    lead_hours: int
    pred_wind_kt: float | None
    true_wind_kt: float | None = Field(
        None, description="OBSERVED verification value, if the forecast is old enough to verify")
    wind_error_kt: float | None = Field(None, description="pred - observed (DERIVED, stored)")


class TrackForecastPoint(BaseModel):
    lead_hours: int
    pred_lat: float | None
    pred_lon: float | None
    error_radius_km: float | None = Field(
        None, description="Model's own historical error radius at this horizon (uncertainty cone)")
    true_lat: float | None = None
    true_lon: float | None = None
    track_error_km: float | None = Field(None, description="Great-circle, pred vs observed (DERIVED, stored)")


class ModelContext(BaseModel):
    """The model's MEASURED skill, verbatim from the committed benchmark
    report a `ModelVersion` row already carries -- never recomputed here.
    Per API_ARCHITECTURE.md §7: without this, an explanation describes a
    forecast as though certain; with it, Gemini can honestly say how good
    the model actually is."""

    model_name: str
    display_name: str = Field(
        description="Human-readable model name as this project's own docs describe it "
                    "(e.g. 'CLIPER-style Ridge' for 'track_cliper'), computed by a fixed, "
                    "deterministic lookup in the evidence builder -- never invented by Gemini")
    model_version: str
    dataset_version: str
    metrics_by_horizon: dict[str, dict] = Field(
        description="Verbatim ModelVersion.metrics, keyed by horizon string")
    skill_vs_persistence_pct: float | None = Field(
        None, description="Precomputed in the evidence builder from two ALREADY-STORED "
                          "metric values (this model's error vs. the stored persistence "
                          "model's error) -- never invented, never computed by Gemini")


class IntensityEvidence(BaseModel):
    origin_ts: dt.datetime
    forecasts: list[IntensityForecastPoint]
    context: ModelContext


class TrackEvidence(BaseModel):
    origin_ts: dt.datetime
    forecasts: list[TrackForecastPoint]
    context: ModelContext


class ClassificationEvidence(BaseModel):
    """Structurally supported per docs/API_ARCHITECTURE.md §7's evidence
    packet design and the task's evidence-field list. NOT wired to a
    per-storm production data source in Phase 9: `db/models.py` documents
    that no `classifications` table exists yet (Phase 5/6 introduced a
    classification model, but never a production per-storm classification
    record) -- adding one would be a database migration, which is out of
    Phase 9's explicitly minimal scope. The evidence builder therefore
    always leaves this `None` today; the field, the validator's grounding
    checks for it, and the fallback template's handling of it are all fully
    implemented and tested against a directly-constructed `EvidencePacket`,
    so wiring a real source later is a data-plumbing change, not a
    Gemini-integration change. Documented in
    docs/PHASE_9_GEMINI_INTEGRATION.md as a known limitation."""

    class_label: str
    confidence: float | None
    model_name: str
    model_version: str


class EvidencePacket(BaseModel):
    """The ENTIRE universe of facts available to Gemini for one call, per
    docs/API_ARCHITECTURE.md §7. Gemini receives nothing else -- no database
    access, no tool, no retrieval (docs/API_ARCHITECTURE.md §6.1/§8 Layer 1)."""

    evidence_schema_version: Literal["v1"] = EVIDENCE_SCHEMA_VERSION
    generated_at: dt.datetime
    storm: StormEvidence
    current_state: CurrentStateEvidence | None
    recent_history: list[HistoryPoint]
    intensity: IntensityEvidence | None
    track: TrackEvidence | None
    classification: ClassificationEvidence | None = None
    known_limitations: list[str]
    forbidden_claims: list[str]


class GeminiStructuredResponse(BaseModel):
    """The constrained response schema Gemini must produce (task §10 /
    docs/API_ARCHITECTURE.md §8 Layer 3: structured output narrows the space
    in which invention can occur)."""

    summary: str
    intensity_explanation: str
    track_explanation: str
    classification_explanation: str
    limitations: str


class ExplainRequest(BaseModel):
    """`sid` is this project's established per-storm identifier, used by
    every other endpoint since Phase 3 (`/api/v1/prediction/{sid}`,
    `/api/v1/tracks/{sid}`) -- used here instead of the task prompt's
    illustrative `session_id`, per the task's own instruction to follow the
    repository's existing naming conventions."""

    sid: str
    intensity_model_version: str | None = Field(
        None, description="Defaults to the recommended production intensity model (LightGBM v1)")
    track_model_version: str | None = Field(
        None, description="Defaults to the recommended production track model (CLIPER-style Ridge v1)")


class ModelRef(BaseModel):
    name: str
    version: str


class ExplainResponse(BaseModel):
    sid: str
    generated_at: dt.datetime
    evidence_schema_version: str

    intensity_model: ModelRef | None
    track_model: ModelRef | None
    classification_model: ModelRef | None = None

    source: Literal["gemini", "fallback"] = Field(
        description="Whether the explanation came from Gemini or the deterministic "
                    "fallback template -- the frontend must never have to guess this")
    fallback_reason: str | None = Field(
        None, description="Set only when source='fallback': e.g. 'not_configured', "
                          "'timeout', 'malformed_json', 'ungrounded_claim'")
    validation_violations: list[str] = Field(
        default_factory=list, description="Grounding-validator violation categories found in "
                                          "Gemini's (rejected) response, if any -- empty when "
                                          "source='gemini' or when Gemini was never called")

    explanation: GeminiStructuredResponse

    disclaimer: str = (
        "Retrospective research-prototype model output. Not an operational forecast, "
        "weather warning, or safety advisory."
    )
