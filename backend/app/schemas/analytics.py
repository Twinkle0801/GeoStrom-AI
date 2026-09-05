"""Model-performance comparison schemas.

Per docs/API_ARCHITECTURE.md §3.6 ("GET /analytics/model-performance ... this
is the methodology page's core content and the project's honesty
guarantee") -- Phase 10 implements exactly this pre-existing, planned
endpoint. Every field is read verbatim from a committed `ml/reports/*.json`
benchmark file (Phase 2/5/6/7/8's own outputs); nothing here is recomputed
or invented, per Phase 10's explicit "do not fabricate scientific values"
rule.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

ModelTier = Literal["floor", "baseline", "exploratory"]


class ModelMetricEntry(BaseModel):
    model_name: str = Field(description="Internal model identifier, e.g. 'intensity_lightgbm'")
    display_name: str = Field(description="Human-readable name, e.g. 'LightGBM'")
    model_version: str
    tier: ModelTier = Field(
        description="'floor' = trivial reference (majority-class/persistence context), "
                    "'baseline' = a Tier-1 production-eligible model, "
                    "'exploratory' = a research model NOT recommended for production")
    is_recommended: bool = Field(
        description="True for exactly one model per task -- the current production baseline")
    metrics_by_horizon: dict[str, dict[str, float]] | None = Field(
        None, description="Intensity/track only: {'6': {...}, '12': {...}, '18': {...}, '24': {...}}")
    metrics: dict[str, float] | None = Field(
        None, description="Classification only: flat metrics (accuracy, macro_f1, ...)")


class TaskComparison(BaseModel):
    task: Literal["intensity", "track", "classification"]
    horizons_h: list[int] | None
    models: list[ModelMetricEntry]
    recommended_model: str = Field(description="display_name of the recommended production model")
    methodology_note: str


class ModelPerformanceResponse(BaseModel):
    generated_at: dt.datetime
    dataset_version: str
    split_version: str
    intensity: TaskComparison
    track: TaskComparison
    classification: TaskComparison
    disclaimer: str = (
        "Metrics are calculated on storm-disjoint held-out test data, per the project's frozen "
        "split. Retrospective evaluation only -- not a claim of operational forecasting skill."
    )
