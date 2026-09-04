"""Prediction and model-version schemas.

Every prediction-bearing response carries `model_name`/`model_version`
(never bare "the prediction"), per UI_UX_ARCHITECTURE.md's ModelBadge rule
and docs/API_ARCHITECTURE.md §2 ("Every model-derived payload carries
model_version"). `data_kind` distinguishes OBSERVED from MODEL PREDICTION
explicitly in the payload, not just by field name, so a client cannot
mis-render one as the other.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    task: str = Field(description="'track' or 'intensity'")
    dataset_build: str
    metrics: dict = Field(description="Verbatim Phase 2 benchmark metrics, keyed by horizon")
    error_radii_km: dict | None = Field(
        None, description="Track models only: mean historical error (km) per horizon, "
                          "for rendering an uncertainty envelope")
    is_active: bool


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task: str
    origin_ts: dt.datetime = Field(description="Forecast initialization time")
    lead_hours: int
    valid_ts: dt.datetime = Field(description="origin_ts + lead_hours")
    model_name: str
    model_version: str

    pred_lat: float | None
    pred_lon: float | None
    pred_wind_kt: float | None
    pred_pressure_hpa: float | None
    error_radius_km: float | None

    true_lat: float | None = Field(None, description="OBSERVED ground truth at valid_ts")
    true_lon: float | None
    true_wind_kt: float | None
    track_error_km: float | None = Field(
        None, description="Great-circle distance, predicted vs. observed (DERIVED)")
    wind_error_kt: float | None = Field(
        None, description="predicted - observed wind (DERIVED)")

    data_kind: str = "model_prediction"
    disclaimer: str = "Historical baseline model prediction, not an operational forecast."

    @field_validator("pred_lat", "true_lat")
    @classmethod
    def _lat_range(cls, v: float | None) -> float | None:
        if v is not None and not (-90.0 <= v <= 90.0):
            raise ValueError(f"latitude={v} outside valid range [-90, 90]")
        return v

    @field_validator("pred_lon", "true_lon")
    @classmethod
    def _lon_range(cls, v: float | None) -> float | None:
        if v is not None and not (-180.0 <= v <= 180.0):
            raise ValueError(f"longitude={v} outside valid range [-180, 180]")
        return v
