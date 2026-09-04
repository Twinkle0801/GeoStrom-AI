"""Storm schemas. Coordinates are always named explicitly (`lat`/`lon`),
never a bare tuple, per docs/API_ARCHITECTURE.md §2 -- avoids any lat/lon
ordering ambiguity at the JSON boundary."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StormSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sid: str
    name: str | None
    season: int
    basin: str
    start_time: dt.datetime
    end_time: dt.datetime
    n_observations: int
    max_wind_kt: float | None = Field(None, description="Lifetime max wind (kt). DISPLAY ONLY.")
    min_pressure_hpa: float | None
    max_category: int | None
    made_landfall: bool | None
    split: str | None = Field(None, description="train | val | test (Phase 2 frozen split)")


class StormDetail(StormSummary):
    has_predictions: bool
    bbox: list[float] | None = Field(
        None, description="[min_lon, min_lat, max_lon, max_lat]")


class ObservationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ts: dt.datetime
    lat: float
    lon: float
    wind_kt: float | None
    pressure_hpa: float | None
    category: int | None
    nature: str | None
    storm_speed_kt: float | None
    storm_dir_deg: float | None
    dist2land_km: float | None
    is_synoptic: bool
    is_observed: bool
    data_kind: str = "observed"

    @field_validator("lat")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"lat={v} outside valid range [-90, 90]")
        return v

    @field_validator("lon")
    @classmethod
    def _lon_range(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"lon={v} outside valid range [-180, 180]")
        return v
