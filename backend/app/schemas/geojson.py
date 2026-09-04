"""Minimal, explicit GeoJSON schemas for the /tracks endpoint.

docs/SYSTEM_ARCHITECTURE.md §6.2: one FeatureCollection per track request,
typed features distinguished by `properties.kind`. GeoJSON coordinate order
is [lon, lat] per the RFC -- enforced here by construction (services/geometry.py
is the only place a Point/LineString is built), never left to the caller.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FeatureKind = Literal[
    "observed_track", "observed_point", "current_position",
    "predicted_track", "predicted_point",
]


class PointGeometry(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float] = Field(description="[lon, lat]")


class LineStringGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]] = Field(description="[[lon, lat], ...]")


class Feature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry | LineStringGeometry
    properties: dict = Field(default_factory=dict)


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature]
