"""Shared response envelopes."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class ProblemDetail(BaseModel):
    """RFC 7807 problem+json, per docs/API_ARCHITECTURE.md §2."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


class HealthStatus(BaseModel):
    status: str = Field(examples=["ok"])
    database: str = Field(examples=["ok"])


class MetaResponse(BaseModel):
    project_name: str
    api_version: str
    active_models: list[str]
    note: str = (
        "Retrospective research prototype. Predictions are historical "
        "baseline model output, not operational forecasts."
    )
