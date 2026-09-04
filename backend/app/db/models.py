"""SQLAlchemy ORM models.

Schema follows docs/SYSTEM_ARCHITECTURE.md §7.2 ("Preliminary schema")
directly, scoped to what Phase 3 actually needs: `model_versions`, `storms`,
`observations`, `predictions`. The `detections`, `classifications`, and
`forecast_cones` tables from the preliminary schema are NOT created here —
no detection or classification model exists yet (Phase 5/6, not built), and
Phase 3's task brief explicitly excludes CNN/vision work. Adding empty
tables for models that don't exist would be speculative schema, which the
project's own scope-creep guard warns against.

`predictions` is LONG FORM per the Phase 0 decision, re-stated explicitly by
the Phase 3 task: one row = one storm + one initialization time + one
horizon + one model. Adding a new horizon is an INSERT, never a migration.

One deliberate, documented addition over the SYSTEM_ARCHITECTURE.md sketch:
a `task` column on `predictions` ('track' | 'intensity'), because Phase 2
produced two disjoint model families (track models predict position, never
wind; intensity models predict wind, never position) and `task` makes that
partition queryable without inspecting which columns are non-null.
"""

from __future__ import annotations

import datetime as dt

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, Integer, JSON, SmallInteger,
    String, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)          # e.g. 'track_cliper'
    version: Mapped[str] = mapped_column(String, nullable=False)       # e.g. 'v1'
    task: Mapped[str] = mapped_column(String, nullable=False)          # 'track' | 'intensity'
    trained_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    dataset_build: Mapped[str] = mapped_column(String, nullable=False)  # Phase 2 dataset_version
    split_version: Mapped[str | None] = mapped_column(String)
    feature_version: Mapped[str | None] = mapped_column(String)
    git_commit: Mapped[str | None] = mapped_column(String)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # metrics: {"6": {"mae_kt": .., ...}, "12": {...}, ...} -- verbatim from
    # ml/reports/phase2_benchmark_results.json, never recomputed here.
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_radii_km: Mapped[dict | None] = mapped_column(JSON)  # track models only
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Storm(Base):
    __tablename__ = "storms"

    sid: Mapped[str] = mapped_column(String, primary_key=True)  # IBTrACS SID
    name: Mapped[str | None] = mapped_column(String)
    season: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    basin: Mapped[str] = mapped_column(String, nullable=False)
    subbasin: Mapped[str | None] = mapped_column(String)
    start_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    n_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    # Lifetime summary fields -- DISPLAY ONLY. Never fed into a model feature
    # pipeline (that would leak the future into early timesteps); the ML
    # layer computes running max-wind-so-far from `observations`, never this.
    max_wind_kt: Mapped[float | None] = mapped_column()
    min_pressure_hpa: Mapped[float | None] = mapped_column()
    max_category: Mapped[int | None] = mapped_column(SmallInteger)
    made_landfall: Mapped[bool | None] = mapped_column(Boolean)
    split: Mapped[str | None] = mapped_column(String)  # train|val|test, from splits_v1.json
    track_geom: Mapped[str | None] = mapped_column(Geography(geometry_type="LINESTRING", srid=4326))
    bbox: Mapped[str | None] = mapped_column(Geography(geometry_type="POLYGON", srid=4326))

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="storm", order_by="Observation.ts", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="storm", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("sid", "ts", name="uq_observation_sid_ts"),
        CheckConstraint("lat >= -90 AND lat <= 90", name="ck_observation_lat_range"),
        CheckConstraint("lon >= -180 AND lon <= 180", name="ck_observation_lon_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sid: Mapped[str] = mapped_column(ForeignKey("storms.sid", ondelete="CASCADE"), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    lat: Mapped[float] = mapped_column(nullable=False)
    lon: Mapped[float] = mapped_column(nullable=False)
    geom: Mapped[str] = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=False)
    wind_kt: Mapped[float | None] = mapped_column()
    pressure_hpa: Mapped[float | None] = mapped_column()
    category: Mapped[int | None] = mapped_column(SmallInteger)
    nature: Mapped[str | None] = mapped_column(String)
    storm_speed_kt: Mapped[float | None] = mapped_column()
    storm_dir_deg: Mapped[float | None] = mapped_column()
    dist2land_km: Mapped[float | None] = mapped_column()
    is_synoptic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_observed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    storm: Mapped["Storm"] = relationship(back_populates="observations")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("sid", "origin_ts", "lead_hours", "model_id",
                         name="uq_prediction_sid_origin_lead_model"),
        CheckConstraint("pred_lat IS NULL OR (pred_lat >= -90 AND pred_lat <= 90)",
                        name="ck_prediction_pred_lat_range"),
        CheckConstraint("pred_lon IS NULL OR (pred_lon >= -180 AND pred_lon <= 180)",
                        name="ck_prediction_pred_lon_range"),
        CheckConstraint("true_lat IS NULL OR (true_lat >= -90 AND true_lat <= 90)",
                        name="ck_prediction_true_lat_range"),
        CheckConstraint("true_lon IS NULL OR (true_lon >= -180 AND true_lon <= 180)",
                        name="ck_prediction_true_lon_range"),
        CheckConstraint("lead_hours > 0", name="ck_prediction_lead_hours_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sid: Mapped[str] = mapped_column(ForeignKey("storms.sid", ondelete="CASCADE"), nullable=False)
    task: Mapped[str] = mapped_column(String, nullable=False)  # 'track' | 'intensity'
    origin_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lead_hours: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    valid_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)

    pred_lat: Mapped[float | None] = mapped_column()
    pred_lon: Mapped[float | None] = mapped_column()
    pred_geom: Mapped[str | None] = mapped_column(Geography(geometry_type="POINT", srid=4326))
    pred_wind_kt: Mapped[float | None] = mapped_column()
    pred_pressure_hpa: Mapped[float | None] = mapped_column()

    error_radius_km: Mapped[float | None] = mapped_column()

    # Ground truth + error -- populated because the system is retrospective.
    # `true_*` is the SAME observed value the `observations` table carries
    # for that (sid, valid_ts); duplicated here for cheap single-query reads,
    # never independently sourced.
    true_lat: Mapped[float | None] = mapped_column()
    true_lon: Mapped[float | None] = mapped_column()
    true_wind_kt: Mapped[float | None] = mapped_column()
    track_error_km: Mapped[float | None] = mapped_column()   # Haversine, same formula as ml/
    wind_error_kt: Mapped[float | None] = mapped_column()

    storm: Mapped["Storm"] = relationship(back_populates="predictions")
    model: Mapped["ModelVersion"] = relationship(back_populates="predictions")
