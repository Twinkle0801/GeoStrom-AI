"""Repository layer: the only place raw SQLAlchemy queries live.

Per docs/API_ARCHITECTURE.md §12: database model -> repository/service ->
Pydantic schema -> JSON response. Routes never touch a `Session` query
directly and never return an ORM object.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ModelVersion, Observation, Prediction, Storm


def list_storms(
    db: Session, *, season: int | None = None, basin: str | None = None,
    split: str | None = None, q: str | None = None,
    limit: int = 50, offset: int = 0,
) -> tuple[list[Storm], int]:
    stmt = select(Storm)
    if season is not None:
        stmt = stmt.where(Storm.season == season)
    if basin is not None:
        stmt = stmt.where(Storm.basin == basin)
    if split is not None:
        stmt = stmt.where(Storm.split == split)
    if q:
        stmt = stmt.where(Storm.sid.ilike(f"%{q}%"))

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    stmt = stmt.order_by(Storm.start_time.desc()).limit(limit).offset(offset)
    items = list(db.scalars(stmt).all())
    return items, total


def get_storm(db: Session, sid: str) -> Storm | None:
    return db.get(Storm, sid)


def storm_has_predictions(db: Session, sid: str) -> bool:
    stmt = select(func.count()).select_from(Prediction).where(Prediction.sid == sid)
    return (db.scalar(stmt) or 0) > 0


def list_observations(db: Session, sid: str) -> list[Observation]:
    stmt = select(Observation).where(Observation.sid == sid).order_by(Observation.ts)
    return list(db.scalars(stmt).all())


def list_predictions(
    db: Session, sid: str, *, origin_ts=None, task: str | None = None,
    model_name: str | None = None,
) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.sid == sid)
        .join(ModelVersion, Prediction.model_id == ModelVersion.id)
    )
    if origin_ts is not None:
        stmt = stmt.where(Prediction.origin_ts == origin_ts)
    if task is not None:
        stmt = stmt.where(Prediction.task == task)
    if model_name is not None:
        stmt = stmt.where(ModelVersion.name == model_name)
    stmt = stmt.order_by(Prediction.origin_ts, Prediction.lead_hours)
    return list(db.scalars(stmt).all())


def latest_origin_ts(db: Session, sid: str):
    stmt = select(func.max(Prediction.origin_ts)).where(Prediction.sid == sid)
    return db.scalar(stmt)


def list_model_versions(db: Session, *, task: str | None = None) -> list[ModelVersion]:
    stmt = select(ModelVersion)
    if task is not None:
        stmt = stmt.where(ModelVersion.task == task)
    return list(db.scalars(stmt.order_by(ModelVersion.task, ModelVersion.name)).all())


def active_model_names(db: Session) -> list[str]:
    stmt = select(ModelVersion.name, ModelVersion.version).where(ModelVersion.is_active.is_(True))
    return [f"{n}_{v}" for n, v in db.execute(stmt).all()]
