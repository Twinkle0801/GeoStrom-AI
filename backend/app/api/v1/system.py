from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import get_db
from app.repositories.storms import active_model_names
from app.schemas.common import HealthStatus, MetaResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus)
def health(db: Session = Depends(get_db)) -> HealthStatus:
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:  # noqa: BLE001
        db_status = "unreachable"
    return HealthStatus(status="ok", database=db_status)


@router.get("/api/v1/meta", response_model=MetaResponse)
def meta(db: Session = Depends(get_db)) -> MetaResponse:
    settings = get_settings()
    return MetaResponse(
        project_name=settings.project_name,
        api_version=settings.api_version,
        active_models=active_model_names(db),
    )
