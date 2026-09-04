"""GET /api/v1/prediction/{sid} -- scalar forecast records for one origin.

Per docs/API_ARCHITECTURE.md §3.5. Powers the intensity panel and forecast
table -- `/tracks/{sid}` is the map's GeoJSON payload; this is the typed,
per-horizon numeric view of the same underlying rows.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.repositories import storms as repo
from app.schemas.common import ProblemDetail
from app.schemas.prediction import ModelVersionOut, PredictionOut

router = APIRouter(prefix="/api/v1/prediction", tags=["prediction"])


def _to_out(p, model_by_id) -> PredictionOut:
    m = model_by_id[p.model_id]
    return PredictionOut(
        task=p.task, origin_ts=p.origin_ts, lead_hours=p.lead_hours, valid_ts=p.valid_ts,
        model_name=m.name, model_version=m.version,
        pred_lat=p.pred_lat, pred_lon=p.pred_lon,
        pred_wind_kt=p.pred_wind_kt, pred_pressure_hpa=p.pred_pressure_hpa,
        error_radius_km=p.error_radius_km,
        true_lat=p.true_lat, true_lon=p.true_lon, true_wind_kt=p.true_wind_kt,
        track_error_km=p.track_error_km, wind_error_kt=p.wind_error_kt,
    )


@router.get(
    "/{sid}", response_model=list[PredictionOut],
    responses={404: {"model": ProblemDetail}},
)
def get_prediction(
    sid: str,
    t: dt.datetime | None = Query(
        None, description="Forecast origin timestamp. Defaults to the latest available."),
    task: str | None = Query(None, description="'track' or 'intensity'"),
    model: str | None = Query(None, description="Model name, e.g. 'track_cliper'"),
    db: Session = Depends(get_db),
) -> list[PredictionOut]:
    if repo.get_storm(db, sid) is None:
        raise HTTPException(status_code=404, detail=f"Storm '{sid}' not found")

    origin_ts = t or repo.latest_origin_ts(db, sid)
    if origin_ts is None:
        return []

    preds = repo.list_predictions(db, sid, origin_ts=origin_ts, task=task, model_name=model)
    if not preds:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast issued for '{sid}' at {origin_ts.isoformat()} "
                   f"(task={task or 'any'}, model={model or 'any'})",
        )
    models = {m.id: m for m in repo.list_model_versions(db)}
    return [_to_out(p, models) for p in preds]


@router.get("/{sid}/series", response_model=list[PredictionOut])
def get_prediction_series(
    sid: str,
    lead_hours: int | None = Query(None),
    task: str | None = Query(None),
    model: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[PredictionOut]:
    """Every forecast issued across the storm's life -- powers the
    error-growth chart (docs/API_ARCHITECTURE.md §3.5)."""
    if repo.get_storm(db, sid) is None:
        raise HTTPException(status_code=404, detail=f"Storm '{sid}' not found")
    preds = repo.list_predictions(db, sid, task=task, model_name=model)
    if lead_hours is not None:
        preds = [p for p in preds if p.lead_hours == lead_hours]
    models = {m.id: m for m in repo.list_model_versions(db)}
    return [_to_out(p, models) for p in preds]


@router.get("/models/list", response_model=list[ModelVersionOut])
def list_models(task: str | None = None, db: Session = Depends(get_db)) -> list[ModelVersionOut]:
    return [ModelVersionOut.model_validate(m) for m in repo.list_model_versions(db, task=task)]
