"""GET /api/v1/tracks/{sid} -- the map's primary payload.

Per docs/API_ARCHITECTURE.md §3.2: one composite GeoJSON FeatureCollection
containing the observed track and the predicted track(s) for one forecast
origin, rather than five small round trips.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.repositories import storms as repo
from app.schemas.common import ProblemDetail
from app.schemas.geojson import FeatureCollection
from app.services.geometry import build_track_feature_collection

router = APIRouter(prefix="/api/v1/tracks", tags=["tracks"])


@router.get(
    "/{sid}", response_model=FeatureCollection,
    responses={404: {"model": ProblemDetail}},
)
def get_track(
    sid: str,
    t: dt.datetime | None = Query(
        None, description="Forecast origin timestamp (ISO 8601). "
                          "Defaults to the storm's latest available origin."),
    db: Session = Depends(get_db),
) -> FeatureCollection:
    storm = repo.get_storm(db, sid)
    if storm is None:
        raise HTTPException(status_code=404, detail=f"Storm '{sid}' not found")

    observations = repo.list_observations(db, sid)

    origin_ts = t or repo.latest_origin_ts(db, sid)
    predictions = []
    model_by_id: dict[int, object] = {}
    if origin_ts is not None:
        predictions = repo.list_predictions(db, sid, origin_ts=origin_ts, task="track")
        models = repo.list_model_versions(db, task="track")
        model_by_id = {m.id: m for m in models}

    return build_track_feature_collection(storm, observations, predictions, model_by_id)
