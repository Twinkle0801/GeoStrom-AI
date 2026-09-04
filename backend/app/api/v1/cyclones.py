"""GET /api/v1/cyclones* -- storm catalogue and observations.

Route names follow docs/API_ARCHITECTURE.md §3.1 exactly (resource name
`cyclones`, not `storms`) -- the Phase 3 task brief explicitly defers to
this document's naming where the two differ. The underlying DB table and
Python identifiers are named `storms`, matching docs/SYSTEM_ARCHITECTURE.md
§7.2's schema; only the URL path uses the documented `cyclones` resource
name.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import get_db
from app.repositories import storms as repo
from app.schemas.common import Page, ProblemDetail
from app.schemas.storm import ObservationOut, StormDetail, StormSummary

router = APIRouter(prefix="/api/v1/cyclones", tags=["cyclones"])


@router.get("", response_model=Page[StormSummary])
def list_cyclones(
    season: int | None = None,
    basin: str | None = None,
    split: str | None = Query(None, description="train | val | test"),
    q: str | None = Query(None, description="substring match on storm id"),
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> Page[StormSummary]:
    settings = get_settings()
    limit = min(limit or settings.default_page_size, settings.max_page_size)
    items, total = repo.list_storms(
        db, season=season, basin=basin, split=split, q=q, limit=limit, offset=offset)
    return Page(items=[StormSummary.model_validate(s) for s in items],
               total=total, limit=limit, offset=offset)


@router.get(
    "/{sid}", response_model=StormDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_cyclone(sid: str, db: Session = Depends(get_db)) -> StormDetail:
    storm = repo.get_storm(db, sid)
    if storm is None:
        raise HTTPException(status_code=404, detail=f"Storm '{sid}' not found")
    obs = repo.list_observations(db, sid)
    bbox = None
    if obs:
        lats, lons = [o.lat for o in obs], [o.lon for o in obs]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
    # StormDetail adds `has_predictions`/`bbox`, neither of which the ORM
    # model exposes in a directly-validatable shape (bbox is a PostGIS
    # WKBElement, not a [minlon,minlat,maxlon,maxlat] list) -- build the
    # base summary from the ORM row, then attach the two derived fields.
    summary = StormSummary.model_validate(storm)
    return StormDetail(
        **summary.model_dump(),
        has_predictions=repo.storm_has_predictions(db, sid),
        bbox=bbox,
    )


@router.get(
    "/{sid}/observations", response_model=list[ObservationOut],
    responses={404: {"model": ProblemDetail}},
)
def get_cyclone_observations(
    sid: str,
    from_: dt.datetime | None = Query(None, alias="from"),
    to: dt.datetime | None = None,
    synoptic_only: bool = True,
    db: Session = Depends(get_db),
) -> list[ObservationOut]:
    if repo.get_storm(db, sid) is None:
        raise HTTPException(status_code=404, detail=f"Storm '{sid}' not found")
    obs = repo.list_observations(db, sid)
    if from_ is not None:
        obs = [o for o in obs if o.ts >= from_]
    if to is not None:
        obs = [o for o in obs if o.ts <= to]
    if synoptic_only:
        obs = [o for o in obs if o.is_synoptic]
    return [ObservationOut.model_validate(o) for o in obs]
