"""GeoStrom AI backend -- Phase 3 vertical slice.

Read-only, offline-inference-free per docs/SYSTEM_ARCHITECTURE.md §1: this
process never imports torch/lightgbm/ml.geostrom_ml and never computes a
prediction. It reads rows a separate offline step (ml/ + backend/scripts/
ingest_phase2_predictions.py) already wrote to PostgreSQL.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import cyclones, explain, prediction, system, tracks
from app.core.config import get_settings
from app.schemas.common import ProblemDetail

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description=(
        "Retrospective tropical cyclone research prototype. Serves precomputed "
        "Phase 2 baseline model predictions. Not an operational forecasting system."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],  # POST added in Phase 9, for /api/v1/explain/* only
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """RFC 7807 problem+json, per docs/API_ARCHITECTURE.md §2."""
    problem = ProblemDetail(
        title=exc.detail if isinstance(exc.detail, str) else "Error",
        status=exc.status_code, detail=str(exc.detail), instance=str(request.url.path),
    )
    return JSONResponse(status_code=exc.status_code, content=problem.model_dump(),
                        media_type="application/problem+json")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    problem = ProblemDetail(
        title="Validation error", status=422, detail=str(exc.errors()),
        instance=str(request.url.path),
    )
    return JSONResponse(status_code=422, content=problem.model_dump(),
                        media_type="application/problem+json")


app.include_router(system.router)
app.include_router(cyclones.router)
app.include_router(tracks.router)
app.include_router(prediction.router)
app.include_router(explain.router)
