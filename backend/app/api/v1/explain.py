"""POST /api/v1/explain/forecast -- Gemini-backed natural-language
explanation of a stored forecast, per `docs/API_ARCHITECTURE.md` §3.7 and
Phase 9's minimal-contract instruction (only this one endpoint is built;
`/explain/storm`, `/explain/compare`, `/explain/ask` remain the documented,
un-built "nice to have"/"guarded" rows in that table).

Read-only with respect to ML: this route calls only the existing
repository layer and the Gemini explanation service. It never computes a
prediction and never imports `ml.geostrom_ml`, per `app/main.py`'s
module-level invariant.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.base import get_db
from app.gemini.client import GeminiClientProtocol, build_gemini_client
from app.gemini.evidence_builder import (
    DEFAULT_INTENSITY_MODEL, DEFAULT_TRACK_MODEL, build_evidence_packet,
)
from app.gemini.schemas import EVIDENCE_SCHEMA_VERSION, ExplainRequest, ExplainResponse, ModelRef
from app.gemini.service import GeminiExplanationService
from app.repositories import storms as repo
from app.schemas.common import ProblemDetail

router = APIRouter(prefix="/api/v1/explain", tags=["explain"])


def get_gemini_client(
    settings: Settings = Depends(get_settings),
) -> GeminiClientProtocol | None:
    """A FastAPI dependency (not a plain function call) specifically so
    tests can override it with a mocked client via
    `app.dependency_overrides`, the same pattern `get_db` already uses --
    the majority of Phase 9 tests must run without a real Gemini API call
    (task §22)."""
    return build_gemini_client(settings)


@router.post(
    "/forecast", response_model=ExplainResponse,
    responses={404: {"model": ProblemDetail}},
)
def explain_forecast(
    body: ExplainRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: GeminiClientProtocol | None = Depends(get_gemini_client),
) -> ExplainResponse:
    if repo.get_storm(db, body.sid) is None:
        raise HTTPException(status_code=404, detail=f"Storm '{body.sid}' not found")

    intensity_model = (
        DEFAULT_INTENSITY_MODEL[0],
        body.intensity_model_version or DEFAULT_INTENSITY_MODEL[1],
    )
    track_model = (
        DEFAULT_TRACK_MODEL[0],
        body.track_model_version or DEFAULT_TRACK_MODEL[1],
    )

    evidence = build_evidence_packet(
        db, body.sid, intensity_model=intensity_model, track_model=track_model,
    )

    result = GeminiExplanationService(client, settings).explain(evidence)

    return ExplainResponse(
        sid=body.sid,
        generated_at=evidence.generated_at,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        intensity_model=(
            ModelRef(name=evidence.intensity.context.model_name,
                     version=evidence.intensity.context.model_version)
            if evidence.intensity is not None else None
        ),
        track_model=(
            ModelRef(name=evidence.track.context.model_name,
                     version=evidence.track.context.model_version)
            if evidence.track is not None else None
        ),
        classification_model=(
            ModelRef(name=evidence.classification.model_name,
                     version=evidence.classification.model_version)
            if evidence.classification is not None else None
        ),
        source=result.source,
        fallback_reason=result.fallback_reason,
        validation_violations=result.violations,
        explanation=result.explanation,
        evidence=evidence,
    )
