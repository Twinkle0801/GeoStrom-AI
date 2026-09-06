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

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.base import get_db
from app.gemini.cache import ExplainCache, explain_cache_key
from app.gemini.client import GeminiClientProtocol, build_gemini_client
from app.gemini.evidence_builder import (
    DEFAULT_INTENSITY_MODEL, DEFAULT_TRACK_MODEL, build_evidence_packet,
)
from app.gemini.ratelimit import RateLimiter
from app.gemini.schemas import EVIDENCE_SCHEMA_VERSION, ExplainRequest, ExplainResponse, ModelRef
from app.gemini.service import ExplainResult, GeminiExplanationService
from app.repositories import storms as repo
from app.schemas.common import ProblemDetail

router = APIRouter(prefix="/api/v1/explain", tags=["explain"])

_settings_for_singletons = get_settings()
# Process-local singletons (Phase 12) -- see cache.py/ratelimit.py module
# docstrings for what "process-local" means here. Exposed as FastAPI
# dependencies, not plain module globals, specifically so tests can swap in
# an isolated fresh instance via `app.dependency_overrides`, the same
# pattern `get_gemini_client` already established in Phase 9.
_explain_cache = ExplainCache(
    maxsize=_settings_for_singletons.gemini_cache_maxsize,
    ttl_seconds=_settings_for_singletons.gemini_cache_ttl_seconds,
)
_rate_limiter = RateLimiter(
    max_requests=_settings_for_singletons.gemini_rate_limit_max_requests,
    window_seconds=_settings_for_singletons.gemini_rate_limit_window_seconds,
)


def get_gemini_client(
    settings: Settings = Depends(get_settings),
) -> GeminiClientProtocol | None:
    """A FastAPI dependency (not a plain function call) specifically so
    tests can override it with a mocked client via
    `app.dependency_overrides`, the same pattern `get_db` already uses --
    the majority of Phase 9 tests must run without a real Gemini API call
    (task §22)."""
    return build_gemini_client(settings)


def get_explain_cache() -> ExplainCache:
    return _explain_cache


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


@router.post(
    "/forecast", response_model=ExplainResponse,
    responses={404: {"model": ProblemDetail}, 429: {"model": ProblemDetail}},
)
def explain_forecast(
    body: ExplainRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: GeminiClientProtocol | None = Depends(get_gemini_client),
    cache: ExplainCache = Depends(get_explain_cache),
    limiter: RateLimiter = Depends(get_rate_limiter),
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

    # Cache lookup first: a hit makes no Gemini call at all, so it is never
    # rate-limited and costs no quota (Phase 12 §9's "do not break normal
    # frontend usage" -- repeated views of the same already-explained
    # forecast must stay free). Only a genuine attempted Gemini call (cache
    # miss) is subject to the rate limiter below.
    cache_key = explain_cache_key(evidence)
    cached_result: ExplainResult | None = cache.get(cache_key)  # type: ignore[assignment]
    if cached_result is not None:
        result = cached_result
    else:
        client_ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.allow(client_ip)
        if not allowed:
            retry_after_s = int(retry_after) + 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for /api/v1/explain/forecast; please retry shortly.",
                headers={"Retry-After": str(retry_after_s)},
            )
        result = GeminiExplanationService(client, settings).explain(evidence)
        # Only a validated, grounded Gemini success is cached -- never a
        # fallback (timeout/api_error/malformed_json/ungrounded_claim/
        # not_configured), per Phase 12 §9's "do not cache validation
        # failures" / "do not cache transport failures".
        if result.source == "gemini":
            cache.set(cache_key, result)

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
