"""GET /api/v1/analytics/model-performance -- per docs/API_ARCHITECTURE.md
§3.6, the benchmark table every model/metric on the frozen test split. The
smallest additive endpoint needed for Phase 10's Model Performance page and
methodology page: reads already-committed `ml/reports/*.json` files
(app/services/analytics.py), never recomputes or invents a metric.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.analytics import ModelPerformanceResponse
from app.services.analytics import get_model_performance

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/model-performance", response_model=ModelPerformanceResponse)
def model_performance() -> ModelPerformanceResponse:
    return get_model_performance()
