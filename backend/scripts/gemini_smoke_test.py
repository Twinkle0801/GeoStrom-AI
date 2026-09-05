"""Phase 9: OPTIONAL manual smoke test against the REAL Gemini API.

Per task §25: NOT run automatically as part of the test suite (pytest never
collects this file -- it lives in `scripts/`, not `tests/`, and defines no
`test_*` function). Run it by hand only if a local `backend/.env` already
has a real `GEMINI_API_KEY`:

    cd backend && python scripts/gemini_smoke_test.py

If no key is configured, this script prints a message and exits 0 --
absence of a key is not a Phase 9 failure; the mocked test suite
(`pytest backend/tests/test_gemini_*.py`) is the required validation.

This script NEVER prints the key, never commits anything, and uses a tiny,
synthetic, hand-built evidence packet -- no database connection needed.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.gemini.client import build_gemini_client  # noqa: E402
from app.gemini.schemas import (  # noqa: E402
    CurrentStateEvidence, EvidencePacket, IntensityEvidence, IntensityForecastPoint,
    ModelContext, StormEvidence, TrackEvidence, TrackForecastPoint,
)
from app.gemini.service import GeminiExplanationService  # noqa: E402


def _synthetic_evidence() -> EvidencePacket:
    origin_ts = dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc)
    return EvidencePacket(
        generated_at=dt.datetime.now(dt.timezone.utc),
        storm=StormEvidence(
            sid="SMOKE_TEST_STORM", name="SMOKETEST", season=2010, basin="NA",
            start_time=dt.datetime(2010, 6, 25, tzinfo=dt.timezone.utc),
            end_time=dt.datetime(2010, 6, 30, tzinfo=dt.timezone.utc), n_observations=17,
        ),
        current_state=CurrentStateEvidence(
            timestamp=origin_ts, lat=25.4, lon=-87.6, wind_kt=95.0, pressure_hpa=948.0,
            category=2, storm_speed_kt=11.0, storm_dir_deg=315.0, dist2land_km=340.0,
        ),
        recent_history=[],
        intensity=IntensityEvidence(
            origin_ts=origin_ts,
            forecasts=[IntensityForecastPoint(lead_hours=24, pred_wind_kt=92.4,
                                              true_wind_kt=90.0, wind_error_kt=2.4)],
            context=ModelContext(
                model_name="intensity_lightgbm", display_name="LightGBM", model_version="v1",
                dataset_version="v1", metrics_by_horizon={"24": {"mae_kt": 8.5}},
                skill_vs_persistence_pct=19.8,
            ),
        ),
        track=TrackEvidence(
            origin_ts=origin_ts,
            forecasts=[TrackForecastPoint(lead_hours=24, pred_lat=25.9, pred_lon=-88.9,
                                          error_radius_km=200.4, true_lat=None, true_lon=None,
                                          track_error_km=None)],
            context=ModelContext(
                model_name="track_cliper", display_name="CLIPER-style Ridge", model_version="v1",
                dataset_version="v1", metrics_by_horizon={"24": {"mean_track_error_km": 200.4}},
                skill_vs_persistence_pct=11.4,
            ),
        ),
        classification=None,
        known_limitations=["This is a synthetic smoke-test packet, not a real storm."],
        forbidden_claims=["landfall timing or location", "evacuation or safety advice"],
    )


def main() -> int:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("No GEMINI_API_KEY configured in backend/.env -- skipping the real smoke test. "
              "This is expected and NOT a Phase 9 failure; see docs/PHASE_9_GEMINI_INTEGRATION.md.")
        return 0

    client = build_gemini_client(settings)
    evidence = _synthetic_evidence()
    result = GeminiExplanationService(client, settings).explain(evidence)

    print(f"source            = {result.source}")
    print(f"fallback_reason   = {result.fallback_reason}")
    print(f"violation_count   = {len(result.violations)}")
    print(f"summary           = {result.explanation.summary}")
    print(f"intensity_explanation = {result.explanation.intensity_explanation}")
    print(f"track_explanation     = {result.explanation.track_explanation}")
    print("\n(No API key or raw request/response content is ever printed by this script.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
