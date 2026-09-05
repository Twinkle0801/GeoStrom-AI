"""A hand-constructed `EvidencePacket` factory shared by the Phase 9 test
files that don't need the real database (validator/fallback/service tests).
Not a test file itself (no `test_` prefix) -- imported by ones that are."""

from __future__ import annotations

import datetime as dt

from app.gemini.schemas import (
    ClassificationEvidence, CurrentStateEvidence, EvidencePacket, IntensityEvidence,
    IntensityForecastPoint, ModelContext, StormEvidence, TrackEvidence, TrackForecastPoint,
)

BASE_LIMITATIONS = ["This is a retrospective research prototype; not an operational forecast."]
BASE_FORBIDDEN = [
    "landfall timing or location", "casualty or damage estimates",
    "evacuation or safety advice", "comparison to any storm not in this packet",
    "any numeric value not present in this packet",
]


def make_evidence_packet(
    *, sid: str = "2010176N16278", storm_name: str | None = "ALEX",
    with_classification: bool = False, classification_label: str = "Eye",
    classification_confidence: float | None = None,
) -> EvidencePacket:
    origin_ts = dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc)
    classification = None
    if with_classification:
        classification = ClassificationEvidence(
            class_label=classification_label, confidence=classification_confidence,
            model_name="cls_logreg", model_version="v1",
        )
    return EvidencePacket(
        generated_at=dt.datetime.now(dt.timezone.utc),
        storm=StormEvidence(
            sid=sid, name=storm_name, season=2010, basin="NA",
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
            forecasts=[
                IntensityForecastPoint(lead_hours=6, pred_wind_kt=97.0, true_wind_kt=None, wind_error_kt=None),
                IntensityForecastPoint(lead_hours=24, pred_wind_kt=92.4, true_wind_kt=90.0, wind_error_kt=2.4),
            ],
            context=ModelContext(
                model_name="intensity_lightgbm", display_name="LightGBM", model_version="v1",
                dataset_version="v1", metrics_by_horizon={"24": {"mae_kt": 8.5}},
                skill_vs_persistence_pct=19.8,
            ),
        ),
        track=TrackEvidence(
            origin_ts=origin_ts,
            forecasts=[
                TrackForecastPoint(lead_hours=24, pred_lat=25.9, pred_lon=-88.9,
                                   error_radius_km=200.4, true_lat=None, true_lon=None,
                                   track_error_km=None),
            ],
            context=ModelContext(
                model_name="track_cliper", display_name="CLIPER-style Ridge", model_version="v1",
                dataset_version="v1", metrics_by_horizon={"24": {"mean_track_error_km": 200.4}},
                skill_vs_persistence_pct=11.4,
            ),
        ),
        classification=classification,
        known_limitations=list(BASE_LIMITATIONS),
        forbidden_claims=list(BASE_FORBIDDEN),
    )
