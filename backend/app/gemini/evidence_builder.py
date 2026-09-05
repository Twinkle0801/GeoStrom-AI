"""Builds an `EvidencePacket` from rows already in the database.

Consumes ONLY the existing Phase 3 repository layer
(`app/repositories/storms.py`) -- no new query pattern, no raw SQL, no
import of `ml.geostrom_ml`. Every value in the resulting packet is either
copied verbatim from a stored row or a small, deterministic derivation
(a percentage between two already-stored metric values); nothing is
invented, matching the evidence-packet principle in task §7.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.db.models import ModelVersion, Storm
from app.repositories import storms as repo
from app.gemini.schemas import (
    ClassificationEvidence, CurrentStateEvidence, EvidencePacket, HistoryPoint,
    IntensityEvidence, IntensityForecastPoint, ModelContext, StormEvidence,
    TrackEvidence, TrackForecastPoint,
)

# The recommended production model for each task (docs/DEVELOPMENT_ROADMAP.md,
# docs/PHASE_7_INTENSITY_PREDICTION.md, docs/PHASE_8_TRACK_PREDICTION.md's own
# recommendations) -- these are the DEFAULTS the evidence builder uses when the
# caller does not request a specific model version. Never silently changed.
DEFAULT_INTENSITY_MODEL = ("intensity_lightgbm", "v1")
DEFAULT_TRACK_MODEL = ("track_cliper", "v1")

_DISPLAY_NAMES = {
    "intensity_persistence": "Persistence",
    "intensity_ridge": "Ridge",
    "intensity_lightgbm": "LightGBM",
    "intensity_gru": "Intensity GRU",
    "intensity_gru_delta": "Intensity GRU (delta)",
    "track_persistence": "Persistence",
    "track_cliper": "CLIPER-style Ridge",
    "track_lightgbm": "LightGBM",
    "track_gru": "Track GRU",
}

BASE_KNOWN_LIMITATIONS = [
    "This is a retrospective research prototype; these are historical baseline model outputs, "
    "not an operational forecast.",
    "No sea-surface temperature, ocean heat content, or vertical wind shear predictors are "
    "available; intensity skill is correspondingly limited.",
    "Models are trained and evaluated on North Atlantic storms only (1980-2015); behaviour in "
    "other basins or periods is unvalidated.",
]

FORBIDDEN_CLAIMS = [
    "landfall timing or location", "casualty or damage estimates",
    "evacuation or safety advice", "comparison to any storm not in this packet",
    "any numeric value not present in this packet",
]


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _find_model_version(model_versions: list[ModelVersion], name: str, version: str) -> ModelVersion | None:
    for m in model_versions:
        if m.name == name and m.version == version:
            return m
    return None


def _headline_skill_vs_persistence(
    model_versions: list[ModelVersion], task: str, this_model: ModelVersion,
    metric_key: str, horizon: str = "24",
) -> float | None:
    """Percentage improvement of `this_model` over the stored persistence
    model's SAME already-computed metric, at the headline 24h horizon --
    a single subtraction/division of two numbers already in the database,
    never a new evaluation run."""
    persistence_name = f"{task}_persistence"
    persistence = next((m for m in model_versions
                        if m.name == persistence_name and m.task == task), None)
    if persistence is None or this_model.name == persistence_name:
        return None
    this_metrics = this_model.metrics.get(horizon)
    base_metrics = persistence.metrics.get(horizon)
    if not this_metrics or not base_metrics:
        return None
    this_val, base_val = this_metrics.get(metric_key), base_metrics.get(metric_key)
    if this_val is None or base_val is None or base_val == 0:
        return None
    return round(100.0 * (base_val - this_val) / base_val, 2)


def build_evidence_packet(
    db: Session, sid: str, *,
    intensity_model: tuple[str, str] = DEFAULT_INTENSITY_MODEL,
    track_model: tuple[str, str] = DEFAULT_TRACK_MODEL,
) -> EvidencePacket:
    storm: Storm = repo.get_storm(db, sid)
    if storm is None:
        raise ValueError(f"Storm '{sid}' not found")

    origin_ts = repo.latest_origin_ts(db, sid)
    observations = repo.list_observations(db, sid)
    model_versions = repo.list_model_versions(db)

    current_state = None
    history_points: list[HistoryPoint] = []
    if observations:
        prior = [o for o in observations if origin_ts is None or o.ts <= origin_ts]
        anchor = prior[-1] if prior else observations[-1]
        current_state = CurrentStateEvidence(
            timestamp=anchor.ts, lat=anchor.lat, lon=anchor.lon, wind_kt=anchor.wind_kt,
            pressure_hpa=anchor.pressure_hpa, category=anchor.category,
            storm_speed_kt=anchor.storm_speed_kt, storm_dir_deg=anchor.storm_dir_deg,
            dist2land_km=anchor.dist2land_km,
        )
        earlier = [o for o in observations if o.ts < anchor.ts]
        history_points = [
            HistoryPoint(timestamp=o.ts, lat=o.lat, lon=o.lon, wind_kt=o.wind_kt)
            for o in earlier[-5:]
        ]

    intensity_evidence = None
    if origin_ts is not None:
        name, version = intensity_model
        mv = _find_model_version(model_versions, name, version)
        preds = repo.list_predictions(db, sid, origin_ts=origin_ts, task="intensity", model_name=name)
        preds = [p for p in preds if _model_id_matches(model_versions, p.model_id, name, version)]
        if mv is not None and preds:
            forecasts = [
                IntensityForecastPoint(
                    lead_hours=p.lead_hours, pred_wind_kt=p.pred_wind_kt,
                    true_wind_kt=p.true_wind_kt, wind_error_kt=p.wind_error_kt,
                )
                for p in sorted(preds, key=lambda p: p.lead_hours)
            ]
            intensity_evidence = IntensityEvidence(
                origin_ts=origin_ts, forecasts=forecasts,
                context=ModelContext(
                    model_name=mv.name, display_name=_display_name(mv.name), model_version=mv.version,
                    dataset_version=mv.dataset_build, metrics_by_horizon=mv.metrics,
                    skill_vs_persistence_pct=_headline_skill_vs_persistence(
                        model_versions, "intensity", mv, "mae_kt"),
                ),
            )

    track_evidence = None
    if origin_ts is not None:
        name, version = track_model
        mv = _find_model_version(model_versions, name, version)
        preds = repo.list_predictions(db, sid, origin_ts=origin_ts, task="track", model_name=name)
        preds = [p for p in preds if _model_id_matches(model_versions, p.model_id, name, version)]
        if mv is not None and preds:
            forecasts = [
                TrackForecastPoint(
                    lead_hours=p.lead_hours, pred_lat=p.pred_lat, pred_lon=p.pred_lon,
                    error_radius_km=p.error_radius_km, true_lat=p.true_lat, true_lon=p.true_lon,
                    track_error_km=p.track_error_km,
                )
                for p in sorted(preds, key=lambda p: p.lead_hours)
            ]
            track_evidence = TrackEvidence(
                origin_ts=origin_ts, forecasts=forecasts,
                context=ModelContext(
                    model_name=mv.name, display_name=_display_name(mv.name), model_version=mv.version,
                    dataset_version=mv.dataset_build, metrics_by_horizon=mv.metrics,
                    skill_vs_persistence_pct=_headline_skill_vs_persistence(
                        model_versions, "track", mv, "mean_track_error_km"),
                ),
            )

    known_limitations = list(BASE_KNOWN_LIMITATIONS)
    # Classification is not yet wired to a per-storm production data source
    # in Phase 9 -- see ClassificationEvidence's docstring. Always None here,
    # documented honestly rather than fabricated.
    classification: ClassificationEvidence | None = None
    if classification is None:
        known_limitations.append(
            "No classification result is available for this storm in the current evidence packet."
        )
    if intensity_evidence is None:
        known_limitations.append(
            "No stored intensity forecast is available for this storm/model combination."
        )
    if track_evidence is None:
        known_limitations.append(
            "No stored track forecast is available for this storm/model combination."
        )

    return EvidencePacket(
        generated_at=dt.datetime.now(dt.timezone.utc),
        storm=StormEvidence.model_validate(storm),
        current_state=current_state,
        recent_history=history_points,
        intensity=intensity_evidence,
        track=track_evidence,
        classification=classification,
        known_limitations=known_limitations,
        forbidden_claims=list(FORBIDDEN_CLAIMS),
    )


def _model_id_matches(model_versions: list[ModelVersion], model_id: int, name: str, version: str) -> bool:
    for m in model_versions:
        if m.id == model_id:
            return m.name == name and m.version == version
    return False
