"""GeoJSON assembly: the ONLY place a Point/LineString is constructed for a
response. Per docs/SYSTEM_ARCHITECTURE.md §6.1, geometry assembly is a
backend concern; the frontend receives GeoJSON that is already correct and
never reprojects or does geodesic maths.

Observed and predicted geometry are built into DISTINCT feature `kind`
values (`observed_track`/`observed_point` vs. `predicted_track`/
`predicted_point`) -- this is the mandatory Phase 3 distinction between
observed and predicted tracks, enforced at the data-shape level so a
frontend bug cannot silently render one as the other.
"""

from __future__ import annotations

from app.db.models import Observation, ModelVersion, Prediction, Storm
from app.schemas.geojson import Feature, FeatureCollection, LineStringGeometry, PointGeometry


def observed_features(observations: list[Observation]) -> list[Feature]:
    features: list[Feature] = []
    coords = [(o.lon, o.lat) for o in observations]
    if len(coords) >= 2:
        features.append(Feature(
            geometry=LineStringGeometry(coordinates=coords),
            properties={"kind": "observed_track", "data_kind": "observed", "n_points": len(coords)},
        ))
    for o in observations:
        features.append(Feature(
            geometry=PointGeometry(coordinates=(o.lon, o.lat)),
            properties={
                "kind": "observed_point", "data_kind": "observed",
                "ts": o.ts.isoformat(), "wind_kt": o.wind_kt, "pressure_hpa": o.pressure_hpa,
                "category": o.category, "is_observed": o.is_observed,
            },
        ))
    return features


def predicted_features_for_origin(
    predictions: list[Prediction], model_by_id: dict[int, ModelVersion],
) -> list[Feature]:
    """One predicted_track (+ predicted_point per horizon) per (task, model)
    present in `predictions`, which the caller has already filtered to a
    single origin_ts. Track-task predictions only -- intensity predictions
    carry no position and are surfaced via /prediction, not /tracks."""
    features: list[Feature] = []
    by_model: dict[int, list[Prediction]] = {}
    for p in predictions:
        if p.task != "track" or p.pred_lat is None or p.pred_lon is None:
            continue
        by_model.setdefault(p.model_id, []).append(p)

    for model_id, preds in by_model.items():
        preds = sorted(preds, key=lambda p: p.lead_hours)
        model = model_by_id[model_id]
        origin = preds[0]
        # the predicted line starts at the forecast origin (an observed
        # point, included here only as the line's anchor, not re-labelled
        # as a prediction) and continues through each predicted point.
        coords = [(origin.true_lon, origin.true_lat)] if origin.true_lon is not None else []
        # fall back to nothing if true origin missing; still show predicted points
        for p in preds:
            coords.append((p.pred_lon, p.pred_lat))
        if len(coords) >= 2:
            features.append(Feature(
                geometry=LineStringGeometry(coordinates=coords),
                properties={
                    "kind": "predicted_track", "data_kind": "model_prediction",
                    "model_name": model.name, "model_version": model.version,
                    "origin_ts": origin.origin_ts.isoformat(),
                },
            ))
        for p in preds:
            features.append(Feature(
                geometry=PointGeometry(coordinates=(p.pred_lon, p.pred_lat)),
                properties={
                    "kind": "predicted_point", "data_kind": "model_prediction",
                    "model_name": model.name, "model_version": model.version,
                    "origin_ts": p.origin_ts.isoformat(), "valid_ts": p.valid_ts.isoformat(),
                    "lead_hours": p.lead_hours,
                    "pred_wind_kt": p.pred_wind_kt,
                    "error_radius_km": p.error_radius_km,
                    "track_error_km": p.track_error_km,
                    "true_lat": p.true_lat, "true_lon": p.true_lon,
                    "disclaimer": "Historical baseline model prediction, not an operational forecast.",
                },
            ))
    return features


def build_track_feature_collection(
    storm: Storm, observations: list[Observation],
    predictions: list[Prediction], model_by_id: dict[int, ModelVersion],
) -> FeatureCollection:
    features = observed_features(observations)
    features.extend(predicted_features_for_origin(predictions, model_by_id))
    return FeatureCollection(features=features)
