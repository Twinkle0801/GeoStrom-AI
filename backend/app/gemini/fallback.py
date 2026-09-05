"""Deterministic, template-based fallback explanation.

Built entirely from the evidence packet's already-verified values -- never
from Gemini, never a hardcoded fake number (task §9: "Use actual evidence
values dynamically. Do not hard-code fake values."). This is what the
system returns whenever Gemini is unavailable, misconfigured, malformed, or
fails grounding validation, so the application remains scientifically
trustworthy under every one of those conditions (task's stated core
requirement).
"""

from __future__ import annotations

from app.gemini.schemas import EvidencePacket, GeminiStructuredResponse

DISCLAIMER_SENTENCE = (
    "These are retrospective model outputs from a research prototype and should not be "
    "interpreted as an operational forecast, weather warning, or safety advisory."
)


def _headline_horizon_forecast(forecasts: list, preferred_h: int = 24):
    if not forecasts:
        return None
    for fc in forecasts:
        if fc.lead_hours == preferred_h:
            return fc
    return max(forecasts, key=lambda f: f.lead_hours)


def build_fallback_explanation(evidence: EvidencePacket) -> GeminiStructuredResponse:
    storm = evidence.storm
    storm_label = f"{storm.name} ({storm.sid})" if storm.name else storm.sid

    # ---- intensity ----
    if evidence.intensity is not None and evidence.intensity.forecasts:
        fc = _headline_horizon_forecast(evidence.intensity.forecasts)
        ctx = evidence.intensity.context
        intensity_explanation = (
            f"The selected intensity model ({ctx.display_name}, {ctx.model_version}) predicts "
            f"approximately {fc.pred_wind_kt:.1f} kt at +{fc.lead_hours}h from "
            f"{evidence.intensity.origin_ts.isoformat()}."
        )
        if fc.true_wind_kt is not None:
            intensity_explanation += (
                f" The observed wind at that valid time was {fc.true_wind_kt:.1f} kt "
                f"(error {fc.wind_error_kt:+.1f} kt)."
            )
        if ctx.skill_vs_persistence_pct is not None:
            intensity_explanation += (
                f" On the frozen historical test split, this model's skill vs. a persistence "
                f"baseline is {ctx.skill_vs_persistence_pct:.1f}%."
            )
    else:
        intensity_explanation = "No stored intensity forecast is available for this storm/model."

    # ---- track ----
    if evidence.track is not None and evidence.track.forecasts:
        fc = _headline_horizon_forecast(evidence.track.forecasts)
        ctx = evidence.track.context
        track_explanation = (
            f"The selected track model ({ctx.display_name}, {ctx.model_version}) predicts a "
            f"position near {fc.pred_lat:.2f}, {fc.pred_lon:.2f} at +{fc.lead_hours}h."
        )
        if fc.error_radius_km is not None:
            track_explanation += (
                f" This model's historical error radius at this horizon is "
                f"{fc.error_radius_km:.0f} km."
            )
        if fc.track_error_km is not None:
            track_explanation += (
                f" The observed track error for this specific forecast was "
                f"{fc.track_error_km:.1f} km."
            )
    else:
        track_explanation = "No stored track forecast is available for this storm/model."

    # ---- classification ----
    if evidence.classification is not None:
        cls = evidence.classification
        classification_explanation = (
            f"The observed cyclone scene was classified as {cls.class_label} "
            f"({cls.model_name} {cls.model_version})."
        )
        if cls.confidence is not None:
            classification_explanation += f" Reported confidence: {cls.confidence * 100:.0f}%."
    else:
        classification_explanation = (
            "No stored classification result is available for this storm."
        )

    summary = (
        f"GeoStrom AI evidence for {storm_label} ({storm.season}, {storm.basin}). "
        f"{intensity_explanation} {track_explanation}"
    )

    if evidence.known_limitations:
        limitations = " ".join(evidence.known_limitations) + " " + DISCLAIMER_SENTENCE
    else:
        limitations = DISCLAIMER_SENTENCE

    return GeminiStructuredResponse(
        summary=summary,
        intensity_explanation=intensity_explanation,
        track_explanation=track_explanation,
        classification_explanation=classification_explanation,
        limitations=limitations,
    )
