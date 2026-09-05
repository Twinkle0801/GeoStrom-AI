"""Deterministic fallback template tests -- task §9.

The fallback must be built entirely from evidence values (never hardcoded
fake numbers) and, since it is the safety net every failure mode degrades
to, it must itself always pass the grounding validator -- this is asserted
directly, not assumed.
"""

from __future__ import annotations

from app.gemini.fallback import build_fallback_explanation
from app.gemini.validator import validate_grounding
from tests.gemini_fixtures import make_evidence_packet


def test_fallback_uses_real_evidence_values_dynamically():
    evidence = make_evidence_packet()
    explanation = build_fallback_explanation(evidence)
    assert "92.4" in explanation.intensity_explanation
    assert "LightGBM" in explanation.intensity_explanation
    assert "25.9" in explanation.track_explanation
    assert "CLIPER-style Ridge" in explanation.track_explanation


def test_fallback_always_passes_its_own_grounding_validator():
    evidence = make_evidence_packet()
    explanation = build_fallback_explanation(evidence)
    assert validate_grounding(explanation, evidence) == []


def test_fallback_handles_missing_intensity_and_track_gracefully():
    evidence = make_evidence_packet()
    evidence = evidence.model_copy(update={"intensity": None, "track": None})
    explanation = build_fallback_explanation(evidence)
    assert "No stored intensity forecast" in explanation.intensity_explanation
    assert "No stored track forecast" in explanation.track_explanation
    assert validate_grounding(explanation, evidence) == []


def test_fallback_never_hardcodes_a_fake_number():
    """Changing the evidence's wind value must change the fallback's text --
    proving the number is read dynamically, not baked in."""
    evidence_a = make_evidence_packet()
    evidence_b = evidence_a.model_copy(deep=True)
    evidence_b.intensity.forecasts[1].pred_wind_kt = 55.5
    explanation_a = build_fallback_explanation(evidence_a)
    explanation_b = build_fallback_explanation(evidence_b)
    assert "92.4" in explanation_a.intensity_explanation
    assert "55.5" in explanation_b.intensity_explanation
    assert explanation_a.intensity_explanation != explanation_b.intensity_explanation


def test_fallback_classification_absent_is_stated_not_fabricated():
    evidence = make_evidence_packet(with_classification=False)
    explanation = build_fallback_explanation(evidence)
    assert "No stored classification" in explanation.classification_explanation


def test_fallback_includes_disclaimer_language():
    evidence = make_evidence_packet()
    explanation = build_fallback_explanation(evidence)
    assert "not" in explanation.limitations.lower()
    assert "operational" in explanation.limitations.lower()
