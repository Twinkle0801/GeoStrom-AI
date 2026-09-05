"""Deterministic grounding validator -- the load-bearing guardrail per
`docs/API_ARCHITECTURE.md` §8 Layer 4.

The evidence packet is the source of truth (task §7). Every numeric,
categorical, or model-identity claim Gemini makes is checked against it;
anything not grounded is rejected outright -- never "approximately"
accepted. This module is pure and deterministic: same (response, evidence)
pair always produces the same verdict, no network access, no randomness.

Scope, stated honestly (also documented in
docs/PHASE_9_GEMINI_INTEGRATION.md): this is a deterministic, regex- and
tolerance-based claim checker, not a full natural-language-understanding
system. It is deliberately conservative for the categories where a
false-accept would be scientifically dangerous (fabricated confidence,
fabricated horizon, fabricated classification label, fabricated model name)
and uses a numeric-tolerance check (allowing "about 92" for 92.4) for plain
magnitude claims (wind/pressure/track-distance/coordinates/counts), pooled
together rather than perfectly unit-typed per category -- a documented,
deliberate simplification, not an oversight.
"""

from __future__ import annotations

import re

from app.gemini.schemas import EvidencePacket, GeminiStructuredResponse

# Phase 5's frozen scene_taxonomy_v1 classes (docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md).
# A plain, literal vocabulary constant -- NOT an import of ml.geostrom_ml (the backend never
# imports that package, per app/main.py's own module docstring).
KNOWN_CLASSIFICATION_LABELS = {
    "CDO", "IrrCDO", "CurvedBand", "Eye", "LargeEye", "Shear", "EmbCenter", "Land",
}

# Every model-family name this project has ever produced, across all phases -- the set Gemini
# might plausibly (or hallucinatorily) mention. A token found in the response that belongs to
# this set must match one of THIS evidence packet's actual model names, or the claim is rejected.
KNOWN_MODEL_NAME_TOKENS = {
    "persistence", "ridge", "cliper", "lightgbm", "gru", "resnet", "resnet-18", "resnet18",
    "logistic regression", "transformer", "lstm", "cnn", "gemini",
}

FORBIDDEN_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bevacuat\w*", r"\bcasualt\w*", r"\bdamage estimate", r"\bwill (?:make landfall|hit|strike)\b",
        r"\blandfall (?:time|location|is expected)", r"\bmust (?:evacuate|take shelter)",
        r"\boperational (?:forecast|warning|advisory)\b", r"\bissue(?:d)? a warning\b",
        r"\bguarante(?:e|ed)\b", r"\btake (?:immediate )?action\b", r"\bofficial (?:warning|advisory)\b",
    ]
]

PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
CONFIDENCE_WORD_RE = re.compile(
    r"confiden(?:t|ce)[^.\n]{0,40}?(-?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
HORIZON_RE = re.compile(r"\b(\d+)\s*[- ]?(?:h\b|hr\b|hrs\b|hour\b|hours\b)", re.IGNORECASE)
# Consumes an entire ISO date/datetime as ONE token, so its internal
# year/month/day/hour/minute/second digit groups are never individually
# re-scanned as unrelated bare numbers (they are calendar components, not
# magnitude claims).
ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)?")
# Finds the single, maximal number span (including a leading '-' for
# negative coordinates/biases) with NO unit-suffix exclusions baked in --
# those are applied afterward in `_iter_bare_numbers` as a plain forward
# check on the text following the match, not as inline negative lookaheads.
#
# This split is deliberate, after finding that lookahead-based exclusion is
# NOT safe under regex backtracking: e.g. a naive `\b\d+(?:\.\d+)?\b(?!\s*%)`
# applied to "19.8%" fails its lookahead on the full "19.8" (blocked by the
# "%"), and the engine then backtracks to the SHORTER match "19" -- which
# passes the same lookahead (nothing-but-".8%" follows immediately) and gets
# wrongly counted as a separate, ungrounded bare number. Finding the maximal
# span FIRST and then deciding in plain Python whether to skip it removes
# that failure mode entirely.
#
# The lookbehind `(?<![\w.-])` (not `\b`) is what makes this safe against IDs
# like "2010176N16278" (a digit run immediately followed by a letter, both
# `\w`, so the trailing `(?!\w)` check fails at every backtrack length -- no
# partial match), while STILL correctly capturing a leading minus sign for
# negative numbers (a `\b` immediately before "-87.15" preceded by a space
# would incorrectly refuse to match at all, since space and "-" are both
# non-word characters and no boundary exists between them -- silently
# dropping the sign and validating the wrong, positive magnitude).
NUMBER_RE = re.compile(r"(?<![\w.-])(-?\d+(?:\.\d+)?)(?!\w)")
_SUFFIX_PERCENT_RE = re.compile(r"^\s*(?:%|percent)", re.IGNORECASE)
_SUFFIX_HOURS_RE = re.compile(r"^\s*(?:hours|hour|hrs|hr|h)\b", re.IGNORECASE)
_SUFFIX_DEGREE_RE = re.compile(r"^\s*°")


def _iter_bare_numbers(text: str):
    """Yields every number in `text` that is NOT immediately followed (after
    optional whitespace) by a %/percent, hour, or degree-symbol suffix --
    those are the dedicated regexes' job (`PERCENT_RE`, `CONFIDENCE_WORD_RE`,
    `HORIZON_RE`) or are folded into the generic pool anyway (coordinates)."""
    for m in NUMBER_RE.finditer(text):
        tail = text[m.end():m.end() + 16]
        if _SUFFIX_PERCENT_RE.match(tail) or _SUFFIX_HOURS_RE.match(tail) or _SUFFIX_DEGREE_RE.match(tail):
            continue
        yield m.group(1)

NUMBER_TOLERANCE_ABS = 0.6
NUMBER_TOLERANCE_REL = 0.02


_NEGATION_WORDS_RE = re.compile(
    r"\b(?:not|never|no|isn't|is not|does not|doesn't|should not|shouldn't|"
    r"cannot|can't|without|non-?operational|forbidden|prohibited|"
    r"not (?:be )?(?:claimed|provided|inferred|given|available|permitted|allowed))\b",
    re.IGNORECASE)


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    """A forbidden-claim pattern match is exempted when a negation word
    appears anywhere in the SAME sentence -- before OR after the match.

    Real, observed Gemini phrasings (found via the manual smoke test, task
    §25) put the negation on either side: "should NOT be interpreted as an
    operational forecast" (before), but equally commonly "...landfall
    timing or location, and evacuation or safety advice are FORBIDDEN"
    or "...are explicitly forbidden claims and NOT provided" (after,
    often because Gemini is directly paraphrasing this packet's own
    `forbidden_claims` list back as a disclaimer -- expected, safe, and
    exactly what the system prompt asks for). All of these are required,
    cautionary constructions; only a genuine, unnegated positive assertion
    ("we issue an operational warning") has no negation anywhere in its
    sentence and is correctly still rejected.

    Scoped to the whole sentence (not a fixed character window in either
    direction): the negating word is frequently well over 30 characters
    from the forbidden token once several forbidden-sounding nouns are
    listed together, as in the examples above."""
    sentence_start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text, 0, match_start):
        sentence_start = m.end()
    end_match = _SENTENCE_BOUNDARY_RE.search(text, match_end)
    sentence_end = end_match.start() if end_match else len(text)
    return bool(_NEGATION_WORDS_RE.search(text[sentence_start:sentence_end]))


def _within_tolerance(claimed: float, pool: set[float]) -> bool:
    for evidence_value in pool:
        tol = max(NUMBER_TOLERANCE_ABS, NUMBER_TOLERANCE_REL * abs(evidence_value))
        if abs(claimed - evidence_value) <= tol:
            return True
    return False


def _collect_pools(evidence: EvidencePacket) -> tuple[set[float], set[int], set[float], set[str]]:
    """Returns (generic_numeric_pool, horizon_hours_pool, percentage_pool, date_strings_pool)."""
    generic: set[float] = {float(evidence.storm.season), float(evidence.storm.n_observations)}
    horizons: set[int] = set()
    percentages: set[float] = set()
    dates: set[str] = {
        evidence.storm.start_time.date().isoformat(),
        evidence.storm.end_time.date().isoformat(),
    }

    def add_num(v) -> None:
        if v is not None:
            generic.add(float(v))

    if evidence.current_state is not None:
        cs = evidence.current_state
        for v in (cs.lat, cs.lon, cs.wind_kt, cs.pressure_hpa, cs.storm_speed_kt,
                  cs.storm_dir_deg, cs.dist2land_km, cs.category):
            add_num(v)
        dates.add(cs.timestamp.date().isoformat())

    for h in evidence.recent_history:
        add_num(h.lat)
        add_num(h.lon)
        add_num(h.wind_kt)
        dates.add(h.timestamp.date().isoformat())

    for block in (evidence.intensity, evidence.track):
        if block is None:
            continue
        dates.add(block.origin_ts.date().isoformat())
        for fc in block.forecasts:
            horizons.add(fc.lead_hours)
            for field in ("pred_wind_kt", "true_wind_kt", "wind_error_kt",
                          "pred_lat", "pred_lon", "true_lat", "true_lon",
                          "error_radius_km", "track_error_km"):
                add_num(getattr(fc, field, None))
        ctx = block.context
        for horizon_metrics in ctx.metrics_by_horizon.values():
            for v in horizon_metrics.values():
                if isinstance(v, (int, float)):
                    add_num(v)
        if ctx.skill_vs_persistence_pct is not None:
            percentages.add(float(ctx.skill_vs_persistence_pct))

    if evidence.classification is not None:
        if evidence.classification.confidence is not None:
            percentages.add(round(evidence.classification.confidence * 100.0, 6))
            percentages.add(round(evidence.classification.confidence, 6))

    generic |= percentages  # a percentage number is still a valid plain number elsewhere too
    generic |= {float(h) for h in horizons}
    return generic, horizons, percentages, dates


def _known_model_names(evidence: EvidencePacket) -> set[str]:
    """Every string Gemini could legitimately use to refer to a model
    actually present in this evidence packet: the raw DB name (e.g.
    'track_cliper') AND its human-readable display name (e.g. 'CLIPER-style
    Ridge'), so a correct, natural mention of "Ridge" for the CLIPER model
    is not mistaken for the hallucinated-model-name case this check exists
    to catch."""
    names: set[str] = set()
    for block in (evidence.intensity, evidence.track):
        if block is not None:
            names.add(block.context.model_name.lower())
            names.add(block.context.display_name.lower())
            # A non-null skill_vs_persistence_pct is itself the licence to
            # describe the comparison baseline as "persistence" in prose,
            # even though persistence has no forecast entries of its own in
            # THIS packet -- the field exists specifically so the skill
            # comparison can be narrated.
            if block.context.skill_vs_persistence_pct is not None:
                names.add("persistence")
    if evidence.classification is not None:
        names.add(evidence.classification.model_name.lower())
    return names


def validate_grounding(response: GeminiStructuredResponse, evidence: EvidencePacket) -> list[str]:
    """Returns a list of violation category strings; empty means grounded.

    An empty list is the ONLY passing result -- callers must treat any
    non-empty list as "reject the response" (task §8: "If the validator
    cannot confidently determine that a claim is grounded -> reject.")."""
    violations: list[str] = []
    text = "\n".join([
        response.summary, response.intensity_explanation, response.track_explanation,
        response.classification_explanation, response.limitations,
    ])

    generic_pool, horizon_pool, percentage_pool, date_pool = _collect_pools(evidence)

    for pattern in FORBIDDEN_PATTERNS:
        for m in pattern.finditer(text):
            if not _is_negated(text, m.start(), m.end()):
                violations.append(f"forbidden_claim:{pattern.pattern}")

    for m in PERCENT_RE.finditer(text):
        claimed = float(m.group(1))
        if not percentage_pool or not _within_tolerance(claimed, percentage_pool):
            violations.append(f"unsupported_percentage:{claimed}")
    for m in CONFIDENCE_WORD_RE.finditer(text):
        claimed = float(m.group(1))
        if not percentage_pool or not _within_tolerance(claimed, percentage_pool):
            violations.append(f"unsupported_confidence:{claimed}")

    for m in HORIZON_RE.finditer(text):
        claimed_h = int(m.group(1))
        if claimed_h not in horizon_pool:
            violations.append(f"unsupported_horizon:{claimed_h}")

    # Extract and validate every ISO date/datetime as ONE unit, then strip it
    # from the text before the generic bare-number scan runs -- otherwise a
    # legitimately-grounded timestamp's own year/month/day/hour components
    # (e.g. the "06" in "2010-06-26") would be individually flagged as
    # ungrounded plain numbers, which they are not (they are calendar
    # components of an already-checked date, not separate magnitude claims).
    stripped_text = text
    for m in ISO_DATETIME_RE.finditer(text):
        date_part = m.group(0)[:10]
        if date_part not in date_pool:
            violations.append(f"unsupported_date:{date_part}")
    stripped_text = ISO_DATETIME_RE.sub(" DATE ", stripped_text)

    for raw in _iter_bare_numbers(stripped_text):
        claimed = float(raw)
        if not _within_tolerance(claimed, generic_pool):
            violations.append(f"unsupported_number:{claimed}")

    text_lower = text.lower()
    # Scoped to `classification_explanation` ONLY -- not the full combined
    # text. Several taxonomy labels (`Land`, `Shear`, `Eye`) are also
    # ordinary English words that legitimately appear elsewhere: "340 km
    # from land" (current_state.dist2land_km), "vertical wind shear"
    # (a known_limitations sentence Gemini is expected to be able to
    # paraphrase), "the storm's eye". Classification claims have no
    # business appearing outside their own dedicated field, so restricting
    # the scan there removes this entire class of false positive without
    # weakening the check on the field that actually matters.
    classification_text = response.classification_explanation
    if evidence.classification is not None:
        true_label = evidence.classification.class_label
        for label in KNOWN_CLASSIFICATION_LABELS:
            if re.search(rf"\b{re.escape(label)}\b", classification_text, re.IGNORECASE) and label != true_label:
                violations.append(f"unsupported_classification_label:{label}")
    else:
        for label in KNOWN_CLASSIFICATION_LABELS:
            if re.search(rf"\b{re.escape(label)}\b", classification_text, re.IGNORECASE):
                violations.append(f"unsupported_classification_label:{label}")

    known_names = _known_model_names(evidence)
    for token in KNOWN_MODEL_NAME_TOKENS:
        if token == "gemini":
            continue
        if token in text_lower and not any(token in n for n in known_names):
            violations.append(f"unsupported_model_name:{token}")

    return violations
