"""`GeminiExplanationService` -- orchestrates evidence packet -> Gemini ->
structured validation -> safe explanation -> deterministic fallback.

Per task §13: this service does NOT know how ML models generate
predictions. It consumes the `EvidencePacket` contract only
(`evidence_builder.py` is a separate, DB-aware caller that constructs one).

Retry policy (task §15, "no automatic retries without control"): a
malformed-JSON or failed-grounding response is retried at most
`Settings.gemini_max_retries` times (default 1), with the violation quoted
back to Gemini, per `docs/API_ARCHITECTURE.md` §8 Layer 4 step 4. A
transport-level failure (timeout / API error / empty response) is NEVER
retried automatically here -- it falls back immediately, so total request
latency stays bounded by one Gemini call plus at most
`gemini_max_retries` more, never an open-ended retry loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.core.config import Settings
from app.gemini.client import (
    GeminiAPIError, GeminiClientProtocol, GeminiEmptyResponseError, GeminiTimeoutError,
)
from app.gemini.fallback import build_fallback_explanation
from app.gemini.prompts import SYSTEM_INSTRUCTION, build_user_content
from app.gemini.schemas import EvidencePacket, GeminiStructuredResponse
from app.gemini.validator import validate_grounding

logger = logging.getLogger("app.gemini")


@dataclass
class ExplainResult:
    source: str  # "gemini" | "fallback"
    explanation: GeminiStructuredResponse
    fallback_reason: str | None = None
    violations: list[str] = field(default_factory=list)


class GeminiExplanationService:
    def __init__(self, client: GeminiClientProtocol | None, settings: Settings):
        self._client = client
        self._settings = settings

    def explain(self, evidence: EvidencePacket) -> ExplainResult:
        if self._client is None:
            return self._fallback(evidence, "not_configured")

        max_attempts = 1 + max(0, self._settings.gemini_max_retries)
        retry_note: str | None = None
        last_violations: list[str] = []

        for attempt in range(max_attempts):
            user_content = build_user_content(evidence)
            if retry_note:
                user_content += (
                    "\n\nYour previous answer was rejected for these reasons -- fix them and "
                    f"answer again using ONLY the evidence packet above: {retry_note}"
                )
            try:
                raw = self._client.generate_structured(
                    system_instruction=SYSTEM_INSTRUCTION,
                    user_content=user_content,
                    response_schema=GeminiStructuredResponse,
                    max_output_tokens=self._settings.gemini_max_output_tokens,
                    temperature=self._settings.gemini_temperature,
                    timeout_seconds=self._settings.gemini_timeout_seconds,
                )
            except GeminiTimeoutError:
                self._log_safe("fallback", "timeout")
                return self._fallback(evidence, "timeout")
            except GeminiEmptyResponseError:
                self._log_safe("fallback", "empty_response")
                return self._fallback(evidence, "empty_response")
            except GeminiAPIError:
                self._log_safe("fallback", "api_error")
                return self._fallback(evidence, "api_error")

            try:
                parsed = GeminiStructuredResponse.model_validate_json(raw)
            except (ValidationError, ValueError):
                retry_note = "the previous response was not valid JSON matching the required schema"
                last_violations = ["malformed_json"]
                continue

            violations = validate_grounding(parsed, evidence)
            if not violations:
                self._log_safe("gemini", None)
                return ExplainResult(source="gemini", explanation=parsed)
            last_violations = violations
            retry_note = "; ".join(violations)

        self._log_safe("fallback", "ungrounded_claim", violations=last_violations)
        return self._fallback(evidence, "ungrounded_claim", violations=last_violations)

    def _fallback(self, evidence: EvidencePacket, reason: str,
                  violations: list[str] | None = None) -> ExplainResult:
        return ExplainResult(
            source="fallback", explanation=build_fallback_explanation(evidence),
            fallback_reason=reason, violations=violations or [],
        )

    @staticmethod
    def _log_safe(source: str, reason: str | None, violations: list[str] | None = None) -> None:
        """Per task §26: never log the API key, headers, or raw response text
        -- only the safe status/category fields."""
        logger.info(
            "gemini_explanation source=%s fallback_reason=%s violation_count=%d",
            source, reason, len(violations or []),
        )
