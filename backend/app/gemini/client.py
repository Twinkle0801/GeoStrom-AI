"""Gemini SDK access -- isolated here per docs/API_ARCHITECTURE.md §6.2
("DECISION: all Gemini calls originate in backend/app/gemini/"). No other
module imports `google.genai`.

Uses the current officially supported Google Gen AI Python SDK
(`google-genai`, `from google import genai`) -- the unified SDK, not the
deprecated `google-generativeai` package. Pinned in requirements.txt to the
exact version installed/verified in this environment (2.20.0), per this
project's established dependency-pinning convention (see e.g.
ml/requirements-deep-learning.txt).

Structural isolation (docs/API_ARCHITECTURE.md §8 Layer 1): `tools` is never
passed to `GenerateContentConfig`, so the model has no function-calling,
retrieval, or code-execution capability -- it is architecturally incapable
of reaching outside the evidence packet, not merely instructed not to.
"""

from __future__ import annotations

from typing import Protocol

from app.core.config import Settings


class GeminiTimeoutError(Exception):
    """The request exceeded the configured timeout."""


class GeminiAPIError(Exception):
    """Any other Gemini SDK/transport failure (auth, rate limit, 5xx, ...).

    Deliberately generic: the original exception's message is NEVER
    included verbatim (per the task's "never expose the API key... in
    error messages" -- an SDK-level exception could in principle echo
    request details we do not want to risk surfacing). Callers get a
    stable, safe message; the caller logs only the exception TYPE name for
    diagnostics (see `service.py`), never `str(exc)`.
    """


class GeminiEmptyResponseError(Exception):
    """Gemini returned no text/candidates."""


class GeminiClientProtocol(Protocol):
    def generate_structured(
        self, *, system_instruction: str, user_content: str,
        response_schema: type, max_output_tokens: int, temperature: float,
        timeout_seconds: float,
    ) -> str:
        """Returns the raw JSON text of Gemini's structured response.

        Raises `GeminiTimeoutError`, `GeminiEmptyResponseError`, or
        `GeminiAPIError` on failure. Never raises anything else -- callers
        rely on this to keep the core API from ever crashing on a Gemini
        failure (task §14)."""
        ...


class RealGeminiClient:
    """Thin wrapper around `google.genai.Client`. Holds the API key only in
    memory (never logged, never returned); constructed once per process
    from `Settings.gemini_api_key`, never hardcoded."""

    def __init__(self, api_key: str, model: str):
        from google import genai  # local import: never required for tests/mocked paths
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_structured(
        self, *, system_instruction: str, user_content: str,
        response_schema: type, max_output_tokens: int, temperature: float,
        timeout_seconds: float,
    ) -> str:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
                    # Explanation from a structured evidence packet is not a
                    # frontier-reasoning task (docs/API_ARCHITECTURE.md
                    # §8.1: "select on latency and cost"), so the thinking
                    # budget is capped small and fixed (not left dynamic/
                    # unbounded) -- found necessary during Phase 9's own
                    # manual smoke test: with no cap, this model spent an
                    # unpredictable, sometimes-large share of
                    # `max_output_tokens` on invisible "thinking" tokens,
                    # occasionally truncating the visible JSON output
                    # mid-string and failing schema validation. A budget of
                    # 0 (fully disabled) was rejected outright by this model
                    # as an invalid argument; a small fixed budget is the
                    # verified-working, latency-bounded middle ground.
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                    # `tools` intentionally omitted -- see module docstring.
                ),
            )
        except TimeoutError as exc:
            raise GeminiTimeoutError("Gemini request timed out") from exc
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see GeminiAPIError docstring
            if _looks_like_timeout(exc):
                raise GeminiTimeoutError("Gemini request timed out") from exc
            raise GeminiAPIError("Gemini request failed") from exc

        text = getattr(response, "text", None)
        if not text:
            raise GeminiEmptyResponseError("Gemini returned an empty response")
        return text


def _looks_like_timeout(exc: Exception) -> bool:
    """The SDK raises various transport-layer exception types for a timed-out
    call depending on the underlying HTTP client; this checks the exception
    TYPE name only (never the message body, which could echo request
    details) for a timeout-shaped name."""
    return "timeout" in type(exc).__name__.lower()


def build_gemini_client(settings: Settings) -> GeminiClientProtocol | None:
    """Returns `None` (not an error) when no API key is configured -- the
    service layer treats a missing client exactly like any other Gemini
    unavailability and falls back, per task §9."""
    if not settings.gemini_api_key:
        return None
    return RealGeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
