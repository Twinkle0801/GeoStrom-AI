"""A fake/mock Gemini client for tests (task §24). Never uses a real API key
or makes a network call. Supports controlled responses: valid, malformed,
hallucinated, timeout, exception, prompt-injection-style, empty."""

from __future__ import annotations

from app.gemini.client import GeminiAPIError, GeminiEmptyResponseError, GeminiTimeoutError


class MockGeminiClient:
    def __init__(self, *, responses: list[str] | None = None, raise_exc: Exception | None = None):
        """`responses`: one JSON string returned per call, in order (last one
        repeats if more calls happen than entries -- e.g. during a retry).
        `raise_exc`: if set, every call raises this exception instead."""
        self._responses = responses
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def generate_structured(
        self, *, system_instruction: str, user_content: str, response_schema: type,
        max_output_tokens: int, temperature: float, timeout_seconds: float,
    ) -> str:
        self.calls.append({
            "system_instruction": system_instruction, "user_content": user_content,
            "max_output_tokens": max_output_tokens, "temperature": temperature,
            "timeout_seconds": timeout_seconds,
        })
        if self._raise_exc is not None:
            raise self._raise_exc
        if not self._responses:
            raise GeminiEmptyResponseError("mock configured with no responses")
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        text = self._responses[idx]
        if not text:
            # Mirrors RealGeminiClient's own contract: never returns an
            # empty string, raises instead.
            raise GeminiEmptyResponseError("mock returned an empty response")
        return text


def timeout_client() -> MockGeminiClient:
    return MockGeminiClient(raise_exc=GeminiTimeoutError("mock timeout"))


def api_error_client() -> MockGeminiClient:
    return MockGeminiClient(raise_exc=GeminiAPIError("mock api error"))


def empty_response_client() -> MockGeminiClient:
    return MockGeminiClient(responses=[""])
