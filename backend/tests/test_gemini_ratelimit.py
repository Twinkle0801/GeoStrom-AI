"""Unit tests for `app.gemini.ratelimit` (Phase 12 §9/§14)."""

from __future__ import annotations

from app.gemini.ratelimit import RateLimiter


def _fake_clock():
    state = {"t": 0.0}
    def clock() -> float:
        return state["t"]
    def advance(seconds: float) -> None:
        state["t"] += seconds
    return clock, advance


def test_requests_within_the_limit_are_allowed():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    for _ in range(3):
        allowed, retry_after = limiter.allow("1.2.3.4")
        assert allowed is True
        assert retry_after == 0.0


def test_repeated_requests_beyond_the_limit_are_refused():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    for _ in range(3):
        assert limiter.allow("1.2.3.4")[0] is True
    allowed, retry_after = limiter.allow("1.2.3.4")
    assert allowed is False
    assert retry_after > 0.0


def test_different_keys_are_isolated_from_each_other():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("1.2.3.4")[0] is True
    # A second, different client IP must not be affected by the first's usage.
    assert limiter.allow("5.6.7.8")[0] is True
    # But the first IP is now over its own limit.
    assert limiter.allow("1.2.3.4")[0] is False


def test_window_rolls_over_after_it_elapses():
    clock, advance = _fake_clock()
    limiter = RateLimiter(max_requests=1, window_seconds=60.0, clock=clock)
    assert limiter.allow("1.2.3.4")[0] is True
    assert limiter.allow("1.2.3.4")[0] is False  # still within the same window
    advance(60.1)
    assert limiter.allow("1.2.3.4")[0] is True  # new window, counter reset


def test_reset_clears_all_counters():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("1.2.3.4")[0] is True
    assert limiter.allow("1.2.3.4")[0] is False
    limiter.reset()
    assert limiter.allow("1.2.3.4")[0] is True
