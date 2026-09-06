"""Lightweight, process-local, fixed-window rate limiter guarding real
Gemini calls made by POST /api/v1/explain/forecast.

This project has no authentication and no user identity (Phase 11's
explicit, retained constraint) -- so the only available key is the
client's IP address, via `request.client.host`. This is a conservative,
single-process mitigation against accidental request storms and quota
exhaustion, NOT distributed or enterprise-grade rate limiting: counters
live in one process's memory, reset on restart, and are not shared across
multiple worker processes. Documented, not oversold, in
docs/PHASE_12_RELEASE_AUDIT.md.

Applied only to the cache-miss path (app/api/v1/explain.py) -- a cache hit
makes no Gemini call and so is never rate-limited, and the deterministic
fallback template itself is pure evidence-derived computation, not a
Gemini call, so it is never separately rate-limited either.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    def __init__(
        self, max_requests: int, window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max_requests = max_requests
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        """Returns `(allowed, retry_after_seconds)`. `retry_after_seconds`
        is `0.0` when allowed. A fixed window: once `max_requests` have
        been seen for `key` within the current window, every further
        request in that window is refused until the window rolls over."""
        with self._lock:
            now = self._clock()
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= self._window:
                window_start, count = now, 0
            count += 1
            self._windows[key] = (window_start, count)
            if count > self._max_requests:
                retry_after = max(0.0, self._window - (now - window_start))
                return False, retry_after
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
