"""Bounded, in-process TTL cache for VALIDATED Gemini explanations only.

Phase 11 recorded a real Gemini call at ~12.5s -- by far the dominant
latency in the whole system. This cache exists solely to avoid repeating
that call for an evidence-identical request (e.g. a user reloading the
same storm/model page), never to serve stale or unvalidated content.

Restart semantics (task's explicit "if in-memory, document restart
semantics"): this is a plain in-process dict guarded by a lock. It is
cleared whenever the backend process restarts, and is NOT shared across
multiple worker processes if the app is ever run with more than one
uvicorn/gunicorn worker. That is an accepted limitation for this
retrospective, single-process research prototype -- see
docs/PHASE_12_RELEASE_AUDIT.md. Introducing Redis or another external
cache purely for this would be new infrastructure disproportionate to a
project with no concurrent-user production traffic.

What is cached: only a `GeminiExplanationService` result whose
`source == "gemini"` (i.e. it already passed `validate_grounding`). A
`"fallback"` result (timeout, api_error, malformed_json, ungrounded_claim,
not_configured) is NEVER cached -- the caller (app/api/v1/explain.py)
enforces this; this module has no opinion on what is stored, but nothing
in this project ever puts a fallback result in it.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Callable

from app.gemini.schemas import EvidencePacket


def explain_cache_key(evidence: EvidencePacket) -> str:
    """A SHA-256 hash of the ENTIRE evidence packet content -- storm id,
    forecast origin timestamp, resolved model name/version for both tasks,
    observed history, forecast values, evidence-schema version -- excluding
    only `generated_at` (a fresh wall-clock stamp set on every packet build
    that must not itself defeat the cache).

    Hashing the whole packet, rather than hand-picking a few fields, is a
    deliberate over-approximation: ANY difference in evidence content --
    new predictions ingested for a later origin time, a different requested
    model version, a different storm, a materially different observed
    history -- produces a different key. A cache hit is therefore only ever
    returned for evidence that is byte-identical (module the timestamp) to
    what was previously validated, which is what "prevent stale
    cross-model/cross-storm responses" requires.
    """
    payload = evidence.model_dump(mode="json", exclude={"generated_at"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExplainCache:
    """LRU-with-TTL, bounded size, thread-safe (a lock guards every access
    since FastAPI may run sync path operations across multiple worker
    threads)."""

    def __init__(
        self, maxsize: int, ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, tuple[float, object]]" = OrderedDict()

    def get(self, key: str) -> object | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = (self._clock() + self._ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict least-recently-used

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
