"""Unit tests for `app.gemini.cache` (Phase 12 §8/§14).

Uses an injectable fake clock (a mutable single-element list plus a
closure) rather than real `time.sleep`, so TTL-expiry behaviour is tested
deterministically and instantly.
"""

from __future__ import annotations

import datetime as dt

from app.gemini.cache import ExplainCache, explain_cache_key
from app.gemini.schemas import EvidencePacket, StormEvidence


def _make_evidence(sid: str = "2010176N16278", generated_at: dt.datetime | None = None) -> EvidencePacket:
    return EvidencePacket(
        generated_at=generated_at or dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        storm=StormEvidence(
            sid=sid, name=None, season=2010, basin="NA",
            start_time=dt.datetime(2010, 6, 26, tzinfo=dt.timezone.utc),
            end_time=dt.datetime(2010, 6, 27, tzinfo=dt.timezone.utc),
            n_observations=5,
        ),
        current_state=None, recent_history=[], intensity=None, track=None,
        classification=None, known_limitations=[], forbidden_claims=[],
    )


def _fake_clock():
    """Returns (clock_fn, advance_fn); clock_fn is the injectable clock,
    advance_fn moves it forward by N seconds."""
    state = {"t": 0.0}
    def clock() -> float:
        return state["t"]
    def advance(seconds: float) -> None:
        state["t"] += seconds
    return clock, advance


def test_cache_miss_on_empty_cache():
    cache = ExplainCache(maxsize=10, ttl_seconds=60.0)
    assert cache.get("nonexistent-key") is None


def test_cache_hit_returns_the_stored_value():
    cache = ExplainCache(maxsize=10, ttl_seconds=60.0)
    cache.set("k1", "value-1")
    assert cache.get("k1") == "value-1"


def test_cache_key_is_identical_for_identical_evidence():
    e1 = _make_evidence()
    e2 = _make_evidence()
    assert explain_cache_key(e1) == explain_cache_key(e2)


def test_cache_key_ignores_generated_at_only():
    e1 = _make_evidence(generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    e2 = _make_evidence(generated_at=dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc))
    assert explain_cache_key(e1) == explain_cache_key(e2)


def test_cache_key_differs_for_different_storms_isolation():
    e1 = _make_evidence(sid="2010176N16278")
    e2 = _make_evidence(sid="2015313N22289")
    assert explain_cache_key(e1) != explain_cache_key(e2)


def test_cache_entries_for_different_keys_do_not_collide():
    cache = ExplainCache(maxsize=10, ttl_seconds=60.0)
    key_a, key_b = explain_cache_key(_make_evidence("A")), explain_cache_key(_make_evidence("B"))
    cache.set(key_a, "value-a")
    cache.set(key_b, "value-b")
    assert cache.get(key_a) == "value-a"
    assert cache.get(key_b) == "value-b"


def test_ttl_expiry_evicts_the_entry():
    clock, advance = _fake_clock()
    cache = ExplainCache(maxsize=10, ttl_seconds=100.0, clock=clock)
    cache.set("k1", "value-1")
    assert cache.get("k1") == "value-1"  # still fresh
    advance(99.9)
    assert cache.get("k1") == "value-1"  # just under TTL
    advance(0.2)
    assert cache.get("k1") is None  # now expired
    assert len(cache) == 0  # expired entry was actually removed, not just hidden


def test_bounded_size_evicts_the_least_recently_used_entry():
    cache = ExplainCache(maxsize=2, ttl_seconds=3600.0)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.get("k1")  # touch k1 so it is now the most-recently-used
    cache.set("k3", "v3")  # over capacity -- must evict k2, not k1
    assert len(cache) == 2
    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_clear_empties_the_cache():
    cache = ExplainCache(maxsize=10, ttl_seconds=3600.0)
    cache.set("k1", "v1")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("k1") is None
