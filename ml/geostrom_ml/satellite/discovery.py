"""HURSAT-B1 archive discovery: season directory listings -> per-storm URLs.

Consolidates the validated Phase 1 listing-fetch/parse logic (originally
inline in `ml/scripts/verify_crosswalk.py::fetch_season_listing`) into a
reusable library component, per the same reuse rule already applied to
`ml/geostrom_ml/data/ibtracs.py`. `verify_crosswalk.py` is left unmodified as
the Phase 1 historical artifact.

This module makes network requests ONLY for small (~10-50 KB) Apache
directory-listing HTML pages -- never for imagery. Listings are cached to
disk so repeated runs (dry-run estimation, then real download) do not
re-fetch them.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

HURSAT_BASE = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"

# HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz
FNAME_RE = re.compile(
    r"HURSAT_b1_v06_(?P<sid>\d{4}\d{3}[NS]\d{5})_(?P<name>[A-Z0-9\-]+)_c(?P<created>\d{8})\.tar\.gz"
)
# IBTrACS SID grammar: YYYY + DDD(day-of-year) + hemisphere + lat(3) + lon(3)
SID_RE = re.compile(r"^\d{4}\d{3}[NS]\d{5}$")


def fetch_season_listing(season: int, cache_dir: Path, *, timeout: int = 120) -> list[dict]:
    """Return parsed HURSAT archive entries for a season (cached on disk).

    Each entry: {filename, sid, name, created, bytes}. Ported verbatim from
    the Phase 1 verification script (see module docstring).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"hursat_v06_{season}_listing.html"
    if cache.exists():
        html = cache.read_text(encoding="utf-8", errors="replace")
    else:
        resp = requests.get(f"{HURSAT_BASE}/{season}/", timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        cache.write_text(html, encoding="utf-8")

    entries = []
    for m in FNAME_RE.finditer(html):
        tail = html[m.end(): m.end() + 260]
        size_m = re.search(r"(\d{4,})\s*<", tail)
        entries.append({
            "filename": m.group(0),
            "sid": m.group("sid"),
            "name": m.group("name"),
            "created": m.group("created"),
            "bytes": int(size_m.group(1)) if size_m else None,
        })
    seen, out = set(), []
    for e in entries:
        if e["filename"] not in seen:
            seen.add(e["filename"])
            out.append(e)
    return out


def resolve_archive(sid: str, season: int, cache_dir: Path) -> dict | None:
    """Find the HURSAT archive entry for one SID within its season's listing.

    Returns None if the season listing has no archive for this SID (storm
    simply lacks HURSAT coverage -- a documented, expected outcome, not an
    error; see docs/DATA_STRATEGY.md check #9, ~96% NA coverage).
    """
    entries = fetch_season_listing(season, cache_dir)
    for e in entries:
        if e["sid"] == sid:
            return e
    return None


def season_of_sid(sid: str) -> int:
    """First 4 characters of an IBTrACS SID are its season (year)."""
    return int(sid[:4])
