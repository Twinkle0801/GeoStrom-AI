"""Configurable, idempotent HURSAT-B1 archive downloader.

Deliberately conservative: downloads exactly the storm archives it is told
to (never "the whole archive"), skips anything already on disk, and supports
a dry-run mode that performs discovery (small HTML listings only) without
transferring any imagery. See ml/scripts/discover_hursat_archive.py for the
companion discovery-only estimator required before any real download.
"""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ml.geostrom_ml.satellite.discovery import HURSAT_BASE, resolve_archive


@dataclass
class DownloadResult:
    sid: str
    season: int
    status: str  # "downloaded" | "cached" | "missing_from_archive" | "dry_run" | "error"
    archive_path: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    n_extracted_files: int | None = None
    detail: str | None = None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_storm_archive(
    sid: str,
    season: int,
    raw_dir: Path,
    listing_cache_dir: Path,
    *,
    dry_run: bool = False,
    timeout: int = 180,
) -> DownloadResult:
    """Resolve, then (unless dry_run) download one storm's HURSAT archive.

    Idempotent: if the archive already exists on disk with a non-zero size,
    it is treated as cached and not re-downloaded.
    """
    entry = resolve_archive(sid, season, listing_cache_dir)
    if entry is None:
        return DownloadResult(sid, season, "missing_from_archive",
                               detail=f"no HURSAT-B1 archive for {sid} in season {season} listing")

    dest_dir = raw_dir / str(season)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["filename"]

    if dry_run:
        return DownloadResult(sid, season, "dry_run", archive_path=str(dest),
                               bytes=entry.get("bytes"))

    if dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(sid, season, "cached", archive_path=str(dest),
                               bytes=dest.stat().st_size, sha256=sha256_of(dest))

    url = f"{HURSAT_BASE}/{season}/{entry['filename']}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        return DownloadResult(sid, season, "error", detail=f"{type(exc).__name__}: {exc}")

    return DownloadResult(sid, season, "downloaded", archive_path=str(dest),
                           bytes=dest.stat().st_size, sha256=sha256_of(dest))


ADT_BASE = "https://www.ncei.noaa.gov/data/oceans/archive/arc0239/0307249/1.1/data/0-data"


def download_adt_archive(sid: str, adt_dir: Path, *, dry_run: bool = False,
                          timeout: int = 120) -> DownloadResult:
    """Download one storm's ADT-HURSAT record file, if NCEI has one.

    A 404 is an expected, non-error outcome (docs/DATA_STRATEGY.md §4.3:
    ADT coverage is not guaranteed for every storm) -- reported as
    'missing_from_archive', never raised.
    """
    adt_dir.mkdir(parents=True, exist_ok=True)
    dest = adt_dir / f"{sid}.nc"

    if dry_run:
        return DownloadResult(sid, 0, "dry_run", archive_path=str(dest))
    if dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(sid, 0, "cached", archive_path=str(dest), bytes=dest.stat().st_size)

    try:
        resp = requests.get(f"{ADT_BASE}/{sid}.nc", timeout=timeout)
        if resp.status_code == 404:
            return DownloadResult(sid, 0, "missing_from_archive", detail="404 from NCEI")
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        return DownloadResult(sid, 0, "error", detail=f"{type(exc).__name__}: {exc}")

    return DownloadResult(sid, 0, "downloaded", archive_path=str(dest), bytes=dest.stat().st_size)


def extract_archive(archive_path: Path, interim_dir: Path) -> list[Path]:
    """Extract a HURSAT storm .tar.gz into interim/hursat/<sid>/*.nc.

    Idempotent: if the destination already has extracted .nc files, skip
    re-extraction and just return the existing list.
    """
    stem = archive_path.name
    for suffix in (".tar.gz", ".tgz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    dest = interim_dir / stem
    existing = sorted(dest.glob("*.hursat-b1.v06.nc")) if dest.exists() else []
    if existing:
        return existing

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        safe_members = []
        for member in tf.getmembers():
            member_path = (dest / member.name).resolve()
            if not str(member_path).startswith(str(dest.resolve())):
                continue  # path-traversal guard; never trust archive-internal paths blindly
            safe_members.append(member)
        tf.extractall(dest, members=safe_members)  # noqa: S202 -- filtered above

    return sorted(dest.rglob("*.hursat-b1.v06.nc"))
