"""HURSAT-B1 NetCDF parsing, metadata extraction, and IRWIN physical QC.

Consolidates the validated Phase 1 frame-parsing logic (originally inline in
`ml/scripts/verify_hursat_join.py::inventory_frames` and
`ml/scripts/qc_gate.py::load_frames`) into a reusable library component, per
the same reuse rule already applied to `ml/geostrom_ml/data/ibtracs.py`. Both
Phase 1 scripts are left unmodified as historical artifacts.

Two-pass design, deliberately: `inventory_frames` reads only small scalar
metadata (fast, low memory, safe to run over thousands of files at once).
`read_irwin` re-opens a single file to pull the full IRWIN grid, and is only
called for frames that survive discovery + deduplication + QC -- so the
(comparatively large) pixel arrays for rejected/duplicate frames are never
materialised.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ml.geostrom_ml.satellite.schema import IRWIN_VALID_RANGE_K

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

FILL = -1.0  # HURSAT global attribute: "Missing Values are -1.0"
EXPECTED_GRID = (301, 301)

# <SID>.<NAME>.<YYYY>.<MM>.<DD>.<HHMM>.<vza>.<SAT>.<sss>.hursat-b1.v06.nc
FRAME_RE = re.compile(
    r"^(?P<sid>\d{7}[NS]\d{5})\.(?P<name>[^.]+)\.(?P<Y>\d{4})\.(?P<M>\d{2})\.(?P<D>\d{2})"
    r"\.(?P<hhmm>\d{4})\.(?P<f1>\d+)\.(?P<sat>[^.]+)\.(?P<f2>\d+)\.hursat-b1\.v06\.nc$"
)


def discover_frame_files(root_dir: Path | list[Path]) -> list[Path]:
    """Recursively find every extracted HURSAT-B1 NetCDF frame under one or
    more directories (data arrives via two acquisition paths: the Phase 1
    verification sample under `samples/hursat/`, and Phase 4 downloads
    extracted to `interim/hursat/` -- both are valid inputs to the same
    pipeline). Duplicate paths across directories are not expected but are
    de-duplicated defensively."""
    roots = [root_dir] if isinstance(root_dir, Path) else list(root_dir)
    found: set[Path] = set()
    for root in roots:
        if root.exists():
            found.update(root.rglob("*.hursat-b1.v06.nc"))
    return sorted(found)


def _scalar(ds: xr.Dataset, name: str):
    if name not in ds.variables:
        return None
    v = ds[name].values
    v = v.item() if getattr(v, "size", 0) == 1 else (v.ravel()[0] if getattr(v, "size", 0) else None)
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip()
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f) or f == FILL:
        return None
    return f


def parse_frame_metadata(nc_path: Path) -> dict:
    """Read one HURSAT NetCDF's scalar metadata (no pixel arrays)."""
    m = FRAME_RE.match(nc_path.name)
    rec: dict = {
        "source_file": str(nc_path),
        "file": nc_path.name,
        "filename_parsed": bool(m),
    }
    if m:
        rec.update({
            "sid_fname": m.group("sid"),
            "name_fname": m.group("name"),
            "satellite_fname": m.group("sat"),
        })
    try:
        with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
            rec["storm_id"] = ds.attrs.get("TC_serial_number")
            rec["satellite_id"] = ds.attrs.get("Satellite_Name")
            rec["satellite_timestamp"] = pd.Timestamp(ds["htime"].values[0]).round("s")
            rec["nlat"], rec["nlon"] = ds.sizes.get("lat"), ds.sizes.get("lon")
            rec["channels"] = ",".join(
                c for c in ("IRWIN", "IRWVP", "IRNIR", "IRSPL", "VSCHN", "VSVAR", "IRVAR")
                if c in ds.variables)
            rec["vza"] = _scalar(ds, "VZA")
            rec["satellite_lat"] = _scalar(ds, "CentLat")
            rec["satellite_lon"] = _scalar(ds, "CentLon")
            has_irwin = "IRWIN" in ds.variables
            rec["has_irwin"] = has_irwin
            if has_irwin:
                a = np.asarray(ds["IRWIN"].values, dtype="float32").ravel()
                valid = np.isfinite(a) & (a >= IRWIN_VALID_RANGE_K[0]) & (a <= IRWIN_VALID_RANGE_K[1])
                rec["irwin_valid_pct"] = round(100.0 * valid.sum() / a.size, 4) if a.size else 0.0
                rec["irwin_constant"] = bool(valid.any() and a[valid].min() == a[valid].max())
    except Exception as exc:  # noqa: BLE001 -- recorded, never silently swallowed
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def inventory_frames(paths: list[Path]) -> pd.DataFrame:
    """Metadata-only inventory of a list of HURSAT frame files."""
    if not paths:
        return pd.DataFrame(columns=["source_file", "error"])
    return pd.DataFrame(parse_frame_metadata(p) for p in paths)


class FrameOpenError(RuntimeError):
    pass


def read_irwin(nc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-open one frame and return (irwin_kelvin[float32], valid_mask[bool]).

    Grid shape is asserted to be the Phase 1 verified (301, 301); a
    mismatched shape is a loud failure, never silently reshaped.
    """
    try:
        with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
            if "IRWIN" not in ds.variables:
                raise FrameOpenError(f"{nc_path}: no IRWIN variable")
            a = np.asarray(ds["IRWIN"].values, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, FrameOpenError):
            raise
        raise FrameOpenError(f"{nc_path}: {type(exc).__name__}: {exc}") from exc

    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.shape != EXPECTED_GRID:
        raise FrameOpenError(
            f"{nc_path}: unexpected IRWIN grid shape {a.shape}, expected {EXPECTED_GRID}")

    valid = np.isfinite(a) & (a >= IRWIN_VALID_RANGE_K[0]) & (a <= IRWIN_VALID_RANGE_K[1])
    out = np.where(valid, a, np.nan).astype("float32")
    return out, valid
