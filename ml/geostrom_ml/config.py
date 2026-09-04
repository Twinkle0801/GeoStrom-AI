"""Central configuration and DATA_ROOT resolution for GeoStrom AI.

Single source of truth for where data lives. Enforces the Phase 0 rule that
datasets never sit inside OneDrive or inside the Git repository — see
docs/DATA_STRATEGY.md §6 "Zone rules".

Resolution order for DATA_ROOT:
    1. explicit argument
    2. DATA_ROOT environment variable
    3. `.env` file at the repository root
    4. DEFAULT_DATA_ROOT below
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (ml/geostrom_ml/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Documented default. Overridable via DATA_ROOT; see .env.example.
DEFAULT_DATA_ROOT = Path("C:/GeoStromData")

# Zone layout (docs/DATA_STRATEGY.md §6)
ZONES = ("raw", "interim", "processed", "datasets", "samples", "reports")


class DataRootError(RuntimeError):
    """DATA_ROOT is missing or violates a safety rule."""


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader. Avoids a python-dotenv dependency for one variable."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def _onedrive_roots() -> list[Path]:
    roots: list[Path] = []
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        val = os.environ.get(var)
        if val:
            resolved = Path(val).resolve()
            if resolved not in roots:
                roots.append(resolved)
    return roots


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_data_root(path: Path) -> list[str]:
    """Return a list of safety violations. Empty list means the path is safe."""
    problems: list[str] = []
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        problems.append(f"DATA_ROOT must be an absolute path, got: {resolved}")

    for od in _onedrive_roots():
        if _is_within(resolved, od):
            problems.append(
                f"DATA_ROOT ({resolved}) is inside OneDrive ({od}). "
                "Sync will corrupt or lock large files mid-write."
            )

    if _is_within(resolved, REPO_ROOT):
        problems.append(
            f"DATA_ROOT ({resolved}) is inside the Git repository ({REPO_ROOT}). "
            "Datasets must never be committed."
        )

    return problems


def get_data_root(explicit: str | Path | None = None, *, create: bool = False) -> Path:
    """Resolve, validate, and optionally create DATA_ROOT."""
    if explicit is not None:
        raw, source = Path(explicit), "explicit argument"
    elif os.environ.get("DATA_ROOT"):
        raw, source = Path(os.environ["DATA_ROOT"]), "DATA_ROOT environment variable"
    else:
        dotenv = _load_dotenv(REPO_ROOT / ".env")
        if dotenv.get("DATA_ROOT"):
            raw, source = Path(dotenv["DATA_ROOT"]), ".env file"
        else:
            raw, source = DEFAULT_DATA_ROOT, "DEFAULT_DATA_ROOT fallback"

    path = raw.expanduser()
    problems = check_data_root(path)
    if problems:
        detail = "\n  - ".join(problems)
        raise DataRootError(
            f"Unsafe DATA_ROOT (from {source}):\n  - {detail}\n"
            "Set a safe DATA_ROOT in .env — see .env.example."
        )

    if create:
        for zone in ZONES:
            (path / zone).mkdir(parents=True, exist_ok=True)
    return path


def zone(name: str, *parts: str, create: bool = False) -> Path:
    """Path inside a DATA_ROOT zone, e.g. zone('raw', 'ibtracs')."""
    if name not in ZONES:
        raise ValueError(f"Unknown zone {name!r}; expected one of {ZONES}")
    path = get_data_root().joinpath(name, *parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# Repo-tracked metadata locations (small files; these ARE committed)
MANIFEST_DIR = REPO_ROOT / "ml" / "manifests"
REPORT_DIR = REPO_ROOT / "ml" / "reports"


if __name__ == "__main__":
    print(f"REPO_ROOT       : {REPO_ROOT}")
    print(f"OneDrive roots  : {[str(p) for p in _onedrive_roots()]}")
    try:
        root = get_data_root(create=True)
        print(f"DATA_ROOT       : {root}  [OK - safe, zones created]")
        for z in ZONES:
            print(f"  {z:<12}: {root / z}")
    except DataRootError as exc:
        print(f"DATA_ROOT       : REJECTED\n{exc}")
