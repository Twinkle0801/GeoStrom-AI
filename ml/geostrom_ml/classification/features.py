"""Phase 5 Task 7: deterministic, physically-grounded image feature extraction.

Every feature is a pure function of one frame's `irwin_k` (float32 Kelvin)
and `valid_mask` (bool) arrays from the canonical Zarr store -- nothing
else. In particular:

  - No IBTrACS field of any kind is read here (no `usa_wind`, no
    `storms.max_wind`, no track/intensity data at all) -- satisfies the
    task's explicit "do not use future observations / future intensity /
    future track information" rule by construction: these functions do not
    even receive that data as an argument.
  - No feature is a disguised copy of `scene_label` -- every feature below
    is a generic radiometric or spatial-structure statistic that would be
    computed identically regardless of what the ADT algorithm decided the
    Scene was.
  - Purely numpy arithmetic on a fixed-shape array -> deterministic by
    construction: the same (kelvin, valid_mask) pair always produces the
    same feature vector, no randomness anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GRID = (301, 301)
CENTER_RADIUS_FRAC = 0.15  # inner "core" disk, as a fraction of the half-width
RING_OUTER_FRAC = 0.45     # annulus outer edge

FEATURE_NAMES: list[str] = [
    "mean_k", "std_k", "min_k", "max_k",
    "p05_k", "p25_k", "p50_k", "p75_k", "p95_k",
    "valid_fraction",
    "center_mean_k", "ring_mean_k", "outer_mean_k",
    "center_minus_ring_k", "ring_minus_outer_k",
    "quad_nw_mean_k", "quad_ne_mean_k", "quad_sw_mean_k", "quad_se_mean_k",
    "quad_std_k",  # spatial asymmetry: std across the four quadrant means
]


def _radial_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(cy, cx)
    center = r <= CENTER_RADIUS_FRAC
    ring = (r > CENTER_RADIUS_FRAC) & (r <= RING_OUTER_FRAC)
    outer = r > RING_OUTER_FRAC
    return center, ring, outer


_CENTER_MASK, _RING_MASK, _OUTER_MASK = _radial_masks(GRID)


def extract_features(kelvin: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    """Extract the fixed, deterministic feature vector for one frame.

    Returns NaN for any statistic with insufficient valid pixels to compute
    (e.g. an entirely-invalid region) -- never a fabricated 0.0.
    """
    if kelvin.shape != GRID or valid_mask.shape != GRID:
        raise ValueError(f"expected {GRID} arrays, got kelvin={kelvin.shape} mask={valid_mask.shape}")

    valid = valid_mask.astype(bool)
    vals = kelvin[valid]
    out: dict[str, float] = {}

    out["valid_fraction"] = float(valid.mean())
    if vals.size == 0:
        nan_out = {name: float("nan") for name in FEATURE_NAMES}
        nan_out["valid_fraction"] = out["valid_fraction"]  # 0.0 is a real, computable value
        return nan_out

    out["mean_k"] = float(vals.mean())
    out["std_k"] = float(vals.std())
    out["min_k"] = float(vals.min())
    out["max_k"] = float(vals.max())
    for p, name in ((5, "p05_k"), (25, "p25_k"), (50, "p50_k"), (75, "p75_k"), (95, "p95_k")):
        out[name] = float(np.percentile(vals, p))

    def _region_mean(region_mask: np.ndarray) -> float:
        m = region_mask & valid
        return float(kelvin[m].mean()) if m.any() else float("nan")

    out["center_mean_k"] = _region_mean(_CENTER_MASK)
    out["ring_mean_k"] = _region_mean(_RING_MASK)
    out["outer_mean_k"] = _region_mean(_OUTER_MASK)
    out["center_minus_ring_k"] = out["center_mean_k"] - out["ring_mean_k"]
    out["ring_minus_outer_k"] = out["ring_mean_k"] - out["outer_mean_k"]

    h, w = GRID
    hy, hx = h // 2, w // 2
    quads = {
        "quad_nw_mean_k": (slice(0, hy), slice(0, hx)),
        "quad_ne_mean_k": (slice(0, hy), slice(hx, w)),
        "quad_sw_mean_k": (slice(hy, h), slice(0, hx)),
        "quad_se_mean_k": (slice(hy, h), slice(hx, w)),
    }
    quad_means = []
    for name, (ys, xs) in quads.items():
        m = valid[ys, xs]
        v = float(kelvin[ys, xs][m].mean()) if m.any() else float("nan")
        out[name] = v
        quad_means.append(v)
    finite = [q for q in quad_means if q == q]  # drop NaN
    out["quad_std_k"] = float(np.std(finite)) if len(finite) >= 2 else float("nan")

    return {name: out[name] for name in FEATURE_NAMES}


def build_feature_matrix(index: pd.DataFrame, zarr_path: Path) -> pd.DataFrame:
    """Extract features for every row of a classification index.

    `index` must have `sample_id` and `zarr_index` columns. Reads the
    canonical Zarr store once (read-only) and extracts one deterministic
    feature vector per row, in `zarr_index` order for efficient sequential
    chunk access.
    """
    import zarr

    root = zarr.open_group(store=zarr.storage.LocalStore(str(zarr_path)), mode="r")
    irwin_k = root["irwin_k"]
    valid_mask = root["valid_mask"]

    ordered = index.sort_values("zarr_index")
    rows = []
    for sample_id, zi in zip(ordered["sample_id"], ordered["zarr_index"]):
        zi = int(zi)
        feats = extract_features(np.asarray(irwin_k[zi]), np.asarray(valid_mask[zi]))
        feats["sample_id"] = sample_id
        rows.append(feats)
    return pd.DataFrame(rows)[["sample_id"] + FEATURE_NAMES]
