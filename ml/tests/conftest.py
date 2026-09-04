"""Shared pytest fixtures: small synthetic storms for fast, deterministic tests.

Synthetic data is used (not the real IBTrACS sample) so these tests run in
milliseconds, have exactly known correct answers, and do not depend on
network access or the ~130 MB of Phase 1 verification data being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def synthetic_storm_df() -> pd.DataFrame:
    """One synthetic storm: 16 synoptic steps (96h), moving NW, intensifying
    then weakening. Enough steps for two L=8/H=4 windows.
    """
    n = 16
    times = pd.date_range("2001-08-01T00:00:00", periods=n, freq="6h")
    lat = 15.0 + 0.15 * np.arange(n)
    lon = -40.0 - 0.20 * np.arange(n)
    wind = 30.0 + 3.0 * np.arange(n) - 0.15 * np.arange(n) ** 2  # rises then falls
    pres = 1005.0 - 0.5 * wind

    df = pd.DataFrame({
        "SID": "2001213N15040", "SEASON": 2001, "NUMBER": 5,
        "BASIN": "NA", "SUBBASIN": "MM", "NAME": "TESTSTORM",
        "ISO_TIME": times, "NATURE": "TS",
        "LAT": lat, "LON": lon,
        "USA_WIND": wind, "USA_PRES": pres,
        "USA_SSHS": 0, "USA_STATUS": "TS",
        "TRACK_TYPE": "main", "IFLAG": "O_____________",
        "STORM_SPEED": 10.0, "STORM_DIR": 315.0,
        "DIST2LAND": 500.0, "LANDFALL": 999.0,
    })
    return df


@pytest.fixture
def two_synthetic_storms(synthetic_storm_df) -> pd.DataFrame:
    """Two independent storms in different seasons, for split/leakage tests."""
    s1 = synthetic_storm_df.copy()

    n = 14
    times = pd.date_range("2010-09-05T00:00:00", periods=n, freq="6h")
    s2 = pd.DataFrame({
        "SID": "2010248N20300", "SEASON": 2010, "NUMBER": 9,
        "BASIN": "NA", "SUBBASIN": "MM", "NAME": "OTHERSTORM",
        "ISO_TIME": times, "NATURE": "TS",
        "LAT": 20.0 + 0.1 * np.arange(n), "LON": -60.0 + 0.25 * np.arange(n),
        "USA_WIND": 40.0 + 2.0 * np.arange(n), "USA_PRES": 990.0 - np.arange(n),
        "USA_SSHS": 0, "USA_STATUS": "TS",
        "TRACK_TYPE": "main", "IFLAG": "O_____________",
        "STORM_SPEED": 12.0, "STORM_DIR": 45.0,
        "DIST2LAND": 300.0, "LANDFALL": 999.0,
    })
    return pd.concat([s1, s2], ignore_index=True)


@pytest.fixture
def storm_with_gap(synthetic_storm_df) -> pd.DataFrame:
    """A storm with a missing synoptic observation (12h gap instead of 6h)."""
    df = synthetic_storm_df.copy()
    return df.drop(index=6).reset_index(drop=True)  # remove one mid-sequence row
