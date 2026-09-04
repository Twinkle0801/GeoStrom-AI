"""ADT-HURSAT Scene loading and joining -- optional, non-blocking, never
a substitute for IBTrACS intensity ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ml.geostrom_ml.satellite.adt import join_adt_scene, load_adt_storm, parse_adt_time


@pytest.fixture
def adt_nc_dir(tmp_path):
    ds = xr.Dataset(data_vars={
        "Date": ("record", np.array(["2001AUG01", "2001AUG01"])),
        "Time": ("record", np.array(["000000", "060000"])),
        "Scene": ("record", np.array(["Eye", "Curved Band"])),
        "EyeScene": ("record", np.array([0, 3])),
        "CloudScene": ("record", np.array([0, 3])),
        "CI": ("record", np.array([4.5, 3.0])),
        "Lat": ("record", np.array([15.0, 15.15])),
        "Lon": ("record", np.array([-40.0, -40.2])),
        "VZA": ("record", np.array([20.0, 22.0])),
    })
    ds.to_netcdf(tmp_path / "2001213N15040.nc")
    return tmp_path


class TestParseAdtTime:
    def test_parses_date_and_hhmmss(self):
        t = parse_adt_time("2005AUG26", "174513")
        assert t == pd.Timestamp("2005-08-26T17:45:13")


class TestLoadAdtStorm:
    def test_missing_storm_returns_none(self, adt_nc_dir):
        assert load_adt_storm("9999999N99999", adt_nc_dir) is None

    def test_present_storm_loads_sorted_records(self, adt_nc_dir):
        df = load_adt_storm("2001213N15040", adt_nc_dir)
        assert len(df) == 2
        assert list(df["Scene"]) == ["Eye", "Curved Band"]
        assert df["adt_time"].is_monotonic_increasing


class TestJoinAdtScene:
    def test_matched_sample_gets_scene_label(self, adt_nc_dir):
        joined = pd.DataFrame({
            "storm_id": ["2001213N15040"],
            "satellite_timestamp": [pd.Timestamp("2001-08-01T00:05:00")],
        })
        out = join_adt_scene(joined, adt_nc_dir, tolerance_min=90)
        assert out.iloc[0]["scene_label"] == "Eye"
        assert out.iloc[0]["adt_qc_status"] == "matched"

    def test_storm_without_adt_file_is_unmatched_not_an_error(self, adt_nc_dir):
        joined = pd.DataFrame({
            "storm_id": ["9999999N99999"],
            "satellite_timestamp": [pd.Timestamp("2001-08-01T00:00:00")],
        })
        out = join_adt_scene(joined, adt_nc_dir, tolerance_min=90)
        assert pd.isna(out.iloc[0]["scene_label"])
        assert out.iloc[0]["adt_qc_status"] == "unmatched"

    def test_never_writes_an_intensity_field(self, adt_nc_dir):
        """CRITICAL: ADT-HURSAT must never replace IBTrACS intensity ground
        truth -- this function must only ever add scene_label / adt_timestamp
        / adt_qc_status, never usa_wind or pressure_if_valid."""
        joined = pd.DataFrame({
            "storm_id": ["2001213N15040"],
            "satellite_timestamp": [pd.Timestamp("2001-08-01T00:00:00")],
            "usa_wind": [55.0],
        })
        out = join_adt_scene(joined, adt_nc_dir, tolerance_min=90)
        assert out.iloc[0]["usa_wind"] == 55.0  # untouched by the ADT join
        assert "pressure_if_valid" not in out.columns or pd.isna(out.get("pressure_if_valid", pd.Series([None])).iloc[0])

    def test_offset_beyond_tolerance_is_unmatched(self, adt_nc_dir):
        joined = pd.DataFrame({
            "storm_id": ["2001213N15040"],
            "satellite_timestamp": [pd.Timestamp("2001-08-01T12:00:00")],  # far from either ADT record
        })
        out = join_adt_scene(joined, adt_nc_dir, tolerance_min=90)
        assert out.iloc[0]["adt_qc_status"] == "unmatched"
