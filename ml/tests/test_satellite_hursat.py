"""HURSAT-B1 NetCDF parsing, IRWIN extraction, and physical QC."""

from __future__ import annotations

import numpy as np
import pytest

from ml.geostrom_ml.satellite.hursat import (
    EXPECTED_GRID,
    FRAME_RE,
    FrameOpenError,
    discover_frame_files,
    inventory_frames,
    parse_frame_metadata,
    read_irwin,
)


class TestFilenameParsing:
    def test_wellformed_filename_matches(self):
        m = FRAME_RE.match(
            "2005236N23285.KATRINA.2005.08.23.1800.27.GOE-12.026.hursat-b1.v06.nc")
        assert m is not None
        assert m.group("sid") == "2005236N23285"

    def test_malformed_filename_does_not_match(self):
        assert FRAME_RE.match("not_a_hursat_file.nc") is None


class TestParseFrameMetadata:
    def test_extracts_identity_time_and_position(self, synthetic_hursat_nc):
        path = synthetic_hursat_nc()
        rec = parse_frame_metadata(path)
        assert rec["storm_id"] == "2001213N15040"
        assert rec["satellite_id"] == "GOE-8"
        assert rec["vza"] == pytest.approx(25.0)
        assert rec["satellite_lat"] == pytest.approx(15.0)
        assert rec["satellite_lon"] == pytest.approx(-40.0)
        assert "error" not in rec

    def test_irwin_valid_pct_is_100_for_a_clean_frame(self, synthetic_hursat_nc):
        rec = parse_frame_metadata(synthetic_hursat_nc(irwin_fill_frac=0.0))
        assert rec["irwin_valid_pct"] == pytest.approx(100.0)

    def test_irwin_valid_pct_drops_with_fill_values(self, synthetic_hursat_nc):
        rec = parse_frame_metadata(synthetic_hursat_nc(irwin_fill_frac=0.5))
        assert 45.0 < rec["irwin_valid_pct"] < 55.0

    def test_open_error_on_a_corrupt_file_is_recorded_not_raised(self, tmp_path):
        bad = tmp_path / "corrupt.hursat-b1.v06.nc"
        bad.write_bytes(b"not a netcdf file")
        rec = parse_frame_metadata(bad)
        assert "error" in rec and rec["error"]


class TestIrwinPhysicalQC:
    """The IRWIN validity mask must use the <150K / >350K physical floor,
    not fill-value (-1.0) equality alone -- per the Phase 4 task's explicit
    instruction to reuse the Phase 1 threshold verbatim."""

    def test_fill_value_pixels_are_masked_invalid(self, synthetic_hursat_nc):
        path = synthetic_hursat_nc(irwin_fill_frac=0.3)
        kelvin, valid = read_irwin(path)
        assert valid.sum() == pytest.approx(kelvin.size * 0.7, abs=5)
        assert np.isnan(kelvin[~valid]).all()

    def test_a_physically_implausible_but_nonfill_value_is_also_masked(self, tmp_path):
        """A value like 500 K is neither NaN nor the -1.0 fill sentinel, but
        is still physically impossible for a brightness temperature -- the
        <150/>350 floor must catch it where fill-equality alone would not."""
        import xarray as xr

        irwin = np.full((301, 301), 260.0, dtype="float32")
        irwin[0, 0] = 500.0  # implausible, not the fill sentinel
        ds = xr.Dataset(
            data_vars={"IRWIN": (("lat", "lon"), irwin),
                       "htime": ("time", np.array([np.datetime64("2001-08-01T00:00:00")]))},
            attrs={"TC_serial_number": "2001213N15040", "Satellite_Name": "GOE-8"},
        )
        path = tmp_path / "implausible.hursat-b1.v06.nc"
        ds.to_netcdf(path)
        kelvin, valid = read_irwin(path)
        assert not valid[0, 0]
        assert np.isnan(kelvin[0, 0])

    def test_valid_pixels_retain_exact_physical_value(self, synthetic_hursat_nc):
        kelvin, valid = read_irwin(synthetic_hursat_nc())
        assert np.all(kelvin[valid] == pytest.approx(260.0))


class TestGridShape:
    def test_expected_grid_is_301x301(self):
        assert EXPECTED_GRID == (301, 301)

    def test_wrong_grid_shape_raises_frame_open_error(self, tmp_path):
        import xarray as xr

        ds = xr.Dataset(
            data_vars={"IRWIN": (("lat", "lon"), np.full((10, 10), 260.0, dtype="float32")),
                       "htime": ("time", np.array([np.datetime64("2001-08-01T00:00:00")]))},
            attrs={"TC_serial_number": "X", "Satellite_Name": "Y"},
        )
        path = tmp_path / "wrong_shape.hursat-b1.v06.nc"
        ds.to_netcdf(path)
        with pytest.raises(FrameOpenError):
            read_irwin(path)


class TestDiscovery:
    def test_discovers_only_hursat_nc_files(self, tmp_path, synthetic_hursat_nc):
        synthetic_hursat_nc(filename="a.hursat-b1.v06.nc")
        (tmp_path / "not_hursat.nc").write_text("x")
        found = discover_frame_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "a.hursat-b1.v06.nc"

    def test_inventory_frames_on_empty_list_returns_empty_frame(self):
        df = inventory_frames([])
        assert len(df) == 0
