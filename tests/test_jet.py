"""jet latitude/speed index (Woollings et al. 2010 on seasonal means)"""

import copy

import numpy as np
import pytest
import xarray as xr

from climate_scenarios_2050.core.config import index_cfg, load_config, validate
from climate_scenarios_2050.preprocess.indices import jet

LAT = np.arange(15.75, 75, 1.5)
LON = np.arange(-59.25, 0, 1.5)
JET_CFG = ["configs/experiments/jet.yaml"]


def wind(centre, peak, width=8.0, lon=LON):
    """(year, lat, lon) zonal wind: a Gaussian jet at latitude `centre` (per year) with maximum `peak`"""
    centre, peak = np.atleast_1d(centre), np.atleast_1d(peak)
    u = peak[:, None] * np.exp(-((LAT[None, :] - centre[:, None]) / width) ** 2)
    u = np.repeat(u[:, :, None], lon.size, axis=2)
    return xr.DataArray(u, dims=("year", "lat", "lon"), coords={"year": np.arange(1900, 1900 + centre.size), "lat": LAT, "lon": lon})


def test_parabola_gives_continuous_latitude():
    centres = np.linspace(40, 50, 41)
    field = wind(centres, np.full(centres.size, 12.0))
    lat_grid, _ = jet.jet_latitude_speed(jet.profile(field), refine="none")
    lat_par, speed = jet.jet_latitude_speed(jet.profile(field), refine="parabola")
    # grid maximum: only grid latitudes; parabola: within a small fraction of the grid spacing
    assert set(np.round(lat_grid.values, 2)) <= set(np.round(LAT, 2))
    assert np.abs(lat_grid.values - centres).max() <= 0.75 + 1e-9
    assert np.abs(lat_par.values - centres).max() < 0.1
    assert len(np.unique(lat_par.values.round(3))) == centres.size
    np.testing.assert_allclose(speed.values, 12.0, rtol=0.01)


def test_jet_at_the_edge_and_missing_data():
    field = wind(np.array([15.75, 74.25, 45.0]), np.array([10.0, 10.0, 10.0]), width=3.0)
    field[2] = np.nan   # a year without data
    lat, speed = jet.jet_latitude_speed(jet.profile(field))
    assert lat.values[0] == pytest.approx(15.75) and lat.values[1] == pytest.approx(74.25)   # edge: grid value
    assert np.isnan(lat.values[2]) and np.isnan(speed.values[2])


def test_anomaly_index_removes_the_own_climatology():
    rng = np.random.default_rng(0)
    years = 120
    obs = wind(45 + rng.normal(0, 2, years), 10 + rng.normal(0, 1, years))
    stats = jet.reference_stats(obs, "latitude", (1950, 2015))
    # a model whose jet is 4 deg further south with the same variations gives the same index
    shifted = jet.raw_index(obs, "latitude").values - 4
    model = wind(shifted, np.full(years, 10.0)).expand_dims(member=["m1"])
    index, clim = jet.project(model, stats, "latitude", (1950, 2015))
    obs_index, obs_clim = jet.project(obs, stats, "latitude", (1950, 2015))
    assert clim == pytest.approx(obs_clim - 4, abs=0.05)
    np.testing.assert_allclose(index.isel(member=0).values, obs_index.values, atol=0.02)
    assert float(index["raw_climatology"][0]) == pytest.approx(clim)
    assert float(obs_index.sel(year=slice(1950, 2015)).std()) == pytest.approx(1.0, rel=0.01)


def test_jet_config_and_index_calculator(tmp_path):
    from climate_scenarios_2050.preprocess.indices.compute import IndexCalculator, index_output

    cfg = load_config(JET_CFG)
    lat_cfg, speed_cfg = index_cfg(cfg, "jet_lat"), index_cfg(cfg, "jet_speed")
    assert lat_cfg["predictor"]["jet"]["quantity"] == "latitude" and speed_cfg["predictor"]["jet"]["quantity"] == "speed"
    assert index_output(cfg, "jet_lat").hash != index_output(cfg, "jet_speed").hash
    rng = np.random.default_rng(1)
    years = 100
    # fields on a wider domain than the jet sector: the calculator cuts them to 0-60W
    lon = np.arange(-89.25, 40, 1.5)
    era5 = wind(45 + rng.normal(0, 2, years), 10 + rng.normal(0, 1, years), lon=lon).assign_coords(year=np.arange(1940, 1940 + years))
    for c, quantity in [(lat_cfg, "latitude"), (speed_cfg, "speed")]:
        c = copy.deepcopy(c)
        c["paths"]["outputs"]["root"] = str(tmp_path)
        calc = IndexCalculator(c, {"ERA5": era5, "20CR": era5}, index_output(c).prepare())
        idx = calc(era5, "ERA5")
        assert idx.attrs["quantity"] == quantity and idx.attrs["units"] == "1"
        assert float(idx.sel(year=slice(1950, 2015)).std()) == pytest.approx(1.0, rel=0.01)
        assert idx.attrs["raw_climatology"] == pytest.approx(45 if quantity == "latitude" else 10, abs=1)


def test_jet_validation():
    cfg = load_config(JET_CFG)
    bad = copy.deepcopy(cfg)
    bad["predictor"]["indices"]["jet_speed"]["jet"]["quantity"] = "width"
    with pytest.raises(ValueError, match="jet.quantity"):
        validate(bad)


def test_mask_and_missing_points():
    # a dataset with a missing (below-ground) block and one with values there give the same profile with the mask
    full = wind(np.array([45.0, 66.0]), np.array([10.0, 10.0]))
    block = (full["lat"] > 60) & (full["lon"] < -30)
    holes = full.where(~block)
    mask = block.transpose("lat", "lon")
    lat_full, _ = jet.jet_latitude_speed(jet.profile(full, mask))
    lat_holes, _ = jet.jet_latitude_speed(jet.profile(holes, mask))
    np.testing.assert_allclose(lat_full.values, lat_holes.values)
    assert lat_holes.values[1] == pytest.approx(66.0, abs=0.1)
    # without the mask the missing block removes those latitudes (min_valid), the year itself stays valid
    lat_nomask, _ = jet.jet_latitude_speed(jet.profile(holes))
    assert np.isfinite(lat_nomask.values).all() and lat_nomask.values[1] < 60
    # a single missing point does not remove a latitude (min_valid 0.8)
    one = full.copy()
    one[0, 20, 3] = np.nan
    np.testing.assert_allclose(jet.profile(one).isel(year=0).values, jet.profile(full).isel(year=0).values, rtol=0.05)


def test_mask_checksum_enters_the_hash():
    from climate_scenarios_2050.preprocess.indices.compute import index_output
    cfg = load_config(JET_CFG)
    assert cfg["predictor"]["jet"]["mask_checksum"]
    other = copy.deepcopy(cfg)
    other["predictor"]["jet"]["mask_checksum"] = "0" * 16
    assert index_output(cfg, "jet_lat").hash != index_output(other, "jet_lat").hash


def test_ua_value_checks():
    from climate_scenarios_2050.preprocess.cmip6 import check_values
    rng = np.random.default_rng(2)
    da = xr.DataArray(rng.normal(5, 5, (24, 10, 10)), dims=("time", "lat", "lon"))
    da[:, 0, 0] = 0.0            # below ground (INM)
    da[:, 0, 1] = 1e20           # undecoded fill value
    da[3, 5, 5] = 9.7e36         # single garbage point
    out, problem = check_values(da, "ua")
    assert problem is None and out[:, 0, :2].isnull().all() and np.isnan(out[3, 5, 5]) and int(out.isnull().sum()) == 49
    corrupt = da.copy()
    corrupt[7, :5] = 3.4e38      # a block of garbage
    assert check_values(corrupt, "ua")[0] is None and "corrupt" in check_values(corrupt, "ua")[1]
    assert check_values(da, "psl")[0] is da   # other variables unchanged
