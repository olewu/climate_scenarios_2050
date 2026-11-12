import numpy as np
import pandas as pd
import xarray as xr
import cftime

from climate_scenarios_2050.core.timeconv import seasonal_mean, complete_seasons, window_means, hindcast_window_means, change


def monthly(start="1949-12", end="1952-03", calendar="standard"):
    time = xr.cftime_range(start, end, freq="MS", calendar=calendar)
    vals = np.arange(len(time), dtype=float)
    return xr.DataArray(vals, dims="time", coords={"time": time})


def test_djfm_labels_and_completeness():
    da = monthly()
    seas = seasonal_mean(da, [12, 1, 2, 3], weighted=False)
    # seasons 1949, 1950, 1951 are complete (Dec 1951 - Mar 1952 included)
    assert list(seas.year.values) == [1949, 1950, 1951]
    # DJFM 1949 = Dec 1949 (0), Jan-Mar 1950 (1, 2, 3)
    assert seas.sel(year=1949) == 1.5


def test_partial_season_dropped():
    da = monthly("1950-01", "1951-02")
    seas = seasonal_mean(da, [12, 1, 2, 3])
    assert list(seas.year.values) == []  # Jan-Mar 1950 lacks Dec 1949, 1950 lacks Mar 1951


def test_day_weighting_noleap():
    da = monthly("1950-12", "1951-03", calendar="noleap")
    seas = seasonal_mean(da, [12, 1, 2, 3])
    w = np.array([31, 31, 28, 31])
    assert np.isclose(seas.values[0], (np.arange(4) * w).sum() / w.sum())


def test_complete_seasons():
    time = pd.date_range("1940-01", "2026-09", freq="MS")
    yrs = complete_seasons(time, [12, 1, 2, 3])
    assert yrs[0] == 1940 and yrs[-1] == 2025


def test_window_means_lag():
    years = np.arange(1900, 2001)
    s = xr.DataArray(years.astype(float), dims="year", coords={"year": years})
    wm = window_means(s, [(1, 5)], lag=0, reference_length=20)
    # init 1950: window 1951-1955 -> 1953, reference 1931-1950 -> 1940.5
    assert wm.window_mean.sel(init_year=1950, window="1-5") == 1953
    assert wm.reference_mean.sel(init_year=1950) == 1940.5
    wm1 = window_means(s, [(1, 5)], lag=1, reference_length=20)
    # lag 1: seasons 1950-1954 -> 1952, reference seasons 1930-1949 -> 1939.5
    assert wm1.window_mean.sel(init_year=1950, window="1-5") == 1952
    assert wm1.reference_mean.sel(init_year=1950) == 1939.5


def test_hindcast_window_means_matches_obs():
    years = np.arange(1900, 2001)
    s = xr.DataArray(years.astype(float), dims="year", coords={"year": years})
    inits = np.array([1950, 1960])
    hc = s.expand_dims(init_year=inits).copy()
    hm = hindcast_window_means(hc, [(1, 5), (11, 30)], lag=1)
    om = window_means(s, [(1, 5), (11, 30)], lag=1)
    xr.testing.assert_allclose(hm, om.window_mean.sel(init_year=inits).transpose(*hm.dims))


def test_change():
    assert change(11.0, 10.0) == 10.0
    assert change(11.0, 10.0, "absolute") == 1.0


def test_join_time_rounding_differences_do_not_duplicate_rows():
    from climate_scenarios_2050.preprocess.cmip6 import _join_time

    def part(start, offset):
        time = xr.cftime_range(start, periods=2, freq="MS", calendar="noleap")
        return xr.Dataset({"v": (("time", "lat", "lon"), np.ones((2, 3, 2)))}, coords={"time": time, "lat": np.array([1.0, 2.0, 3.0]) + offset, "lon": [0.0, 1.0]})

    joined = _join_time([part("2014-11", 0.0), part("2015-01", 1e-6)])
    assert joined.sizes["lat"] == 3 and joined["v"].notnull().all()


def test_open_field_keeps_full_model_names(tmp_path):
    from climate_scenarios_2050.preprocess import cmip6

    cfg = {
        "preprocess": {"grid_spacing": 1.5, "fields": {"psl": {"method": "bilinear", "domain": {}, "aggregations": {"DJFM": [12, 1, 2, 3]}}}},
        "cmip6": {"period": [1850, 2100]},
        "paths": {"outputs": {"cmip6_processed": str(tmp_path)}},
    }
    store, _ = cmip6.field_store(cfg, "psl", "DJFM")
    for model, mid, extra in [("A-1", "m0", {}), ("LONGER-MODEL-NAME", "m12", {"height": 10.0})]:
        ds = xr.Dataset(
            {"psl": (("member", "year"), np.ones((1, 2), dtype="float32"))},
            coords={"member": [f"{mid}r1"], "year": [2000, 2001], "model": ("member", [model]), "model_id": ("member", [mid]), **extra},
        )
        ds.to_zarr(store / "ssp370" / f"{model}.zarr")
    field = cmip6.open_field(cfg, "psl", "DJFM", "ssp370")
    assert sorted(field["model_id"].values) == ["m0", "m12"]
    assert "LONGER-MODEL-NAME" in field["model"].values and "height" not in field.coords
    # the names also survive writing the joined data (with and without the safe writer)
    field.to_netcdf(tmp_path / "direct.nc")
    assert sorted(xr.open_dataset(tmp_path / "direct.nc")["model_id"].values) == ["m0", "m12"]


def test_write_netcdf_ignores_inherited_string_width(tmp_path):
    from climate_scenarios_2050.core.io import write_netcdf

    cfg = {"paths": {"outputs": {"root": str(tmp_path)}}}
    ds = xr.Dataset({"v": ("member", [1.0, 2.0])}, coords={"member": ["a", "b"], "model": ("member", ["short", "much-longer-name"])})
    ds["model"].encoding["dtype"] = np.dtype("<U5")
    write_netcdf(ds, tmp_path / "out.nc", cfg, compress=True)
    assert list(xr.open_dataset(tmp_path / "out.nc")["model"].values) == ["short", "much-longer-name"]
