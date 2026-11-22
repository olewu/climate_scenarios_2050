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


def fake_archive(tmp_path, models, scenario="ssp370"):
    """archive with one (non-empty) historical and scenario file per member, and a config pointing to it"""
    from climate_scenarios_2050.preprocess import cmip6

    cmip6._discover_cached.cache_clear()
    cmip6._common_cached.cache_clear()
    for model, variants in models.items():
        for variant in variants:
            for exp, period in [("historical", "185001-201412"), (scenario, "201501-210012")]:
                fl = tmp_path / "archive" / exp / model / "psl" / f"psl_Amon_{model}_{exp}_{variant}_gn_{period}.nc"
                fl.parent.mkdir(parents=True, exist_ok=True)
                fl.write_bytes(b"x")
    return {
        "preprocess": {"grid_spacing": 1.5, "fields": {"psl": {"method": "bilinear", "domain": {}, "aggregations": {"DJFM": [12, 1, 2, 3]}}}},
        "cmip6": {"period": [1850, 2100]},
        "paths": {"outputs": {"cmip6_processed": str(tmp_path / "processed")}, "inputs": {"cmip6": str(tmp_path / "archive")}},
    }


def test_incomplete_members_left_out(tmp_path):
    from climate_scenarios_2050.preprocess import cmip6

    fake_archive(tmp_path, {"A": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1"]})
    archive = tmp_path / "archive"
    def move(exp, variant, old, new):
        fl = archive / exp / "A" / "psl" / f"psl_Amon_A_{exp}_{variant}_gn_{old}.nc"
        fl.rename(fl.with_name(f"psl_Amon_A_{exp}_{variant}_gn_{new}.nc"))
    move("ssp370", "r2i1p1f1", "201501-210012", "201501-202012")      # short scenario run
    move("historical", "r3i1p1f1", "185001-201412", "197001-201412")  # late historical start
    move("historical", "r4i1p1f1", "185001-201412", "185001-194912")  # gap 1950-2014 ...
    (archive / "historical" / "A" / "psl" / "psl_Amon_A_historical_r4i1p1f1_gn_195101-201412.nc").write_bytes(b"x")  # ... minus 1951-2014
    move("ssp370", "r5i1p1f1", "201501-210012", "201501-205512")      # ends exactly at the end of the period: kept
    found = cmip6.discover_members(archive, "psl", "ssp370", full_period=(1850, 2055))
    assert list(found["A"]) == ["r1i1p1f1", "r5i1p1f1"]
    assert len(cmip6.discover_members(archive, "psl", "ssp370")["A"]) == 5


def test_common_members_across_variables(tmp_path):
    from climate_scenarios_2050.preprocess import cmip6

    cfg = fake_archive(tmp_path, {"A": ["r1i1p1f1", "r2i1p1f1"], "B": ["r1i1p1f1"]})
    cmip6._common_cached.cache_clear()
    # ua as a subset in its own folder, only for A r1 and B r1
    for model, variant in [("A", "r1i1p1f1"), ("B", "r1i1p1f1")]:
        for exp, period in [("historical", "185001-201412"), ("ssp370", "201501-210012")]:
            fl = tmp_path / "archive" / exp / model / "ua_plev850" / f"ua_Amon_{model}_{exp}_{variant}_gn_{period}_plev850.nc"
            fl.parent.mkdir(parents=True, exist_ok=True)
            fl.write_bytes(b"x")
    cfg["download"] = {"subsets": {"ua": {"tag": "plev850"}}}
    cfg["cmip6"].update(full_period=[1850, 2055], common_members=None)
    assert sorted(cmip6.members(cfg, "psl", "ssp370")["A"]) == ["r1i1p1f1", "r2i1p1f1"]
    cfg["cmip6"]["common_members"] = ["psl", "ua"]
    assert sorted(cmip6.members(cfg, "psl", "ssp370")["A"]) == ["r1i1p1f1"]
    assert sorted(cmip6.members(cfg, "ua", "ssp370")) == ["A", "B"]
    # the member list enters the hash
    before = cmip6.members_hash(cfg, "psl", "ssp370")
    cfg["cmip6"]["common_members"] = None
    assert cmip6.members_hash(cfg, "psl", "ssp370") != before


def test_member_list_enters_hash(tmp_path):
    from climate_scenarios_2050.preprocess import cmip6

    cfg = fake_archive(tmp_path, {"A": ["r1i1p1f1"], "B": ["r1i1p1f1"]})
    before = cmip6.field_id(cfg, "psl", "DJFM", "ssp370")
    keys = {m: cmip6.model_key(v) for m, v in cmip6.members(cfg, "psl", "ssp370").items()}
    assert cmip6.field_id(cfg, "psl", "DJFM", "ssp245") != before   # other scenario, other members

    fake_archive(tmp_path, {"A": ["r2i1p1f1"]})   # new member of A
    after = cmip6.field_id(cfg, "psl", "DJFM", "ssp370")
    keys_after = {m: cmip6.model_key(v) for m, v in cmip6.members(cfg, "psl", "ssp370").items()}
    assert after != before
    assert keys_after["B"] == keys["B"] and keys_after["A"] != keys["A"]   # only A is regridded again


def test_open_field_keeps_full_model_names(tmp_path):
    from climate_scenarios_2050.preprocess import cmip6

    cfg = fake_archive(tmp_path, {"A-1": ["r1i1p1f1"], "LONGER-MODEL-NAME": ["r1i1p1f1"]})
    store, _ = cmip6.field_store(cfg, "psl", "DJFM")
    found = cmip6.members(cfg, "psl", "ssp370")
    for model, mid, extra in [("A-1", "m0", {}), ("LONGER-MODEL-NAME", "m12", {"height": 10.0})]:
        ds = xr.Dataset(
            {"psl": (("member", "year"), np.ones((1, 2), dtype="float32"))},
            coords={"member": [f"{mid}r1"], "year": [2000, 2001], "model": ("member", [model]), "model_id": ("member", [mid]), **extra},
        )
        ds.to_zarr(cmip6.model_store(store, "ssp370", model, found[model]))
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
