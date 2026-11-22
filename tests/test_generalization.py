"""region/target/predictor generalization: date-line domains, global mean, land-masked EOFs, targets, hashes"""

import copy

import numpy as np
import pytest
import xarray as xr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import load_config, used_fields, validate
from climate_scenarios_2050.preprocess import cmip6
from climate_scenarios_2050.preprocess.indices import eof

PACIFIC = {"lon": [110, -100], "lat": [20, 70]}


def global_field(n_year=60, spacing=3.0):
    lat = np.arange(-90 + spacing / 2, 90, spacing)
    lon = np.arange(-180 + spacing / 2, 180, spacing)
    return xr.DataArray(np.zeros((n_year, len(lat), len(lon))), dims=("year", "lat", "lon"), coords={"year": np.arange(1950, 1950 + n_year), "lat": lat, "lon": lon})


def test_dateline_domain_selection():
    field = global_field()
    sub = dom.select(field, PACIFIC)
    lon = sub.lon.values
    assert lon.min() >= 110 and lon.max() <= 260 and np.all(np.diff(lon) > 0)
    assert sub.lat.min() >= 20 and sub.lat.max() <= 70
    # a normal domain keeps -180 ... 180
    assert float(dom.select(field, {"lon": [-35, 45], "lat": [25, 75]}).lon.min()) < 0
    grid = cmip6.target_grid(1.5, PACIFIC)
    assert float(grid.lon.min()) == pytest.approx(110.25) and float(grid.lon.max()) == pytest.approx(258.75)
    assert dom.point_lon(-130.0, PACIFIC) == pytest.approx(230.0)
    # curvilinear grid (2-D lat/lon), longitudes 0 ... 360 as in many ocean grids
    lat2, lon2 = np.meshgrid(np.linspace(-80, 80, 20), np.linspace(0, 357, 120), indexing="ij")
    ds = xr.Dataset({"tos": (("j", "i"), np.ones(lat2.shape))}, coords={"lat": (("j", "i"), lat2), "lon": (("j", "i"), lon2)})
    cut = cmip6.cut_native(cmip6.standardize_coords(ds), PACIFIC, margin=0)
    assert float(cut.lon.min()) >= 110 and float(cut.lon.max()) <= 260 and int(cut["tos"].notnull().sum()) > 0


def test_global_mean_ignores_land():
    lat = np.array([-60.0, 0.0, 60.0])
    da = xr.DataArray(np.array([[[1.0, np.nan], [3.0, 3.0], [1.0, 1.0]]]), dims=("time", "lat", "lon"), coords={"time": [0], "lat": lat, "lon": [0.0, 180.0]})
    w = np.cos(np.deg2rad(lat))
    expected = (1 * w[0] + 3 * 2 * w[1] + 1 * 2 * w[2]) / (w[0] + 2 * w[1] + 2 * w[2])
    assert float(cmip6.global_mean(da).squeeze()) == pytest.approx(expected)


def pdo_like(rng, members=None, n_year=60):
    """SST in the North Pacific with land, a dipole mode, a global-mean trend and noise"""
    field = dom.select(global_field(n_year), PACIFIC)
    lat, lon = np.meshgrid(field.lat, field.lon, indexing="ij")
    pattern = np.where(lon > 200, 1.0, -1.0) * np.cos(np.deg2rad(lat))
    land = (lon > 240) & (lat > 40)
    shape = (n_year,) if members is None else (members, n_year)
    signal = rng.standard_normal(shape)
    trend = np.linspace(0, 1, n_year)
    values = signal[..., None, None] * pattern + trend[..., None, None] + 0.3 * rng.standard_normal((*shape, *lat.shape))
    values = np.where(land, np.nan, values)
    dims = ("year", "lat", "lon") if members is None else ("member", "year", "lat", "lon")
    coords = {"year": field.year, "lat": field.lat, "lon": field.lon}
    if members is not None:
        coords["member"] = [f"m{i}" for i in range(members)]
    gm = xr.DataArray(np.broadcast_to(trend, shape), dims=dims[:-2], coords={k: v for k, v in coords.items() if k in dims[:-2]})
    return xr.DataArray(values, dims=dims, coords=coords), gm, signal


def test_eof_with_land_and_global_mean():
    rng = np.random.default_rng(0)
    field, gm, signal = pdo_like(rng)
    anom = field - gm
    normal = (1950, 2009)
    pattern = eof.fit_pattern(anom, normal, n_modes=3, detrend_fit=True, sign_point={"lat": 45.0, "lon": 230.0})
    assert pattern["components"].sel(mode=1).isnull().any()          # land stays NaN
    index = eof.project(anom, pattern, 1, eof.normal_period_climatology(anom, normal))
    assert index.notnull().all()
    assert np.corrcoef(index.values, signal)[0, 1] > 0.95             # positive by the sign point
    flipped = eof.fit_pattern(anom, normal, n_modes=3, detrend_fit=True, sign_point={"lat": 45.0, "lon": 230.0, "sign": "negative"})
    np.testing.assert_allclose(flipped["components"].sel(mode=1), -pattern["components"].sel(mode=1))

    # an ensemble: land does not discard member-years, and the climatology uses the full members
    ens, gm_ens, signal_ens = pdo_like(rng, members=3)
    clim = eof.normal_period_climatology(ens - gm_ens, normal)
    assert clim.notnull().sum() > 0
    own = eof.fit_pattern(ens - gm_ens, normal, n_modes=3, detrend_fit=True)
    assert own.attrs["n_samples"] == 3 * 60
    idx = eof.project(ens - gm_ens, pattern, 1, clim)
    assert idx.notnull().all() and abs(np.corrcoef(idx.values.ravel(), signal_ens.ravel())[0, 1]) > 0.9


def test_config_validation_and_used_fields():
    cfg = load_config()
    assert used_fields(cfg) == ["psl", "sfcWind", "tas", "pr"]
    assert cfg["preprocess"]["fields"]["tas"]["domain"] == cfg["region"]["domain"]
    bad = copy.deepcopy(cfg)
    bad["region"]["targets"] = ["wsp100", "unknown"]
    with pytest.raises(ValueError, match="unknown"):
        validate(bad)
    bad = copy.deepcopy(cfg)
    bad["obs"]["matching_index"] = "DCENT"
    with pytest.raises(ValueError, match="matching_index"):
        validate(bad)
    sa = load_config(["configs/experiments/south_america_pdo.yaml"])
    assert used_fields(sa) == ["tos", "tas", "pr", "sfcWind"] and sa["obs"]["merge"] is None


def write_era5(root, folder, name, years, value):
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    lat, lon = np.arange(90, -90.1, -10.0), np.arange(0, 360, 10.0)   # ERA5 layout: descending lat, 0 ... 360
    for year in years:
        for month in range(1, 13):
            time = np.array([np.datetime64(f"{year}-{month:02d}-01")])
            da = xr.DataArray(np.full((1, len(lat), len(lon)), value), dims=("valid_time", "latitude", "longitude"), coords={"valid_time": time, "latitude": lat, "longitude": lon})
            da.to_dataset(name=name).to_netcdf(path / f"{folder}_{year}_{month:02d}.nc")


def small_cfg(tmp_path):
    cfg = load_config()
    cfg["paths"]["inputs"]["era5_monthly"] = str(tmp_path / "era5")
    cfg["paths"]["inputs"]["cmip6"] = str(tmp_path / "archive")
    cfg["paths"]["outputs"] = {"root": str(tmp_path / "out"), "cmip6_processed": str(tmp_path / "processed"), "figures": str(tmp_path / "fig")}
    return cfg


def test_era5_target_annual_in_region(tmp_path):
    from climate_scenarios_2050.preprocess import obs

    cfg = small_cfg(tmp_path)
    write_era5(tmp_path / "era5", "total_precipitation", "tp", [2000, 2001], 0.002)   # m/day
    annual = obs.open_obs_target(cfg, "pr")
    assert annual.dims == ("year", "latitude", "longitude") and list(annual.year.values) == [2000, 2001]
    assert float(annual.longitude.min()) >= -35 and float(annual.longitude.max()) <= 45
    assert float(annual.latitude.min()) >= 25 and float(annual.latitude.max()) <= 75
    assert float(annual.mean()) == pytest.approx(0.002 * 1000 / 86400)            # kg m-2 s-1


def fake_cmip6(root, variable, model, variants, exps=("historical", "ssp245")):
    from climate_scenarios_2050.preprocess import cmip6 as c

    c._discover_cached.cache_clear()
    c._common_cached.cache_clear()
    for variant in variants:
        for exp, period in zip(exps, ["185001-201412", "201501-210012"]):
            fl = root / exp / model / variable / f"{variable}_Amon_{model}_{exp}_{variant}_gn_{period}.nc"
            fl.parent.mkdir(parents=True, exist_ok=True)
            fl.write_bytes(b"x")


def test_target_downloads_leave_the_predictor_chain_alone(tmp_path):
    from climate_scenarios_2050.methods.target import target_hindcast_output
    from climate_scenarios_2050.split.run import split_output

    cfg = small_cfg(tmp_path)
    cfg["cmip6"]["common_members"] = None   # members per variable (with common members a tas-only member is not used at all)
    archive = tmp_path / "archive"
    for var in ["psl", "tas", "sfcWind"]:
        fake_cmip6(archive, var, "A", ["r1i1p1f1"])
    before = {"split": split_output(cfg, "ssp245").hash, "tas": target_hindcast_output(cfg, "ssp245", "tas").hash, "wsp100": target_hindcast_output(cfg, "ssp245", "wsp100").hash}
    fake_cmip6(archive, "tas", "A", ["r2i1p1f1"])     # more tas members
    after = {"split": split_output(cfg, "ssp245").hash, "tas": target_hindcast_output(cfg, "ssp245", "tas").hash, "wsp100": target_hindcast_output(cfg, "ssp245", "wsp100").hash}
    assert after["split"] == before["split"] and after["wsp100"] == before["wsp100"]
    assert after["tas"] != before["tas"]
    fake_cmip6(archive, "psl", "A", ["r2i1p1f1"])     # more psl members: everything downstream changes
    assert split_output(cfg, "ssp245").hash != before["split"]
    # common members: r2 counts only once it is complete in all listed variables
    cfg["cmip6"]["common_members"] = ["psl", "tas", "sfcWind"]
    common_before = split_output(cfg, "ssp245").hash
    assert common_before == before["split"]
    fake_cmip6(archive, "sfcWind", "A", ["r2i1p1f1"])
    assert split_output(cfg, "ssp245").hash != common_before


def test_sst_preprocessing_with_global_mean_and_dateline(tmp_path):
    """tos files (Omon, 0...360 longitudes, land NaN) -> annual North Pacific store with the global mean"""
    import cftime

    cfg = small_cfg(tmp_path)
    cfg["preprocess"]["grid_spacing"] = 10.0
    model, variant = "AWI-CM-1-1-MR", "r1i1p1f1"
    lat, lon = np.arange(-85, 90, 10.0), np.arange(5, 360, 10.0)
    land = (lon[None, :] > 250) & (lon[None, :] < 300) & (lat[:, None] > 30)
    for exp, years in [("historical", (1850, 2014)), ("ssp245", (2015, 2100))]:
        time = [cftime.DatetimeNoLeap(y, m, 15) for y in range(years[0], years[1] + 1) for m in range(1, 13)]
        values = np.broadcast_to(np.arange(len(time), dtype=float)[:, None, None] * 0.001 + 10, (len(time), len(lat), len(lon))).copy()
        values[:, land] = np.nan
        ds = xr.Dataset({"tos": (("time", "lat", "lon"), values)}, coords={"time": time, "lat": lat, "lon": lon})
        fl = tmp_path / "archive" / exp / model / "tos" / f"tos_Omon_{model}_{exp}_{variant}_gn_{years[0]}01-{years[1]}12.nc"
        fl.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(fl)
    cmip6._discover_cached.cache_clear()
    found = cmip6.members(cfg, "tos", "ssp245")
    assert list(found) == [model]
    store, _ = cmip6.field_store(cfg, "tos", "annual")
    res = cmip6.process_model(cfg, model, found[model], "tos", "ssp245", {"annual": store})
    assert res["status"] == "written"
    out = xr.open_zarr(cmip6.model_store(store, "ssp245", model, found[model]))
    assert float(out.lon.min()) >= 100 and float(out.lon.max()) <= 280 and np.all(np.diff(out.lon.values) > 0)
    gm = out["tos_global_mean"].isel(member=0)
    assert gm.sizes["year"] == 251 and bool(gm.notnull().all())
    assert float(gm.sel(year=2000)) == pytest.approx(10 + 0.001 * (150 * 12 + 5.5), abs=0.01)   # annual mean of the ramp
