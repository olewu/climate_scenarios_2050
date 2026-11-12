"""
Observational/reanalysis fields, processed exactly like the CMIP6 fields (complete seasons,
day-weighted means on the native grid, regridded to the common grid and domain), and the
observed (ERA5-based) wind data as complete calendar-year means.
"""

import logging
from pathlib import Path

import xarray as xr

from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import seasonal_mean
from climate_scenarios_2050.preprocess.cmip6 import cut_native, get_regridder, grid_key, standardize_coords, target_grid

log = logging.getLogger(__name__)

# dataset -> variable -> (variable name in file, factor to CMIP6 units)
OBS_VARIABLES = {
    "ERA5": {"psl": ("msl", 1.0)},
    "HadSLP": {"psl": ("slp", 100.0)},   # hPa -> Pa
    "20CR": {"psl": ("prmsl", 1.0)},
}
ERA5_FOLDERS = {"psl": "mean_sea_level_pressure"}


def _prep_era5(ds):
    """recent CDS files carry `valid_time`, `expver` and `number`"""
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars(["expver", "number"], errors="ignore")


def era5_files(cfg, variable):
    folder = Path(cfg["paths"]["inputs"]["era5_monthly"]) / ERA5_FOLDERS[variable]
    return sorted(fl for fl in folder.glob(f"{ERA5_FOLDERS[variable]}_*.nc") if fl.stem.split("_")[-1].isdigit())


def open_obs_monthly(cfg, dataset: str, variable: str, domain: dict, files=None) -> xr.DataArray:
    """monthly field of an observational dataset, cut to the domain (plus margin), in CMIP6 units"""
    name, factor = OBS_VARIABLES[dataset][variable]
    if dataset == "ERA5":
        files = files or era5_files(cfg, variable)
        ds = xr.open_mfdataset(files, preprocess=_prep_era5, combine="nested", concat_dim="time", parallel=False)
    else:
        ds = xr.open_dataset(cfg["paths"]["inputs"][dataset.lower()])
    ds = standardize_coords(ds[[name]])
    da = cut_native(ds, domain)[name].load() * factor
    return da.rename(variable)


def obs_field_output(cfg, variable: str) -> StepOutput:
    fcfg = cfg["preprocess"]["fields"][variable]
    return StepOutput(cfg, "obs_fields", sections=[], extra={"variable": variable, "field": fcfg, "grid_spacing": cfg["preprocess"]["grid_spacing"]}).prepare()


def process_obs_field(cfg, dataset: str, variable: str, overwrite=False):
    """seasonal/annual means of one observational dataset on the common grid"""
    fcfg = cfg["preprocess"]["fields"][variable]
    out = obs_field_output(cfg, variable)
    tgt = target_grid(cfg["preprocess"]["grid_spacing"], fcfg["domain"])
    monthly = open_obs_monthly(cfg, dataset, variable, fcfg["domain"])
    weights_file = out.dir / "regrid_weights" / f"{dataset}_{grid_key(monthly)}_{fcfg['method']}.nc"
    regridder = get_regridder(monthly, tgt, fcfg["method"], weights_file)
    for agg, months in fcfg["aggregations"].items():
        fl = out.path(f"{dataset}_{variable}_{agg}.nc")
        if fl.exists() and not overwrite:
            log.info(f"{fl} exists, skipping")
            continue
        seas = regridder(seasonal_mean(monthly, months), keep_attrs=True).astype("float32")
        seas.attrs.update({"dataset": dataset, "aggregation": agg, "months": str(months)})
        write_netcdf(seas.to_dataset(name=variable), fl, cfg, overwrite=overwrite)
        log.info(f"wrote {fl} ({int(seas.year.min())}-{int(seas.year.max())})")
    return out


def open_obs_field(cfg, dataset: str, variable: str, aggregation: str) -> xr.DataArray:
    out = obs_field_output(cfg, variable)
    with xr.open_dataset(out.path(f"{dataset}_{variable}_{aggregation}.nc")) as ds:
        return ds[variable].load()


#---------------observed wind--------------#
def wind_output(cfg, wind_variable: str) -> StepOutput:
    return StepOutput(cfg, "obs_wind", sections=[], extra={"wind_variable": wind_variable, "file": cfg["paths"]["inputs"]["wind"][wind_variable]}).prepare()


def process_obs_wind(cfg, wind_variable: str, overwrite=False):
    """complete calendar-year means (day-weighted) of the observed monthly wind data"""
    out = wind_output(cfg, wind_variable)
    fl = out.path(f"{wind_variable}_annual.nc")
    if fl.exists() and not overwrite:
        log.info(f"{fl} exists, skipping")
        return out
    with xr.open_dataset(cfg["paths"]["inputs"]["wind"][wind_variable]) as ds:
        monthly = ds[wind_variable].load()
    annual = seasonal_mean(monthly, list(range(1, 13)))
    annual.attrs.update(monthly.attrs)
    write_netcdf(annual.to_dataset(name=wind_variable), fl, cfg, overwrite=overwrite)
    log.info(f"wrote {fl} ({int(annual.year.min())}-{int(annual.year.max())})")
    return out


def open_obs_wind(cfg, wind_variable: str) -> xr.DataArray:
    """annual means of the observed wind; computed first if the input file is new (quick)"""
    out = wind_output(cfg, wind_variable)
    if not out.exists(f"{wind_variable}_annual.nc"):
        log.info(f"no annual means of {cfg['paths']['inputs']['wind'][wind_variable]} yet, computing them")
        process_obs_wind(cfg, wind_variable)
    with xr.open_dataset(out.path(f"{wind_variable}_annual.nc")) as ds:
        return ds[wind_variable].load()
