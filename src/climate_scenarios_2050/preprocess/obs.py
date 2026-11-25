"""
Observational data.

- Predictor fields: the datasets of `predictor.obs_datasets`, processed exactly like the CMIP6 fields
  (complete seasons, day-weighted means on the native grid, regridded to the common grid and domain),
  optionally with the global mean (`<variable>_global_mean`). How a dataset is read is described in
  `obs_sources` (variable name in the files, conversion to CMIP6 units).
- Prediction targets (`targets`): complete calendar-year means of monthly data, cut to the region on
  their own grid, dims (year, latitude, longitude). Source: a prepared file (e.g. the GWA-scaled wind) or
  a dataset of `obs_sources` (e.g. ERA5 2 m temperature).
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import field_variable, target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import seasonal_mean
from climate_scenarios_2050.preprocess.cmip6 import cut_native, get_regridder, global_mean, global_mean_mask, grid_key, standardize_coords, target_grid

log = logging.getLogger(__name__)

CMIP6_UNITS = {"psl": "Pa", "tas": "K", "pr": "kg m-2 s-1", "sfcWind": "m s-1", "tos": "degC", "ua": "m s-1"}


def _prep_era5(ds):
    """
    recent CDS files carry `valid_time`, `expver` and `number`; a few (e.g. 2m_temperature 2024-09/10)
    have no time dimension but a scalar `date` (YYYYMMDD)
    """
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.dims and "date" in ds.coords and ds["date"].size == 1:
        day = str(int(ds["date"].values))
        ds = ds.drop_vars("date").expand_dims(time=[np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:8]}", "ns")])
    return ds.drop_vars(["expver", "number", "date"], errors="ignore")


def source_variable(cfg, dataset: str, variable: str) -> dict:
    try:
        return cfg["obs_sources"][dataset]["variables"][variable]
    except KeyError:
        raise KeyError(f"obs_sources.{dataset} does not provide {variable}") from None


def era5_files(cfg, variable, dataset="ERA5"):
    folder_name = source_variable(cfg, dataset, variable)["folder"]
    folder = Path(cfg["paths"]["inputs"]["era5_monthly"]) / folder_name
    return sorted(fl for fl in folder.glob(f"{folder_name}_*.nc") if fl.stem.split("_")[-1].isdigit())


def open_obs_dataset(cfg, dataset: str, variable: str, files=None) -> xr.Dataset:
    """monthly data of one variable as a (lazy) Dataset with standardized lat/lon, in CMIP6 units"""
    spec = source_variable(cfg, dataset, variable)
    kind = cfg["obs_sources"][dataset]["kind"]
    if kind == "era5_monthly":
        files = files or era5_files(cfg, variable, dataset)
        # only the variable itself: some files carry extra variables (e.g. `date`) that would conflict
        ds = xr.open_mfdataset(files, preprocess=lambda d: _prep_era5(d)[[spec["name"]]].reset_coords(drop=True), combine="nested", concat_dim="time", parallel=False)
    elif kind == "file":
        # the file of the variable (`path` of the variable, else of the dataset; keys of paths.inputs)
        path = spec.get("path") or cfg["obs_sources"][dataset]["path"]
        ds = xr.open_dataset(cfg["paths"]["inputs"][path], chunks={"time": 120})
    else:
        raise ValueError(f"obs_sources.{dataset}: unknown kind {kind}")
    ds = ds[[spec["name"]]]
    if "level" in spec:
        # one pressure level of a multi-level file (e.g. 20CR uwnd), in the units of the file's level coordinate
        dim = spec.get("level_dim", "level")
        ds = ds.sel({dim: spec["level"]}).drop_vars(dim)
    ds = standardize_coords(ds)
    da = ds[spec["name"]] * spec.get("factor", 1.0) + spec.get("offset", 0.0)
    da.attrs = {"units": CMIP6_UNITS.get(variable, ""), "source": dataset}
    return da.rename(variable).to_dataset()


def open_obs_monthly(cfg, dataset: str, variable: str, domain: dict, files=None, with_global_mean=False):
    """
    monthly field cut to the domain (plus margin); with `with_global_mean` (true or "ocean") also the
    monthly global mean (over the ocean only for "ocean")
    """
    ds = open_obs_dataset(cfg, dataset, variable, files)
    da = cut_native(ds, domain)[variable].load()
    if with_global_mean:
        native = ds[variable]
        return da, global_mean(native, global_mean_mask(cfg, with_global_mean, native)).load()
    return da


#---------------predictor fields--------------#
def obs_field_output(cfg, field: str) -> StepOutput:
    fcfg = cfg["preprocess"]["fields"][field]
    variable = field_variable(cfg, field)
    sources = {ds: cfg["obs_sources"][ds]["variables"][variable] for ds in cfg["predictor"]["obs_datasets"] if variable in cfg["obs_sources"].get(ds, {}).get("variables", {})}
    extra = {"variable": variable, "field": fcfg, "grid_spacing": cfg["preprocess"]["grid_spacing"], "sources": sources}
    if field != variable:
        extra["name"] = field
    return StepOutput(cfg, "obs_fields", sections=[], extra=extra).prepare()


def process_obs_field(cfg, dataset: str, field: str, overwrite=False):
    """seasonal/annual means of one observational dataset on the common grid"""
    fcfg = cfg["preprocess"]["fields"][field]
    variable = field_variable(cfg, field)
    out = obs_field_output(cfg, field)
    todo = [agg for agg in fcfg["aggregations"] if overwrite or not out.exists(f"{dataset}_{field}_{agg}.nc")]
    if not todo:
        log.info(f"{dataset} {field}: all aggregations exist, skipping")
        return out
    with_gm = fcfg.get("global_mean") or False
    tgt = target_grid(cfg["preprocess"]["grid_spacing"], fcfg["domain"])
    monthly = open_obs_monthly(cfg, dataset, variable, fcfg["domain"], with_global_mean=with_gm)
    monthly, gm = monthly if with_gm else (monthly, None)
    # per dataset: regridding method (e.g. nearest_s2d for a coarse grid with gaps, where bilinear would spread
    # every gap to the neighbouring points) and partial seasons of anomaly data (`min_months`)
    spec = source_variable(cfg, dataset, variable)
    method = spec.get("regrid", fcfg["method"])
    weights_file = out.dir / "regrid_weights" / f"{dataset}_{grid_key(monthly)}_{method}.nc"
    regridder = get_regridder(monthly, tgt, method, weights_file)
    for agg in todo:
        months = fcfg["aggregations"][agg]
        fl = out.path(f"{dataset}_{field}_{agg}.nc")
        seas = regridder(seasonal_mean(monthly, months, min_months=spec.get("min_months")), keep_attrs=True).astype("float32")
        seas.attrs.update({"dataset": dataset, "aggregation": agg, "months": str(months)})
        ds = seas.to_dataset(name=variable)
        if gm is not None:
            ds[f"{variable}_global_mean"] = seasonal_mean(gm, months, min_months=spec.get("min_months")).astype("float32").reindex(year=ds.year)
        write_netcdf(ds, fl, cfg, overwrite=overwrite)
        log.info(f"wrote {fl} ({int(seas.year.min())}-{int(seas.year.max())})")
    return out


def open_obs_field(cfg, dataset: str, field: str, aggregation: str, with_global_mean=False):
    out = obs_field_output(cfg, field)
    variable = field_variable(cfg, field)
    with xr.open_dataset(out.path(f"{dataset}_{field}_{aggregation}.nc")) as ds:
        if with_global_mean:
            return ds[variable].load(), ds[f"{variable}_global_mean"].load()
        return ds[variable].load()


#---------------prediction targets--------------#
def target_output(cfg, target_variable: str) -> StepOutput:
    # `forced` only concerns the split, not the target data
    spec = {k: v for k, v in target_spec(cfg, target_variable).items() if k != "forced"}
    extra = {"target_variable": target_variable, "spec": spec, "domain": cfg["region"]["domain"]}
    if spec["source"] == "file":
        extra["file"] = cfg["paths"]["inputs"]["targets"][spec["file"]]
    else:
        extra["source"] = cfg["obs_sources"][spec["source"]]["variables"][spec["variable"]]
    return StepOutput(cfg, "obs_target", sections=[], extra=extra).prepare()


def _open_target_monthly(cfg, target_variable: str) -> xr.DataArray:
    """monthly target data cut to the region (lat/lon), native grid"""
    spec = target_spec(cfg, target_variable)
    region = cfg["region"]["domain"]
    if spec["source"] == "file":
        with xr.open_dataset(cfg["paths"]["inputs"]["targets"][spec["file"]]) as ds:
            ds = standardize_coords(ds[[spec["variable"]]])
            da = dom.select(ds, region)[spec["variable"]].load()
        return da.rename(target_variable)
    ds = open_obs_dataset(cfg, spec["source"], spec["variable"])
    return dom.select(ds, region)[spec["variable"]].load().rename(target_variable)


def process_obs_target(cfg, target_variable: str, overwrite=False):
    """complete calendar-year means (day-weighted) of the observed monthly target data in the region"""
    out = target_output(cfg, target_variable)
    fl = out.path(f"{target_variable}_annual.nc")
    if fl.exists() and not overwrite:
        log.info(f"{fl} exists, skipping")
        return out
    monthly = _open_target_monthly(cfg, target_variable)
    annual = seasonal_mean(monthly, list(range(1, 13)))
    annual.attrs.update(monthly.attrs)
    # downstream code works on latitude/longitude
    annual = annual.rename(lat="latitude", lon="longitude").transpose("year", "latitude", "longitude")
    write_netcdf(annual.to_dataset(name=target_variable), fl, cfg, overwrite=overwrite)
    log.info(f"wrote {fl} ({int(annual.year.min())}-{int(annual.year.max())})")
    return out


def open_obs_target(cfg, target_variable: str) -> xr.DataArray:
    """annual means of the observed target; computed first if the input is new"""
    out = target_output(cfg, target_variable)
    if not out.exists(f"{target_variable}_annual.nc"):
        log.info(f"no annual means of {target_variable} yet, computing them")
        process_obs_target(cfg, target_variable)
    with xr.open_dataset(out.path(f"{target_variable}_annual.nc")) as ds:
        return ds[target_variable].load()
