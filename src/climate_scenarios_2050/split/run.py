"""
Step 2: forced/internal split, per scenario (`<root>/split/<hash>/`, the scenario is part of the hash).

- `cmip6_index.nc`    internal (member, year), forced (model_id, year), coefficients per model
- `basis.nc`          regressors (term, year) and the MMEM
- `obs_<name>.nc`     coefficients for an observed index, one variable per leakage case:
    `coef_full` (term): all years (leakage option A)
    `coef_up_to_init` (init_year, term): years up to init (matching, forecasts)
    `coef_window_lag0`/`coef_window_lag<L>` (init_year, window, term): target window and reference
    period left out (leakage option C), for verifying the index itself (lag 0) and for pairing it
    with wind targets (lag L = target.predictor_lag)
- `target_<variable>_<hash>.nc` the same for an observed target (`coef_window` only for the hindcast
    scenarios), hash of the processed target input (a new target file gives new coefficients without
    touching the rest). The forced part of the target (`targets.<name>.forced`):
    `index` (default, wind/energy): the predictor's basis, season t-L for calendar year t
    `cmip6`: per grid point the MMEM of the target's own CMIP6 variable in year t, regridded to the target
    grid (stored as `basis_cmip6`); the file name ends in `_cmip6-<field id>` (members and settings of the
    CMIP6 field)

The split of the indices depends only on the predictor; with `forced: cmip6` the target coefficients also
depend on the target's CMIP6 field.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.config import active_targets, index_cfg, index_names, lm_predictors, primary_index, target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices.compute import cmip6_index_output, index_output, obs_index_names, open_cmip6_index, open_obs_index
from climate_scenarios_2050.split import forced as fc

log = logging.getLogger(__name__)


def split_output(cfg, scenario: str, index: str = None) -> StepOutput:
    """split of one scenario and index (default: the primary index)"""
    cfg = index_cfg(cfg, index)
    return StepOutput(
        cfg, "split", sections=["split", "target"],
        extra={"scenario": scenario, "weighting": cfg["cmip6"]["weighting"]},
        upstream={"indices": index_output(cfg).hash, "cmip6_indices": cmip6_index_output(cfg, scenario).hash},
    )


def predictor_split_hashes(cfg, scenario: str):
    """
    upstream record of the steps using the linear-model predictors: the primary split's hash with one
    predictor (as in a single-index configuration), otherwise {index: hash} of the primary and the predictors
    """
    names = lm_predictors(cfg)
    if names == [primary_index(cfg)]:
        return split_output(cfg, scenario).hash
    return {name: split_output(cfg, scenario, name).hash for name in dict.fromkeys([primary_index(cfg), *names])}


FORCED = ("index", "cmip6")


def target_forced(cfg, target_variable: str) -> str:
    """how the forced part of an observed target is estimated: index | cmip6"""
    return target_spec(cfg, target_variable).get("forced", "index")


def target_forced_id(cfg, scenario: str, target_variable: str):
    """identifies the forced estimate of a target for downstream hashes (None for the default `index`)"""
    if target_forced(cfg, target_variable) == "index":
        return None
    spec = target_spec(cfg, target_variable)
    return {"forced": "cmip6", "cmip6_fields": cmip6.field_id(cfg, spec["cmip6"], "annual", scenario)}


def target_file(cfg, target_variable: str, scenario: str) -> str:
    name = f"target_{target_variable}_{obs.target_output(cfg, target_variable).hash}"
    forced = target_forced_id(cfg, scenario, target_variable)
    if forced:
        name += f"_cmip6-{forced['cmip6_fields']}"
    return name + ".nc"


def cmip6_on_target_grid(da: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """CMIP6 field (..., lat, lon) bilinearly on the target grid (latitude, longitude), nearest value at the edges"""
    lat, lon = like["latitude"].values, like["longitude"].values
    linear = da.interp(lat=lat, lon=lon)
    nearest = da.sel(lat=lat, lon=lon, method="nearest").assign_coords(lat=lat, lon=lon)
    out = linear.fillna(nearest).rename(lat="latitude", lon="longitude")
    return out.assign_coords(latitude=like["latitude"], longitude=like["longitude"])


def target_cmip6_basis(cfg, scenario: str, target_variable: str, like: xr.DataArray) -> xr.DataArray:
    """MMEM (year, latitude, longitude) of the target's CMIP6 variable on the target grid"""
    from climate_scenarios_2050.methods.target import cmip6_mmem, open_cmip6_target

    spec = target_spec(cfg, target_variable)
    field = open_cmip6_target(cfg, spec["cmip6"], scenario)
    if field is None:
        raise FileNotFoundError(f"targets.{target_variable}.forced: cmip6 needs the preprocessed {spec['cmip6']} field of {scenario}")
    mmem = cmip6_mmem(field, cfg["cmip6"]["weighting"])
    return cmip6_on_target_grid(mmem, like).astype("float32").rename("basis_cmip6")


def available_target_variables(cfg) -> list:
    """active targets (`region.targets`) whose input exists"""
    out = []
    for name in active_targets(cfg):
        spec = target_spec(cfg, name)
        if spec["source"] == "file":
            path = cfg["paths"]["inputs"]["targets"][spec["file"]]
            if not Path(path).exists():
                log.warning(f"{name}: input file {path} does not exist, skipping")
                continue
        out.append(name)
    return out


def _as_tuples(windows):
    return [tuple(w) for w in windows]


def index_coefficients(series: xr.DataArray, basis: xr.DataArray, cfg) -> xr.Dataset:
    """all leakage cases for an observed index (season-labelled, paired with the basis of the same season)"""
    series = series.dropna("year")
    years = series.year.values
    design = fc.design_matrix(basis, years)
    min_years = cfg["split"]["min_fit_years"]
    windows = _as_tuples(cfg["target"]["windows"])
    ref = cfg["target"]["reference_length"]
    lag = cfg["target"]["predictor_lag"]
    out = {
        "coef_full": fc.fit_coefficients(series, design, fc.full_period_exclusions(years), min_years),
        "coef_up_to_init": fc.fit_coefficients(series, design, fc.up_to_init_exclusions(years, years), min_years),
    }
    for lg in sorted({0, lag}):
        excl = fc.window_exclusions(years, years, windows, lag=lg, reference_length=ref)
        out[f"coef_window_lag{lg}"] = fc.fit_coefficients(series, design, excl, min_years)
    return xr.Dataset(out)


def target_coefficients(wind: xr.DataArray, design: xr.DataArray, cfg, with_windows: bool) -> xr.Dataset:
    """all leakage cases for an observed annual target (calendar years) and its design matrix"""
    years = wind.year.values
    min_years = cfg["split"]["min_fit_years"]
    out = {
        "coef_full": fc.fit_coefficients(wind, design, fc.full_period_exclusions(years), min_years),
        "coef_up_to_init": fc.fit_coefficients(wind, design, fc.up_to_init_exclusions(years, years), min_years),
    }
    if with_windows:
        excl = fc.window_exclusions(years, years, _as_tuples(cfg["target"]["windows"]), lag=0, reference_length=cfg["target"]["reference_length"])
        out["coef_window"] = fc.fit_coefficients(wind, design, excl, min_years)
    return xr.Dataset(out)


def ensure_target_coefficients(cfg, scenario: str, out, basis, target_variable: str):
    """forced-fit coefficients of the observed target; computed if the target input is new (a few minutes)"""
    fl = out.path(target_file(cfg, target_variable, scenario))
    if not fl.exists():
        forced = target_forced(cfg, target_variable)
        log.info(f"split {scenario}: forced-fit coefficients ({forced}) for the current {target_variable} input -> {fl.name}")
        with_windows = scenario in cfg["cmip6"]["scenarios_hindcast"]
        wind = obs.open_obs_target(cfg, target_variable)
        if forced == "cmip6":
            field = target_cmip6_basis(cfg, scenario, target_variable, wind)
            coefs = target_coefficients(wind, fc.pointwise_design(field, wind.year.values), cfg, with_windows)
            coefs["basis_cmip6"] = field
        else:
            coefs = target_coefficients(wind, fc.design_matrix(basis, wind.year.values, lag=cfg["target"]["predictor_lag"]), cfg, with_windows)
        write_netcdf(coefs, fl, cfg, compress=True)
    return fl


def run_scenario(cfg, scenario: str, index: str = None):
    """split of one index; the observed targets are split with the primary index's basis only"""
    with_targets = (index or primary_index(cfg)) == primary_index(cfg)
    cfg = index_cfg(cfg, index)
    out = split_output(cfg, scenario).prepare()
    log.info(f"split {cfg['predictor']['name']} {scenario} -> {out.dir}")
    index = open_cmip6_index(cfg, scenario)
    mmem = fc.multi_model_mean(index, cfg["cmip6"]["weighting"], cfg["split"]["min_model_fraction"])
    basis = fc.make_basis(mmem, cfg["split"])

    if not out.exists("basis.nc"):
        write_netcdf(xr.Dataset({"basis": basis, "mmem": mmem.rename("mmem")}), out.path("basis.nc"), cfg)
    if not out.exists("cmip6_index.nc"):
        write_netcdf(fc.split_ensemble(index, basis), out.path("cmip6_index.nc"), cfg)

    for name in obs_index_names(cfg):
        if not out.exists(f"obs_{name}.nc"):
            write_netcdf(index_coefficients(open_obs_index(cfg, name), basis, cfg), out.path(f"obs_{name}.nc"), cfg)

    if with_targets:
        for target_variable in available_target_variables(cfg):
            ensure_target_coefficients(cfg, scenario, out, basis, target_variable)

    log.info(f"split {scenario} done")
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        for index in index_names(cfg):
            run_scenario(cfg, scenario, index)


class SplitResults:
    """access to the split of one scenario and index (default: the primary index, which also holds the targets)"""

    def __init__(self, cfg, scenario: str, index: str = None):
        cfg = index_cfg(cfg, index)
        self.cfg = cfg
        self.index = cfg["predictor"]["name"]
        self.scenario = scenario
        self.out = split_output(cfg, scenario)
        with xr.open_dataset(self.out.path("basis.nc")) as ds:
            self.basis = ds["basis"].load()
            self.mmem = ds["mmem"].load()

    def cmip6_internal(self) -> xr.DataArray:
        """internal variability (member, year) with the member coordinates of the index"""
        with xr.open_dataset(self.out.path("cmip6_index.nc")) as ds:
            internal = ds["internal"].load()
        index = open_cmip6_index(self.cfg, self.scenario)
        extra = [name for name in index.coords if name not in index.dims and index[name].dims == ("member",)]
        return internal.assign_coords({name: index[name].sel(member=internal.member) for name in extra})

    def cmip6_forced(self) -> xr.DataArray:
        with xr.open_dataset(self.out.path("cmip6_index.nc")) as ds:
            return ds["forced"].load()

    def obs_coef(self, name: str, case: str) -> xr.DataArray:
        with xr.open_dataset(self.out.path(f"obs_{name}.nc")) as ds:
            return ds[case].load()

    def obs_internal(self, name: str, coef: xr.DataArray) -> xr.DataArray:
        """internal part of an observed index for given coefficients (any case dims)"""
        series = open_obs_index(self.cfg, name).dropna("year")
        return series - fc.forced_part(fc.design_matrix(self.basis, series.year.values), coef)

    def target_coef(self, target_variable: str, case: str) -> xr.DataArray:
        fl = ensure_target_coefficients(self.cfg, self.scenario, self.out, self.basis, target_variable)
        return xr.open_dataset(fl)[case]

    def target_design(self, target_variable: str, years) -> xr.DataArray:
        """design matrix of the target's forced fit: (year, term), or (year, term, latitude, longitude) for `forced: cmip6`"""
        if target_forced(self.cfg, target_variable) == "index":
            return fc.design_matrix(self.basis, years, lag=self.cfg["target"]["predictor_lag"])
        fl = ensure_target_coefficients(self.cfg, self.scenario, self.out, self.basis, target_variable)
        with xr.open_dataset(fl) as ds:
            return fc.pointwise_design(ds["basis_cmip6"].load(), years)

