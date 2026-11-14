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
    touching the rest); the design matrix uses the basis of season t-L for calendar year t

The split depends only on the predictor: the CMIP6 data of the targets enter in `hindcast-target`, so a
download of e.g. more tas members leaves the split and the index steps unchanged.
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


def target_file(cfg, target_variable: str) -> str:
    return f"target_{target_variable}_{obs.target_output(cfg, target_variable).hash}.nc"


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


def target_coefficients(wind: xr.DataArray, basis: xr.DataArray, cfg, with_windows: bool) -> xr.Dataset:
    """all leakage cases for observed annual wind (calendar years, paired with the previous season's basis)"""
    years = wind.year.values
    design = fc.design_matrix(basis, years, lag=cfg["target"]["predictor_lag"])
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
    fl = out.path(target_file(cfg, target_variable))
    if not fl.exists():
        log.info(f"split {scenario}: forced-fit coefficients for the current {target_variable} input -> {fl.name}")
        with_windows = scenario in cfg["cmip6"]["scenarios_hindcast"]
        coefs = target_coefficients(obs.open_obs_target(cfg, target_variable), basis, cfg, with_windows)
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

    def target_design(self, years) -> xr.DataArray:
        return fc.design_matrix(self.basis, years, lag=self.cfg["target"]["predictor_lag"])

