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
- `wind_<variable>_<hash>.nc` the same for observed wind (`coef_window` only for the hindcast scenarios),
    hash of the processed wind input (a new wind file gives new coefficients without touching the rest);
    the design matrix uses the basis of season t-L for calendar year t
- `sfcWind_mmem.nc`   MMEM of annual CMIP6 sfcWind (year, lat, lon), where sfcWind is available
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices.compute import index_output, open_cmip6_index, open_obs_index
from climate_scenarios_2050.split import forced as fc

log = logging.getLogger(__name__)


def split_output(cfg, scenario: str) -> StepOutput:
    return StepOutput(
        cfg, "split", sections=["split", "target"],
        extra={"scenario": scenario, "weighting": cfg["cmip6"]["weighting"]},
        upstream={"indices": index_output(cfg).hash},
    )


def wind_file(cfg, wind_variable: str) -> str:
    return f"wind_{wind_variable}_{obs.wind_output(cfg, wind_variable).hash}.nc"


def available_wind_variables(cfg) -> list:
    """wind variables whose input file exists"""
    out = []
    for name, path in cfg["paths"]["inputs"]["wind"].items():
        if Path(path).exists():
            out.append(name)
        else:
            log.warning(f"{name}: input file {path} does not exist, skipping")
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


def wind_coefficients(wind: xr.DataArray, basis: xr.DataArray, cfg, with_windows: bool) -> xr.Dataset:
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


def sfcwind_mmem(cfg, scenario: str):
    """model- or member-weighted MMEM of annual sfcWind; None if not available for this scenario"""
    try:
        field = cmip6.open_field(cfg, "sfcWind", "annual", scenario)["sfcWind"]
    except FileNotFoundError:
        log.info(f"no preprocessed sfcWind for {scenario}")
        return None
    if cfg["cmip6"]["weighting"] == "member":
        mmem = field.mean("member")
    else:
        smems = [field.isel(member=(field["model_id"] == mid).values).mean("member") for mid in np.unique(field["model_id"].values)]
        mmem = xr.concat(smems, dim="model_id").mean("model_id")
    return mmem.compute().rename("sfcWind_mmem")


def run_scenario(cfg, scenario: str):
    out = split_output(cfg, scenario).prepare()
    log.info(f"split {scenario} -> {out.dir}")
    index = open_cmip6_index(cfg, scenario)
    mmem = fc.multi_model_mean(index, cfg["cmip6"]["weighting"], cfg["split"]["min_model_fraction"])
    basis = fc.make_basis(mmem, cfg["split"])

    if not out.exists("basis.nc"):
        write_netcdf(xr.Dataset({"basis": basis, "mmem": mmem.rename("mmem")}), out.path("basis.nc"), cfg)
    if not out.exists("cmip6_index.nc"):
        write_netcdf(fc.split_ensemble(index, basis), out.path("cmip6_index.nc"), cfg)

    for name in [*cfg["preprocess"]["obs_datasets"], "merged"]:
        if not out.exists(f"obs_{name}.nc"):
            write_netcdf(index_coefficients(open_obs_index(cfg, name), basis, cfg), out.path(f"obs_{name}.nc"), cfg)

    with_windows = scenario in cfg["cmip6"]["scenarios_hindcast"]
    for wind_variable in available_wind_variables(cfg):
        if not out.exists(wind_file(cfg, wind_variable)):
            coefs = wind_coefficients(obs.open_obs_wind(cfg, wind_variable), basis, cfg, with_windows)
            write_netcdf(coefs, out.path(wind_file(cfg, wind_variable)), cfg, compress=True)

    if not out.exists("sfcWind_mmem.nc"):
        mmem_wind = sfcwind_mmem(cfg, scenario)
        if mmem_wind is not None:
            write_netcdf(mmem_wind.to_dataset(), out.path("sfcWind_mmem.nc"), cfg, compress=True)
    log.info(f"split {scenario} done")
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        run_scenario(cfg, scenario)


class SplitResults:
    """access to the split of one scenario"""

    def __init__(self, cfg, scenario: str):
        self.cfg = cfg
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

    def wind_coef(self, wind_variable: str, case: str) -> xr.DataArray:
        return xr.open_dataset(self.out.path(wind_file(self.cfg, wind_variable)))[case]

    def wind_design(self, years) -> xr.DataArray:
        return fc.design_matrix(self.basis, years, lag=self.cfg["target"]["predictor_lag"])

    def sfcwind_mmem(self):
        fl = self.out.path("sfcWind_mmem.nc")
        if not fl.exists():
            return None
        with xr.open_dataset(fl) as ds:
            return ds["sfcWind_mmem"].load()
