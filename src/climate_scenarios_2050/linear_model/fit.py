"""
Step 4: linear model predicting wind from the predictor index (`<root>/linear_model/<hash>/`).

    wind_internal(window mean) = slope * index_internal(window mean) + intercept + eps

trained on observations (ERA5 index, ERA5-based wind), with one fit per grid point and target case.
Samples are sliding window means of the target window's length, so the model works on the same
time scale as the prediction. An annual wind value of year t is paired with the DJFM index of season
t - lag (the winter ending in year t).

`hindcast_<wind>.nc`, dims (init_year, window, ...): for verification. The forced signal is removed
with coefficients fitted without the target window and reference period (leakage option C), and
training samples overlapping the target window are excluded. Also holds the observed internal
window and reference means of wind and index of each case, so that a "perfect index" prediction can
be verified. Only for the hindcast scenarios.

`forecast_<wind>.nc`, dims (init_year, length, ...): for real forecasts, only data up to the init year.
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.preprocess import obs
from climate_scenarios_2050.preprocess.indices.compute import open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, available_wind_variables, split_output

log = logging.getLogger(__name__)


def linear_model_output(cfg, scenario: str, wind_variable: str) -> StepOutput:
    return StepOutput(
        cfg, "linear_model", sections=["linear_model", "target"],
        extra={"scenario": scenario, "wind_variable": wind_variable, "training_index": cfg["obs"]["training_index"]},
        upstream={"split": split_output(cfg, scenario).hash, "obs_wind": obs.wind_output(cfg, wind_variable).hash},
    )


def sliding_means(values: np.ndarray, length: int) -> np.ndarray:
    """means of `length` consecutive values along the first axis; NaN if any value is missing"""
    filled = np.nan_to_num(values)
    csum = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(filled, axis=0)])
    missing = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(~np.isfinite(values), axis=0)])
    means = (csum[length:] - csum[:-length]) / length
    return np.where(missing[length:] - missing[:-length] > 0, np.nan, means)


def regress(x: np.ndarray, y: np.ndarray):
    """slope, intercept, residual std (ddof=2) of y (samples, points) on x (samples)"""
    x_anom = x - x.mean()
    y_mean = y.mean(axis=0)
    slope = (x_anom[:, None] * (y - y_mean)).sum(axis=0) / (x_anom**2).sum()
    intercept = y_mean - slope * x.mean()
    residual = y - (slope * x[:, None] + intercept)
    sigma = np.sqrt((residual**2).sum(axis=0) / (len(x) - 2))
    return slope, intercept, sigma


def fit_case(index_internal: np.ndarray, wind_internal: np.ndarray, length: int, target_years=None, min_samples: int = 20):
    """
    `index_internal` (years) and `wind_internal` (years, points) are paired year by year and have
    no gaps. `target_years`: (first, last, year axis) of the target window; overlapping samples
    are excluded. Returns (slope, intercept, sigma, n_samples), NaN with too few samples.
    """
    x = sliding_means(index_internal, length)
    y = sliding_means(wind_internal, length)
    keep = np.isfinite(x)
    if target_years is not None:
        first, last, years = target_years
        end = years[length - 1 :]
        keep &= (end < first) | (end - length + 1 > last)
    n_points = wind_internal.shape[1]
    if keep.sum() < min_samples:
        return np.full(n_points, np.nan), np.full(n_points, np.nan), np.full(n_points, np.nan), int(keep.sum())
    return (*regress(x[keep], y[keep]), int(keep.sum()))


class Training:
    """observed series and split of one scenario, paired for training"""

    def __init__(self, cfg, scenario: str, wind_variable: str):
        self.split = SplitResults(cfg, scenario)
        self.lag = cfg["target"]["predictor_lag"]
        self.ref = cfg["target"]["reference_length"]
        self.min_samples = cfg["linear_model"]["min_samples"]
        self.index_name = cfg["obs"]["training_index"]
        self.wind_variable = wind_variable
        index = open_obs_index(cfg, self.index_name).dropna("year")
        wind = obs.open_obs_wind(cfg, wind_variable)
        # pair wind of year t with the season t - lag
        years = np.intersect1d(wind.year.values, index.year.values + self.lag)
        self.years = years
        self.wind_raw = wind.sel(year=years).transpose("year", ...)
        self.space_dims = list(self.wind_raw.dims[1:])
        self.n_points = int(np.prod([self.wind_raw.sizes[d] for d in self.space_dims]))
        self.wind_values = self.wind_raw.values.reshape(len(years), -1)
        self.index_raw = index.sel(year=years - self.lag)
        self.index_design = fc.design_matrix(self.split.basis, self.index_raw.year.values).values
        self.wind_design = self.split.wind_design(years).values

    def internal(self, index_coef: np.ndarray, wind_coef: np.ndarray):
        """internal index (years) and wind (years, points) for coefficients of one case"""
        index_internal = self.index_raw.values - self.index_design @ index_coef
        wind_internal = self.wind_values - self.wind_design @ wind_coef.reshape(wind_coef.shape[0], -1)
        return index_internal, wind_internal

    def unstack(self, values: np.ndarray, extra_shape=()) -> np.ndarray:
        return values.reshape(*extra_shape, *[self.wind_raw.sizes[d] for d in self.space_dims])

    def space_coords(self) -> dict:
        return {d: self.wind_raw[d].values for d in self.space_dims}


def hindcast_parameters(cfg, training: Training) -> xr.Dataset:
    """one fit per (init_year, window); option A uses the same coefficients for all cases"""
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    labels = [window_label(w) for w in windows]
    full_period = cfg["split"]["obs_leakage"] == "full_period"
    split, name, var = training.split, training.index_name, training.wind_variable
    lag = training.lag
    index_case = split.obs_coef(name, "coef_full" if full_period else f"coef_window_lag{lag}")
    wind_case = split.wind_coef(var, "coef_full" if full_period else "coef_window").load()
    if full_period:
        all_inits = training.years
    else:
        all_inits = np.intersect1d(index_case.init_year.values, wind_case.init_year.values)
    fields = ["slope", "intercept", "sigma", "obs_window", "obs_reference"]
    shape = (len(all_inits), len(windows), training.n_points)
    out = {f: np.full(shape, np.nan, dtype="float32") for f in fields}
    index_window = np.full(shape[:2], np.nan)
    index_reference = np.full(shape[:2], np.nan)
    n_samples = np.zeros(shape[:2], dtype="int32")

    years = training.years
    for ii, init in enumerate(all_inits):
        for iw, (a, b) in enumerate(windows):
            if init + b > years.max() or init - training.ref + 1 - lag < years.min() - lag:
                continue
            if full_period:
                icoef, wcoef = index_case.values, wind_case.values
            else:
                icoef = index_case.sel(init_year=init).isel(window=iw).values
                wcoef = wind_case.sel(init_year=init).isel(window=iw).values
            if not np.isfinite(icoef).all() or not np.isfinite(wcoef).any():
                continue
            index_internal, wind_internal = training.internal(icoef, wcoef)
            first, last = init + a, init + b
            slope, intercept, sigma, n = fit_case(index_internal, wind_internal, b - a + 1, (first, last, years), training.min_samples)
            in_win = (years >= first) & (years <= last)
            in_ref = (years > init - training.ref) & (years <= init)
            if in_ref.sum() < training.ref:
                continue
            out["slope"][ii, iw], out["intercept"][ii, iw], out["sigma"][ii, iw] = slope, intercept, sigma
            out["obs_window"][ii, iw] = wind_internal[in_win].mean(axis=0)
            out["obs_reference"][ii, iw] = wind_internal[in_ref].mean(axis=0)
            index_window[ii, iw] = index_internal[in_win].mean()
            index_reference[ii, iw] = index_internal[in_ref].mean()
            n_samples[ii, iw] = n

    dims = ("init_year", "window", *training.space_dims)
    ds = xr.Dataset(
        {f: (dims, training.unstack(out[f], shape[:2])) for f in fields},
        coords={"init_year": all_inits, "window": labels, **training.space_coords()},
    )
    ds["index_window"] = (("init_year", "window"), index_window)
    ds["index_reference"] = (("init_year", "window"), index_reference)
    ds["n_samples"] = (("init_year", "window"), n_samples)
    return ds.sel(init_year=ds["n_samples"].max("window") > 0)


def forecast_parameters(cfg, training: Training) -> xr.Dataset:
    """one fit per (init_year, window length), using only data up to the init year"""
    lengths = sorted({w[1] - w[0] + 1 for w in cfg["target"]["windows"]})
    split, name, var = training.split, training.index_name, training.wind_variable
    index_case = split.obs_coef(name, "coef_up_to_init")
    wind_case = split.wind_coef(var, "coef_up_to_init").load()
    years = training.years
    inits = np.intersect1d(np.intersect1d(index_case.init_year.values, wind_case.init_year.values), years)
    shape = (len(inits), len(lengths), training.n_points)
    out = {f: np.full(shape, np.nan, dtype="float32") for f in ["slope", "intercept", "sigma"]}
    n_samples = np.zeros(shape[:2], dtype="int32")
    for ii, init in enumerate(inits):
        icoef = index_case.sel(init_year=init).values
        wcoef = wind_case.sel(init_year=init).values
        if not np.isfinite(icoef).all() or not np.isfinite(wcoef).any():
            continue
        index_internal, wind_internal = training.internal(icoef, wcoef)
        known = years <= init
        for il, length in enumerate(lengths):
            slope, intercept, sigma, n = fit_case(index_internal[known], wind_internal[known], length, None, training.min_samples)
            out["slope"][ii, il], out["intercept"][ii, il], out["sigma"][ii, il] = slope, intercept, sigma
            n_samples[ii, il] = n
    dims = ("init_year", "length", *training.space_dims)
    ds = xr.Dataset(
        {f: (dims, training.unstack(out[f], shape[:2])) for f in out},
        coords={"init_year": inits, "length": lengths, **training.space_coords()},
    )
    ds["n_samples"] = (("init_year", "length"), n_samples)
    return ds.sel(init_year=ds["n_samples"].max("length") > 0)


def perfect_index_skill(params: xr.Dataset) -> xr.Dataset:
    """
    skill of the model given the observed index: out-of-sample prediction of the internal change
    relative to the reference period, an upper limit for any index forecast
    """
    predicted = params["slope"] * (params["index_window"] - params["index_reference"])
    target = params["obs_window"] - params["obs_reference"]
    mask = predicted.notnull() & target.notnull()
    predicted, target = predicted.where(mask), target.where(mask)
    rmse = np.sqrt(((predicted - target) ** 2).mean("init_year"))
    return xr.Dataset({
        "corr": xr.corr(predicted, target, dim="init_year"),
        "nrmse": 1 - rmse / target.std("init_year"),
        "n_samples": mask.sum("init_year"),
    })


def run_scenario(cfg, scenario: str):
    hindcast_scenario = scenario in cfg["cmip6"]["scenarios_hindcast"]
    for wind_variable in available_wind_variables(cfg):
        out = linear_model_output(cfg, scenario, wind_variable).prepare()
        log.info(f"linear model {scenario} {wind_variable} -> {out.dir}")
        training = Training(cfg, scenario, wind_variable)
        if hindcast_scenario and not out.exists("hindcast.nc"):
            write_netcdf(hindcast_parameters(cfg, training), out.path("hindcast.nc"), cfg, compress=True)
        if hindcast_scenario and not out.exists("perfect_index_skill.nc"):
            with xr.open_dataset(out.path("hindcast.nc")) as params:
                write_netcdf(perfect_index_skill(params.load()), out.path("perfect_index_skill.nc"), cfg, compress=True)
        if not out.exists("forecast.nc"):
            write_netcdf(forecast_parameters(cfg, training), out.path("forecast.nc"), cfg, compress=True)
        if hindcast_scenario:
            from climate_scenarios_2050.plotting.linear_model import plot_linear_model
            plot_linear_model(cfg, out, scenario, wind_variable)


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        run_scenario(cfg, scenario)
