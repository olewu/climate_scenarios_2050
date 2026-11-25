"""
Step 4: linear model predicting a target from the predictor index(es) (`<root>/linear_model/<hash>/`).

    target_internal(window mean) = sum_p slope_p * index_p_internal(window mean) + intercept + eps

trained on observations (e.g. ERA5 index, ERA5-based wind), with one fit per grid point and target case.
The predictors are `linear_model.predictors` (default: the primary index); with several, an optional
ridge penalty (`linear_model.ridge`, relative to the predictors' variance) shrinks the slopes. Samples are
sliding window means of the target window's length, so the model works on the same time scale as the
prediction. An annual target value of year t is paired with the index of season t - lag (e.g. the winter
ending in year t). Every index is split into forced and internal parts with its own split; the target with
the primary index's split.

`hindcast.nc`, dims (init_year, window, [predictor,] ...): for verification. The forced signal is removed
with coefficients fitted without the target window and reference period (leakage option C), and
training samples overlapping the target window are excluded. Also holds the observed internal
window and reference means of target and indices of each case, so that a "perfect index" prediction can
be verified. Only for the hindcast scenarios.

`forecast.nc`, dims (init_year, length, [predictor,] ...): for real forecasts, only data up to the init year.

`slope`, `index_window`, `index_reference` carry a `predictor` dimension.
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.config import lm_predictors
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.preprocess import obs
from climate_scenarios_2050.preprocess.indices.compute import open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, available_target_variables, predictor_split_hashes, target_forced_id

log = logging.getLogger(__name__)


def linear_model_output(cfg, scenario: str, target_variable: str) -> StepOutput:
    extra = {"scenario": scenario, "target_variable": target_variable, "training_index": cfg["obs"]["training_index"]}
    forced = target_forced_id(cfg, scenario, target_variable)
    if forced:   # only when not the default (`index`), so existing hashes stay valid
        extra["target_forced"] = forced
    return StepOutput(
        cfg, "linear_model", sections=["linear_model", "target"],
        extra=extra,
        upstream={"split": predictor_split_hashes(cfg, scenario), "obs_target": obs.target_output(cfg, target_variable).hash},
    )


def sliding_means(values: np.ndarray, length: int) -> np.ndarray:
    """means of `length` consecutive values along the first axis; NaN if any value is missing"""
    filled = np.nan_to_num(values)
    csum = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(filled, axis=0)])
    missing = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(~np.isfinite(values), axis=0)])
    means = (csum[length:] - csum[:-length]) / length
    return np.where(missing[length:] - missing[:-length] > 0, np.nan, means)


def regress(x: np.ndarray, y: np.ndarray, ridge: float = 0.0):
    """
    slopes (predictor, points), intercept and residual std (ddof = 1 + predictors) of y (samples, points)
    on x (samples[, predictor]); `ridge` adds ridge * var(x_p) * n to the diagonal (scale-free shrinkage)
    """
    x = x[:, None] if x.ndim == 1 else x
    n, p = x.shape
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xa, ya = x - x_mean, y - y_mean
    gram = xa.T @ xa
    if ridge:
        gram = gram + ridge * np.diag(np.diag(gram))
    slope = np.linalg.solve(gram, xa.T @ ya)
    intercept = y_mean - x_mean @ slope
    residual = ya - xa @ slope
    sigma = np.sqrt((residual**2).sum(axis=0) / (n - p - 1))
    return slope, intercept, sigma


def fit_case(index_internal: np.ndarray, wind_internal: np.ndarray, length: int, target_years=None, min_samples: int = 20, ridge: float = 0.0):
    """
    `index_internal` (years[, predictor]) and `wind_internal` (years, points) are paired year by year and
    have no gaps. `target_years`: (first, last, year axis) of the target window; overlapping samples
    are excluded. Returns (slope (predictor, points), intercept, sigma, n_samples), NaN with too few samples.
    """
    index_internal = index_internal[:, None] if index_internal.ndim == 1 else index_internal
    x = sliding_means(index_internal, length)
    y = sliding_means(wind_internal, length)
    keep = np.isfinite(x).all(axis=1)
    if target_years is not None:
        first, last, years = target_years
        end = years[length - 1 :]
        keep &= (end < first) | (end - length + 1 > last)
    n_points = wind_internal.shape[1]
    if keep.sum() < max(min_samples, x.shape[1] + 2):
        nan = np.full(n_points, np.nan)
        return np.full((x.shape[1], n_points), np.nan), nan, nan.copy(), int(keep.sum())
    return (*regress(x[keep], y[keep], ridge), int(keep.sum()))


class Training:
    """observed indices, target and splits of one scenario, paired for training"""

    def __init__(self, cfg, scenario: str, target_variable: str):
        self.predictors = lm_predictors(cfg)
        self.split = SplitResults(cfg, scenario)          # primary index: forced part of the target
        self.splits = {name: SplitResults(cfg, scenario, name) for name in self.predictors}
        self.lag = cfg["target"]["predictor_lag"]
        self.ref = cfg["target"]["reference_length"]
        self.min_samples = cfg["linear_model"]["min_samples"]
        self.ridge = float(cfg["linear_model"].get("ridge") or 0.0)
        self.index_name = cfg["obs"]["training_index"]
        self.target_variable = target_variable
        indices = {name: open_obs_index(cfg, self.index_name, name).dropna("year") for name in self.predictors}
        wind = obs.open_obs_target(cfg, target_variable)
        # pair the target of year t with the season t - lag
        years = wind.year.values
        for index in indices.values():
            years = np.intersect1d(years, index.year.values + self.lag)
        self.years = years
        self.wind_raw = wind.sel(year=years).transpose("year", ...)
        self.space_dims = list(self.wind_raw.dims[1:])
        self.n_points = int(np.prod([self.wind_raw.sizes[d] for d in self.space_dims]))
        self.wind_values = self.wind_raw.values.reshape(len(years), -1)
        self.index_raw = {name: index.sel(year=years - self.lag) for name, index in indices.items()}
        self.index_design = {name: fc.design_matrix(self.splits[name].basis, self.index_raw[name].year.values).values for name in self.predictors}
        design = self.split.target_design(target_variable, years)
        if design.ndim > 2:   # forced: cmip6, a regressor per grid point
            design = design.transpose("year", "term", *self.space_dims)
        self.wind_design = design.values.reshape(len(years), design.sizes["term"], -1) if design.ndim > 2 else design.values

    def index_cases(self, case: str) -> dict:
        """leave-out coefficients of every predictor index for a case (e.g. coef_window_lag1)"""
        return {name: self.splits[name].obs_coef(self.index_name, case) for name in self.predictors}

    def internal(self, index_coefs: dict, wind_coef: np.ndarray):
        """internal indices (years, predictor) and target (years, points) for the coefficients of one case"""
        index_internal = np.stack([self.index_raw[n].values - self.index_design[n] @ index_coefs[n] for n in self.predictors], axis=1)
        wind_internal = self.wind_values - fc.forced_values(self.wind_design, wind_coef.reshape(wind_coef.shape[0], -1))
        return index_internal, wind_internal

    def unstack(self, values: np.ndarray, extra_shape=()) -> np.ndarray:
        return values.reshape(*extra_shape, *[self.wind_raw.sizes[d] for d in self.space_dims])

    def space_coords(self) -> dict:
        return {d: self.wind_raw[d].values for d in self.space_dims}


def _case_inits(cases: dict, wind_case) -> np.ndarray:
    inits = wind_case.init_year.values
    for coef in cases.values():
        inits = np.intersect1d(inits, coef.init_year.values)
    return inits


def hindcast_parameters(cfg, training: Training) -> xr.Dataset:
    """one fit per (init_year, window); option A uses the same coefficients for all cases"""
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    labels = [window_label(w) for w in windows]
    full_period = cfg["split"]["obs_leakage"] == "full_period"
    var, lag, names = training.target_variable, training.lag, training.predictors
    index_cases = training.index_cases("coef_full" if full_period else f"coef_window_lag{lag}")
    index_cases = {n: c.load() for n, c in index_cases.items()}
    wind_case = training.split.target_coef(var, "coef_full" if full_period else "coef_window").load()
    all_inits = training.years if full_period else _case_inits(index_cases, wind_case)
    n_pred = len(names)
    shape = (len(all_inits), len(windows), training.n_points)
    out = {f: np.full(shape, np.nan, dtype="float32") for f in ["intercept", "sigma", "obs_window", "obs_reference"]}
    slope = np.full((len(all_inits), len(windows), n_pred, training.n_points), np.nan, dtype="float32")
    index_window = np.full((*shape[:2], n_pred), np.nan)
    index_reference = np.full((*shape[:2], n_pred), np.nan)
    n_samples = np.zeros(shape[:2], dtype="int32")

    years = training.years
    for ii, init in enumerate(all_inits):
        for iw, (a, b) in enumerate(windows):
            if init + b > years.max() or init - training.ref + 1 - lag < years.min() - lag:
                continue
            if full_period:
                icoef = {n: c.values for n, c in index_cases.items()}
                wcoef = wind_case.values
            else:
                icoef = {n: c.sel(init_year=init).isel(window=iw).values for n, c in index_cases.items()}
                wcoef = wind_case.sel(init_year=init).isel(window=iw).values
            if not all(np.isfinite(c).all() for c in icoef.values()) or not np.isfinite(wcoef).any():
                continue
            index_internal, wind_internal = training.internal(icoef, wcoef)
            first, last = init + a, init + b
            s, intercept, sigma, n = fit_case(index_internal, wind_internal, b - a + 1, (first, last, years), training.min_samples, training.ridge)
            in_win = (years >= first) & (years <= last)
            in_ref = (years > init - training.ref) & (years <= init)
            if in_ref.sum() < training.ref:
                continue
            slope[ii, iw], out["intercept"][ii, iw], out["sigma"][ii, iw] = s, intercept, sigma
            out["obs_window"][ii, iw] = wind_internal[in_win].mean(axis=0)
            out["obs_reference"][ii, iw] = wind_internal[in_ref].mean(axis=0)
            index_window[ii, iw] = index_internal[in_win].mean(axis=0)
            index_reference[ii, iw] = index_internal[in_ref].mean(axis=0)
            n_samples[ii, iw] = n

    dims = ("init_year", "window", *training.space_dims)
    ds = xr.Dataset(
        {f: (dims, training.unstack(out[f], shape[:2])) for f in out},
        coords={"init_year": all_inits, "window": labels, "predictor": names, **training.space_coords()},
    )
    ds["slope"] = (("init_year", "window", "predictor", *training.space_dims), training.unstack(slope, (*shape[:2], n_pred)))
    ds["index_window"] = (("init_year", "window", "predictor"), index_window)
    ds["index_reference"] = (("init_year", "window", "predictor"), index_reference)
    ds["n_samples"] = (("init_year", "window"), n_samples)
    ds.attrs["ridge"] = training.ridge
    return ds.sel(init_year=ds["n_samples"].max("window") > 0)


def forecast_parameters(cfg, training: Training) -> xr.Dataset:
    """one fit per (init_year, window length), using only data up to the init year"""
    lengths = sorted({w[1] - w[0] + 1 for w in cfg["target"]["windows"]})
    var, names = training.target_variable, training.predictors
    index_cases = {n: c.load() for n, c in training.index_cases("coef_up_to_init").items()}
    wind_case = training.split.target_coef(var, "coef_up_to_init").load()
    years = training.years
    inits = np.intersect1d(_case_inits(index_cases, wind_case), years)
    shape = (len(inits), len(lengths), training.n_points)
    out = {f: np.full(shape, np.nan, dtype="float32") for f in ["intercept", "sigma"]}
    slope = np.full((len(inits), len(lengths), len(names), training.n_points), np.nan, dtype="float32")
    n_samples = np.zeros(shape[:2], dtype="int32")
    for ii, init in enumerate(inits):
        icoef = {n: c.sel(init_year=init).values for n, c in index_cases.items()}
        wcoef = wind_case.sel(init_year=init).values
        if not all(np.isfinite(c).all() for c in icoef.values()) or not np.isfinite(wcoef).any():
            continue
        index_internal, wind_internal = training.internal(icoef, wcoef)
        known = years <= init
        for il, length in enumerate(lengths):
            s, intercept, sigma, n = fit_case(index_internal[known], wind_internal[known], length, None, training.min_samples, training.ridge)
            slope[ii, il], out["intercept"][ii, il], out["sigma"][ii, il] = s, intercept, sigma
            n_samples[ii, il] = n
    dims = ("init_year", "length", *training.space_dims)
    ds = xr.Dataset(
        {f: (dims, training.unstack(out[f], shape[:2])) for f in out},
        coords={"init_year": inits, "length": lengths, "predictor": names, **training.space_coords()},
    )
    ds["slope"] = (("init_year", "length", "predictor", *training.space_dims), training.unstack(slope, (*shape[:2], len(names))))
    ds["n_samples"] = (("init_year", "length"), n_samples)
    return ds.sel(init_year=ds["n_samples"].max("length") > 0)


def perfect_index_skill(params: xr.Dataset) -> xr.Dataset:
    """
    skill of the model given the observed indices: out-of-sample prediction of the internal change
    relative to the reference period, an upper limit for any index forecast
    """
    predicted = (params["slope"] * (params["index_window"] - params["index_reference"])).sum("predictor", skipna=False)
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
    for target_variable in available_target_variables(cfg):
        out = linear_model_output(cfg, scenario, target_variable).prepare()
        log.info(f"linear model {scenario} {target_variable} ({' + '.join(lm_predictors(cfg))}) -> {out.dir}")
        training = Training(cfg, scenario, target_variable)
        if hindcast_scenario and not out.exists("hindcast.nc"):
            write_netcdf(hindcast_parameters(cfg, training), out.path("hindcast.nc"), cfg, compress=True)
        if hindcast_scenario and not out.exists("perfect_index_skill.nc"):
            with xr.open_dataset(out.path("hindcast.nc")) as params:
                write_netcdf(perfect_index_skill(params.load()), out.path("perfect_index_skill.nc"), cfg, compress=True)
        if not out.exists("forecast.nc"):
            write_netcdf(forecast_parameters(cfg, training), out.path("forecast.nc"), cfg, compress=True)
        if hindcast_scenario:
            from climate_scenarios_2050.plotting.linear_model import plot_linear_model
            plot_linear_model(cfg, out, scenario, target_variable)


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        run_scenario(cfg, scenario)
