"""
Step 5b: verification of the target methods (`<root>/evaluate_target/<hash>/`), per scenario and target variable.

Change of the target-window mean against the observed reference (years Y-19 ... Y), relative [%] or
absolute as the target's `change`. All predictions are Gaussian (mean, spread):
  M0    reversion to the long-term mean, no index information: change = -observed internal reference
  perfect_index  the linear model given the observed index in the target window (upper limit, not a forecast)
  M1    persistence: no change, spread of the observed changes (non-overlapping samples)
  M2a/M3a  change of the target's CMIP6 variable (e.g. sfcWind for wsp100) in all/analogue members
  M2b/M3b  linear model on the index of all/analogue members:
           internal change = slope * x + intercept - observed internal reference,
           spread^2 = slope^2 * var(x) + sigma^2, x = predictor window mean of the members
Targets:
  full      observed change. M2b/M3b add the forced part: `a` (regression on the index MMEM, the
            default) and, with a CMIP6 variable, `b` (change of its CMIP6 MMEM; suffix _b)
  internal  observed internal change (forced signal removed without target and reference period).
            M2a/M3a: minus the change of the CMIP6 MMEM
M2a/M3a and `b` need a CMIP6 variable (not available for energy). The CMIP6 changes are taken as they
are: with `change: absolute` they are in CMIP6 units (e.g. K for tas, m/s of the 10 m wind for wsp100).

Outputs: `scores.nc` (fc_method, target, window, latitude, longitude), `sites.nc` with
block-bootstrap intervals of the correlation at the sites of `region.sites`, and
`significance.nc`: local and field significance of the skill beyond M0 (see `surrogates.py`).
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.special import ndtr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label, window_means
from climate_scenarios_2050.evaluate import metrics
from climate_scenarios_2050.evaluate.significance import score_with_uncertainty
from climate_scenarios_2050.linear_model.fit import linear_model_output
from climate_scenarios_2050.methods.target import target_hindcast_output
from climate_scenarios_2050.preprocess import obs
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, available_target_variables

log = logging.getLogger(__name__)


def target_evaluation_output(cfg, scenario, target_variable) -> StepOutput:
    return StepOutput(
        cfg, "evaluate_target", sections=["evaluation"],
        extra={"scenario": scenario, "target_variable": target_variable, "spec": target_spec(cfg, target_variable), "sites": cfg["region"]["sites"]},
        upstream={"hindcast_target": target_hindcast_output(cfg, scenario, target_variable).hash, "linear_model": linear_model_output(cfg, scenario, target_variable).hash},
    )


def to_target_grid(da: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """CMIP6 field (lat, lon) interpolated to the grid of an observed target (latitude, longitude)"""
    return da.interp(lat=like["latitude"].values, lon=like["longitude"].values).rename(lat="latitude", lon="longitude").assign_coords(latitude=like["latitude"], longitude=like["longitude"])


def index_forced_change(split, target_variable, years, init_years, windows, reference_length) -> xr.DataArray:
    """change of the forced part (a) in units of the target, per (init_year, window, space)"""
    coef = split.target_coef(target_variable, "coef_window").load()
    design = split.target_design(years)
    delta = np.full((len(init_years), len(windows), design.sizes["term"]), np.nan)
    for ii, init in enumerate(init_years):
        ref = design.sel(year=slice(int(init) - reference_length + 1, int(init))).mean("year")
        for iw, (a, b) in enumerate(windows):
            delta[ii, iw] = (design.sel(year=slice(int(init) + a, int(init) + b)).mean("year") - ref).values
    delta = xr.DataArray(delta, dims=("init_year", "window", "term"), coords={"init_year": init_years, "window": [window_label(w) for w in windows], "term": design["term"]})
    return (delta * coef.sel(init_year=init_years)).sum("term", skipna=False)


def persistence_spread(target: xr.DataArray, windows) -> xr.DataArray:
    """leave-out spread of the observed changes around zero (non-overlapping samples)"""
    out = []
    for (a, b), label in zip(windows, target.window.values):
        sub = target.sel(window=label)
        keep = metrics.separated(sub.init_year.values, b - a + 1)
        out.append(xr.concat([np.sqrt((sub.isel(init_year=keep[ii]) ** 2).mean("init_year")) for ii in range(sub.sizes["init_year"])], dim=sub.init_year))
    return xr.concat(out, dim=target.window).transpose(*target.dims)


def gaussian_scores(mean, spread, target, reference_spread, quantiles) -> xr.Dataset:
    mask = mean.notnull() & target.notnull()
    scores = metrics.deterministic(mean.where(mask), target.where(mask))
    crps = metrics.crps_gaussian(mean, spread, target).where(mask).mean("init_year")
    crps_ref = metrics.crps_gaussian(xr.zeros_like(reference_spread), reference_spread, target).where(mask).mean("init_year")
    thresholds = target.where(mask).quantile(list(quantiles), dim="init_year")
    prob = 1 - xr.apply_ufunc(ndtr, (thresholds - mean) / spread)
    event = (target > thresholds).astype("float32")
    brier = ((prob - event) ** 2).where(mask).mean("init_year")
    brier_ref = ((1 - thresholds["quantile"] - event) ** 2).where(mask).mean("init_year")
    return scores.assign(crps=crps, crpss=1 - crps / crps_ref, brier=brier, bss=1 - brier / brier_ref)


def build_predictions(cfg, scenario, target_variable):
    """targets and predictions (mean, spread) in the requested units"""
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    ref_len = cfg["target"]["reference_length"]
    params = xr.open_dataset(linear_model_output(cfg, scenario, target_variable).path("hindcast.nc")).load()
    inputs = target_hindcast_output(cfg, scenario, target_variable)
    split = SplitResults(cfg, scenario)
    with xr.open_dataset(inputs.path("index_stats.nc")) as ds:
        index = ds.load()
    inits = np.intersect1d(params.init_year.values, index.init_year.values)
    params, index = params.sel(init_year=inits), index.sel(init_year=inits)

    wind = obs.open_obs_target(cfg, target_variable)
    raw = window_means(wind, windows, lag=0, reference_length=ref_len).sel(init_year=inits)
    raw_reference = raw["reference_mean"]
    absolute = target_spec(cfg, target_variable)["change"] == "absolute"
    # units of the target -> units of the change (the CMIP6 changes are already in those units)
    to_units = xr.ones_like(raw_reference) if absolute else 100 / raw_reference.where(raw_reference > 0)

    target_full = (raw["window_mean"] - raw_reference) * to_units
    target_internal = (params["obs_window"] - params["obs_reference"]) * to_units
    forced_a = index_forced_change(split, target_variable, wind.year.values, inits, windows, ref_len) * to_units

    methods = {"full": {}, "internal": {}}
    for ensemble, name in [("all", "M2b"), ("analogue", "M3b")]:
        x_mean = index["mean"].sel(ensemble=ensemble, drop=True)
        x_std = index["std"].sel(ensemble=ensemble, drop=True)
        internal = (params["slope"] * x_mean + params["intercept"] - params["obs_reference"]) * to_units
        spread = np.sqrt((params["slope"] * x_std) ** 2 + params["sigma"] ** 2) * to_units
        methods["internal"][name] = (internal, spread)
        methods["full"][name] = (internal + forced_a, spread)

    # references for the value of the index: reversion to the long-term mean without any index information
    # (the internal part has zero mean over the fitting years), and the observed index in the target window
    reversion = -params["obs_reference"] * to_units
    perfect = (params["slope"] * params["index_window"] + params["intercept"] - params["obs_reference"]) * to_units
    methods["internal"]["M0"] = (reversion, params["sigma"] * to_units)
    methods["full"]["M0"] = (reversion + forced_a, params["sigma"] * to_units)
    methods["internal"]["perfect_index"] = (perfect, params["sigma"] * to_units)
    methods["full"]["perfect_index"] = (perfect + forced_a, params["sigma"] * to_units)

    if inputs.exists("cmip6.nc") and inputs.exists("cmip6_mmem.nc"):
        with xr.open_dataset(inputs.path("cmip6.nc")) as ds:
            sfc = ds.sel(init_year=inits).load()
        with xr.open_dataset(inputs.path("cmip6_mmem.nc")) as ds:
            mmem = to_target_grid(ds["mmem_change"].sel(init_year=inits).load(), wind)
        for ensemble, name in [("all", "M2a"), ("analogue", "M3a")]:
            mean = to_target_grid(sfc["mean"].sel(ensemble=ensemble, drop=True), wind)
            spread = to_target_grid(sfc["std"].sel(ensemble=ensemble, drop=True), wind)
            methods["full"][name] = (mean, spread)
            methods["internal"][name] = (mean - mmem, spread)
        for name in ["M2b", "M3b"]:
            internal, spread = methods["internal"][name]
            methods["full"][f"{name}_b"] = (internal + mmem, spread)

    targets = {"full": target_full, "internal": target_internal}
    for kind, target in targets.items():
        target = target.where(target.notnull()).transpose("init_year", "window", ...)
        targets[kind] = target
        spread = persistence_spread(target, windows)
        methods[kind]["M1"] = (xr.zeros_like(target), spread)
    return targets, methods, to_units


def _site_row(cfg, site, window, length, kind, name, mean, target):
    boot = cfg["evaluation"]["bootstrap"]
    block = max(int(round(len(target) ** (1 / 3))), length)
    res = score_with_uncertainty(target, mean, block, boot["n"], "corr", boot["seed"])
    return dict(site=site, window=window, target=kind, fc_method=name, **res)


def site_table(cfg, targets, methods, windows) -> xr.Dataset:
    jobs = []
    for site, loc in cfg["region"]["sites"].items():
        loc = {"lat": loc["lat"], "lon": dom.point_lon(loc["lon"], cfg["region"]["domain"])}
        for (a, b) in windows:
            label = window_label((a, b))
            for kind, target in targets.items():
                t = target.sel(window=label).sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest").values
                for name, (mean, _) in methods[kind].items():
                    m = mean.sel(window=label).sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest").transpose("init_year").values
                    if name != "M1":
                        jobs.append(delayed(_site_row)(cfg, site, label, b - a + 1, kind, name, m, t))
    rows = Parallel(n_jobs=cfg["resources"]["n_jobs"])(jobs)
    return xr.Dataset.from_dataframe(pd.DataFrame(rows).set_index(["target", "fc_method", "window", "site"]))


def run_case(cfg, scenario, target_variable):
    out = target_evaluation_output(cfg, scenario, target_variable).prepare()
    log.info(f"target evaluation {scenario} {target_variable} -> {out.dir}")
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    targets, methods, to_units = build_predictions(cfg, scenario, target_variable)
    quantiles = cfg["evaluation"]["quantiles"]
    if not out.exists("scores.nc"):
        kinds = []
        for kind, target in targets.items():
            reference = methods[kind]["M1"][1]
            rows = [gaussian_scores(mean, spread, target, reference, quantiles).expand_dims(fc_method=[name]) for name, (mean, spread) in methods[kind].items()]
            kinds.append(xr.concat(rows, dim="fc_method").expand_dims(target=[kind]))
        scores = xr.concat(kinds, dim="target", join="outer")
        write_netcdf(scores, out.path("scores.nc"), cfg, compress=True)
    if not out.exists("sites.nc"):
        write_netcdf(site_table(cfg, targets, methods, windows), out.path("sites.nc"), cfg)
    if cfg["evaluation"].get("significance", {}).get("enabled") and not out.exists("significance.nc"):
        from climate_scenarios_2050.evaluate.surrogates import surrogate_test

        write_netcdf(surrogate_test(cfg, scenario, target_variable, targets, methods, to_units), out.path("significance.nc"), cfg, compress=True)
    from climate_scenarios_2050.plotting.target import plot_target_evaluation
    plot_target_evaluation(cfg, out, scenario, target_variable)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        for target_variable in available_target_variables(cfg):
            run_case(cfg, scenario, target_variable)
