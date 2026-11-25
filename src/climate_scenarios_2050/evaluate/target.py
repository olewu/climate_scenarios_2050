"""
Step 5b: verification of the target methods (`<root>/evaluate_target/<hash>/`), per scenario and target variable.

Change of the target-window mean against the observed reference (years Y-19 ... Y), relative [%] or
absolute as the target's `change`. All predictions are Gaussian (mean, spread):
  M0    reversion to the long-term mean, no index information: change = -observed internal reference;
        spread: RMS of its own errors over the non-overlapping samples (as M1, M0c, M0t; its error is the
        observed internal window mean)
  perfect_index  the linear model given the observed index in the target window (upper limit, not a forecast)
  M1    persistence: no change, spread of the observed changes (non-overlapping samples)
  M0c/M0t  observations only (`evaluation.obs_methods`, full target only): M0 with the forced part estimated
           from the observed target alone, a constant (M0c: return to the observed mean) or a linear trend
           (M0t), fitted on the years up to init (`fit: up_to_init`) or on all years except the target window
           and reference (`leave_out`, as M0); change = fit(target window) - R[obs]; spread: RMS of their
           own errors (non-overlapping samples, as M1)
  M2a/M3a  change of the target's CMIP6 variable (e.g. sfcWind for wsp100) in all/analogue members
  M2b/M3b  linear model on the index (indices) of all/analogue members:
           internal change = sum_p slope_p x_p + intercept - observed internal reference,
           spread^2 = slope' C slope + sigma^2, x = predictor window means of the members, C their covariance
Targets:
  full      observed change. M2b/M3b add the forced part: `a` (the forced part of the target's split: regression
            on the index MMEM with `targets.<name>.forced: index`, the scaled MMEM of the target's CMIP6 variable
            with `forced: cmip6`; the default) and, with a CMIP6 variable, `b` (change of its CMIP6 MMEM; suffix _b)
  internal  observed internal change (forced signal removed without target and reference period).
            M3a: minus the change of the CMIP6 MMEM (M2a is left out: its mean is the MMEM change, so it
            would predict zero up to rounding)
M2a/M3a and `b` need a CMIP6 variable (not available for energy). The CMIP6 changes are taken as they
are: with `change: absolute` they are in CMIP6 units (e.g. K for tas, m/s of the 10 m wind for wsp100).

Outputs: `scores.nc` (fc_method, target, window, latitude, longitude), `sites.nc` with
block-bootstrap intervals of the correlation at the sites of `region.sites`, and
`significance.nc`: local and field significance of the skill beyond M0 (see `surrogates.py`),
`significance_m1.nc`: whether a method beats M1 (paired block bootstrap, `versus_m1.py`).
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
    """
    change of the forced part (a) of the target's split in units of the target, per (init_year, window, space):
    the regression on the index MMEM (`forced: index`) or the scaled MMEM of the target's CMIP6 variable (`cmip6`)
    """
    coef = split.target_coef(target_variable, "coef_window").load().sel(init_year=init_years)
    labels = [window_label(w) for w in windows]
    coef = coef.sel(window=labels)
    design = split.target_design(target_variable, years)
    space = [d for d in coef.dims if d not in ("init_year", "window", "term")]
    if design.ndim > 2:
        design = design.transpose("year", "term", *space)
    D = design.values.reshape(design.sizes["year"], design.sizes["term"], -1)   # (year, term, points or 1)
    C = coef.transpose("init_year", "window", "term", *space).values
    C = C.reshape(*C.shape[:3], -1)
    design_years = design.year.values
    out = np.full((len(init_years), len(windows), C.shape[-1]), np.nan)
    for ii, init in enumerate(init_years):
        in_ref = (design_years > int(init) - reference_length) & (design_years <= int(init))
        ref = D[in_ref].mean(axis=0)
        for iw, (a, b) in enumerate(windows):
            in_win = (design_years >= int(init) + a) & (design_years <= int(init) + b)
            if in_win.sum() < b - a + 1 or in_ref.sum() < reference_length:
                continue
            delta = D[in_win].mean(axis=0) - ref                                    # (term, points or 1)
            out[ii, iw] = (delta * C[ii, iw]).sum(axis=0)
    shape = [coef.sizes[d] for d in space]
    coords = {"init_year": init_years, "window": labels, **{d: coef[d] for d in space}}
    return xr.DataArray(out.reshape(len(init_years), len(windows), *shape), dims=("init_year", "window", *space), coords=coords)


OBS_METHODS = ("M0c", "M0t")


def observation_reference_operators(years, init_years, windows, reference_length, fit="up_to_init", min_years=20):
    """
    operators (method, init, window, year) of the observations-only methods: prediction = op @ obs, the fit of a
    constant (M0c) or a linear trend (M0t) evaluated over the target window minus the reference mean R[obs].
    Fit years: up to init, or all years except the target window and reference (`leave_out`). NaN where
    fewer than `min_years` years are available.
    """
    years = np.asarray(years, dtype=float)
    ops = np.full((len(OBS_METHODS), len(init_years), len(windows), len(years)), np.nan)
    for ii, init in enumerate(init_years):
        in_ref = (years > init - reference_length) & (years <= init)
        if in_ref.sum() < reference_length:
            continue
        ref = in_ref / in_ref.sum()
        for iw, (a, b) in enumerate(windows):
            in_win = (years >= init + a) & (years <= init + b)
            keep = years <= init if fit == "up_to_init" else ~(in_win | in_ref)
            if keep.sum() < min_years:
                continue
            t0 = (init + a + init + b) / 2 - years[keep].mean()
            t = years - years[keep].mean()
            const = keep / keep.sum()
            slope = keep * t / (keep * t**2).sum()
            ops[0, ii, iw] = const - ref
            ops[1, ii, iw] = const + t0 * slope - ref
    return ops


def observation_reference(wind: xr.DataArray, init_years, windows, reference_length, fit, min_years) -> dict:
    """predicted full change (target units) of M0c/M0t, {method: (init_year, window, latitude, longitude)}"""
    years = wind.year.values
    ops = observation_reference_operators(years, init_years, windows, reference_length, fit, min_years)
    values = wind.transpose("year", ...).values.reshape(len(years), -1)
    missing = np.isnan(values).astype(float)
    filled = np.nan_to_num(values)
    out = {}
    for im, name in enumerate(OBS_METHODS):
        op = ops[im]
        pred = np.einsum("iwy,yp->iwp", np.nan_to_num(op), filled)
        used = np.einsum("iwy,yp->iwp", (np.nan_to_num(op) != 0).astype(float), missing)
        pred = np.where(np.isnan(op).any(axis=-1)[..., None] | (used > 0), np.nan, pred)
        da = xr.DataArray(pred.reshape(len(init_years), len(windows), *wind.shape[1:]), dims=("init_year", "window", *wind.dims[1:]),
                          coords={"init_year": init_years, "window": [window_label(w) for w in windows], **{d: wind[d] for d in wind.dims[1:]}})
        out[name] = da
    return out


def error_spread(target: xr.DataArray, pred, windows) -> xr.DataArray:
    """leave-out RMS of the errors target - pred over the non-overlapping samples (pred 0: persistence)"""
    return persistence_spread(target - pred, windows)


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
    slope = params["slope"]
    for ensemble, name in [("all", "M2b"), ("analogue", "M3b")]:
        x_mean = index["mean"].sel(ensemble=ensemble, drop=True)
        cov = index["cov"].sel(ensemble=ensemble, drop=True)
        internal = ((slope * x_mean).sum("predictor", skipna=False) + params["intercept"] - params["obs_reference"]) * to_units
        # spread^2 = slope' C slope + sigma^2, C: covariance of the members' predictor window means
        explained = (slope * (cov * slope.rename(predictor="predictor2")).sum("predictor2", skipna=False)).sum("predictor", skipna=False)
        spread = np.sqrt(explained + params["sigma"] ** 2) * to_units
        methods["internal"][name] = (internal, spread)
        methods["full"][name] = (internal + forced_a, spread)

    # references for the value of the index: reversion to the long-term mean without any index information
    # (the internal part has zero mean over the fitting years), and the observed index in the target window
    reversion = -params["obs_reference"] * to_units
    perfect = ((slope * params["index_window"]).sum("predictor", skipna=False) + params["intercept"] - params["obs_reference"]) * to_units
    # M0 spread: placeholder, replaced by the RMS of its own errors once the targets are final (below)
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
            if name != "M2a":
                # M2a on the internal target would be its mean minus the MMEM: zero up to rounding (same weighting)
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
        if cfg["evaluation"].get("m0_spread", "own_errors") == "own_errors":
            mean = methods[kind]["M0"][0].transpose(*target.dims)
            methods[kind]["M0"] = (mean, error_spread(target, mean, windows))
    # observations only (full target): M0 with a constant or a linear trend as forced part
    ocfg = cfg["evaluation"].get("obs_methods") or {}
    if ocfg.get("enabled"):
        preds = observation_reference(wind, inits, windows, ref_len, ocfg.get("fit", "up_to_init"), ocfg.get("min_years", 20))
        target = targets["full"]
        for name, pred in preds.items():
            pred = (pred * to_units).transpose(*target.dims).where(target.notnull())
            methods["full"][name] = (pred, error_spread(target, pred, windows))
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
    if not jobs:   # no sites configured (region.sites: {})
        return xr.Dataset()
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
    if cfg["evaluation"].get("significance", {}).get("versus_m1") and not out.exists("significance_m1.nc"):
        from climate_scenarios_2050.evaluate.versus_m1 import m1_test

        write_netcdf(m1_test(cfg, targets, methods, windows), out.path("significance_m1.nc"), cfg, compress=True)
    from climate_scenarios_2050.plotting.target import plot_target_evaluation
    plot_target_evaluation(cfg, out, scenario, target_variable)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        for target_variable in available_target_variables(cfg):
            run_case(cfg, scenario, target_variable)
