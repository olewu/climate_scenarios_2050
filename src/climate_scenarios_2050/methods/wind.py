"""
Step 5a: what the wind methods need from the CMIP6 ensembles, per scenario
(`<root>/hindcast_wind/<hash>/`). Ensembles: `all` (all members) and `analogue` (selected members).

- `index_stats.nc`: mean and std over the members of the predictor window mean, (ensemble, init_year,
  window). Seasons Y+a-lag ... Y+b-lag for wind window (a, b). Used by M2b/M3b.
- `sfcwind.nc`: mean and std over the members of the relative sfcWind change [%] of the window against each
  member's own reference, (ensemble, init_year, window, lat, lon). Used by M2a/M3a.
- `sfcwind_mmem.nc`: relative change [%] of the sfcWind MMEM, the forced estimate of option (b).

`all` uses the weighting of `cmip6.weighting` (model: every model counts once); analogue members are weighted equally.
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.analogues.generate import hindcast_output, open_hindcast
from climate_scenarios_2050.analogues.matching import align_leads
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.preprocess import cmip6
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)


def wind_hindcast_output(cfg, scenario: str) -> StepOutput:
    return StepOutput(
        cfg, "hindcast_wind", sections=["target"],
        extra={"scenario": scenario, "weighting": cfg["cmip6"]["weighting"], "sfcWind_aggregation": "annual"},
        upstream={"hindcast_nao": hindcast_output(cfg, scenario).hash, "split": split_output(cfg, scenario).hash},
    )


def member_weights(model_ids, weighting: str) -> np.ndarray:
    """weights summing to one"""
    model_ids = np.asarray(model_ids)
    if weighting == "member":
        return np.full(len(model_ids), 1 / len(model_ids))
    _, inverse, counts = np.unique(model_ids, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    return weights / weights.sum()


def weighted_mean_std(values: np.ndarray, weights: np.ndarray):
    """over the first axis, members with NaN are ignored and the weights renormalized"""
    w = weights.reshape(-1, *([1] * (values.ndim - 1)))
    valid = np.isfinite(values)
    w_valid = w * valid
    total = w_valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (w_valid * np.nan_to_num(values)).sum(axis=0) / total
        var = (w_valid * (np.nan_to_num(values) - mean) ** 2).sum(axis=0) / total
    return mean, np.sqrt(var)


def index_stats(cfg, split, hindcast, inits, windows) -> xr.Dataset:
    lag = cfg["target"]["predictor_lag"]
    internal = split.cmip6_internal().transpose("member", "year")
    weights = member_weights(internal["model_id"].values, cfg["cmip6"]["weighting"])
    leads = hindcast.lead.values
    mean = np.full((2, len(inits), len(windows)), np.nan)
    std = np.full_like(mean, np.nan)
    for ii, init in enumerate(inits):
        all_members = align_leads(internal.values, internal.year.values, int(init), leads)
        analogue = hindcast["index"].sel(init_year=init).values
        for iw, (a, b) in enumerate(windows):
            use = (leads >= a - lag) & (leads <= b - lag)
            mean[0, ii, iw], std[0, ii, iw] = weighted_mean_std(_window(all_members, use), weights)
            mean[1, ii, iw], std[1, ii, iw] = weighted_mean_std(_window(analogue, use), np.full(analogue.shape[0], 1 / analogue.shape[0]))
    coords = {"ensemble": ["all", "analogue"], "init_year": inits, "window": [window_label(w) for w in windows]}
    return xr.Dataset({"mean": (tuple(coords), mean), "std": (tuple(coords), std)}, coords=coords)


def _window(values, use):
    """window mean over the second axis, NaN unless complete"""
    sub = values[:, use]
    return np.where(np.isfinite(sub).all(axis=1), np.nanmean(np.where(np.isfinite(sub), sub, 0.0), axis=1), np.nan)


def sfcwind_change(values: np.ndarray, years: np.ndarray, init: int, window, reference_length: int, lags=None, reference_end=None) -> np.ndarray:
    """
    relative change [%] per member (member, lat, lon) of the window against the member's own reference
    (the `reference_length` years ending in `reference_end`, default: the init year)
    """
    reference_end = init if reference_end is None else reference_end
    n = values.shape[0]
    lags = np.zeros(n, dtype=int) if lags is None else lags
    out = np.full((n, *values.shape[2:]), np.nan, dtype="float32")
    for lag in np.unique(lags):
        rows = np.flatnonzero(lags == lag)
        in_win = (years >= init + window[0] + lag) & (years <= init + window[1] + lag)
        in_ref = (years > reference_end - reference_length + lag) & (years <= reference_end + lag)
        if in_win.sum() != window[1] - window[0] + 1 or in_ref.sum() != reference_length:
            continue
        sub = values[rows]
        out[rows] = (sub[:, in_win].mean(axis=1) / sub[:, in_ref].mean(axis=1) - 1) * 100
    return out


def sfcwind_stats(cfg, hindcast, inits, windows, scenario):
    """None if there is no sfcWind for this scenario"""
    try:
        field = cmip6.open_field(cfg, "sfcWind", "annual", scenario)["sfcWind"].load().transpose("member", "year", "lat", "lon")
    except FileNotFoundError:
        return None
    ref = cfg["target"]["reference_length"]
    values, years = field.values, field.year.values
    members = field.member.values.astype(str)
    position = {m: i for i, m in enumerate(members)}
    weights = member_weights(field["model_id"].values, cfg["cmip6"]["weighting"])
    shape = (2, len(inits), len(windows), *values.shape[2:])
    mean, std = np.full(shape, np.nan, dtype="float32"), np.full(shape, np.nan, dtype="float32")
    coverage = np.full(len(inits), np.nan)
    for ii, init in enumerate(inits):
        selected = hindcast["member"].sel(init_year=init).values.astype(str)
        lags = hindcast["lag"].sel(init_year=init).values
        found = np.array([m in position for m in selected])
        coverage[ii] = found.mean()
        rows = np.array([position[m] for m in selected[found]], dtype=int)
        for iw, window in enumerate(windows):
            mean[0, ii, iw], std[0, ii, iw] = weighted_mean_std(sfcwind_change(values, years, int(init), window, ref), weights)
            changes = sfcwind_change(values[rows], years, int(init), window, ref, lags[found])
            mean[1, ii, iw], std[1, ii, iw] = weighted_mean_std(changes, np.full(len(rows), 1 / max(len(rows), 1)))
    coords = {"ensemble": ["all", "analogue"], "init_year": inits, "window": [window_label(w) for w in windows], "lat": field.lat.values, "lon": field.lon.values}
    ds = xr.Dataset({"mean": (tuple(coords), mean), "std": (tuple(coords), std), "analogue_coverage": ("init_year", coverage)}, coords=coords)
    log.info(f"{scenario}: {np.nanmean(coverage):.0%} of the analogue members have sfcWind")
    return ds


def mmem_change(mmem: xr.DataArray, inits, windows, reference_length: int, reference_gap: int = 0) -> xr.DataArray:
    """relative change [%] of the sfcWind MMEM per (init_year, window); the reference ends `reference_gap` years before init"""
    out = []
    for init in inits:
        end = int(init) - reference_gap
        ref = mmem.sel(year=slice(end - reference_length + 1, end)).mean("year")
        out.append(xr.concat([(mmem.sel(year=slice(int(init) + a, int(init) + b)).mean("year") / ref - 1) * 100 for a, b in windows], dim=xr.DataArray([window_label(w) for w in windows], dims="window", name="window")))
    return xr.concat(out, dim=xr.DataArray(inits, dims="init_year", name="init_year")).rename("mmem_change")


def run_scenario(cfg, scenario: str, inits=None):
    out = wind_hindcast_output(cfg, scenario).prepare()
    log.info(f"wind hindcast inputs {scenario} -> {out.dir}")
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    hindcast = open_hindcast(cfg, scenario)
    split = SplitResults(cfg, scenario)
    inits = hindcast.init_year.values if inits is None else inits
    if not out.exists("index_stats.nc"):
        write_netcdf(index_stats(cfg, split, hindcast, inits, windows), out.path("index_stats.nc"), cfg)
    if not out.exists("sfcwind.nc"):
        stats = sfcwind_stats(cfg, hindcast, inits, windows, scenario)
        if stats is not None:
            write_netcdf(stats, out.path("sfcwind.nc"), cfg, compress=True)
    mmem = split.sfcwind_mmem()
    if mmem is not None and not out.exists("sfcwind_mmem.nc"):
        write_netcdf(mmem_change(mmem, inits, windows, cfg["target"]["reference_length"]).to_dataset(), out.path("sfcwind_mmem.nc"), cfg, compress=True)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        run_scenario(cfg, scenario)
