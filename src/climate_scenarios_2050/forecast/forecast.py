"""
Step 6: forecast of the wind change for an arbitrary initialization year (`<root>/forecast/<hash>/`).

Only what is known at init enters: observations up to the init year for the forced-signal coefficients,
the linear model and the analogue selection (the runs' own future is part of the CMIP6 data).

Methods, per scenario, as (mean, spread) of the change of each target window against the observed
reference (the `reference_length` years up to init). If the wind record ends before the init year (the
index can be more recent than the wind data), the reference is the last `reference_length` years of the
record instead (`reference_end`, also selectable explicitly), and all methods use these same years:
  M0   return to the long-term mean (no NAO): internal reference removed, forced part added
  M1   persistence: no change, spread of the past observed changes
  M2a  sfcWind of all members (wind speed only, where sfcWind exists)
  M2b  linear model on the NAO of all members
  M3a  sfcWind of the analogue members (wind speed only, where sfcWind exists)
  M3b  linear model on the NAO of the analogue members
The components of the linear-model methods are stored as well: `internal`, `forced_nao` and `forced_cmip6`
(the latter where sfcWind exists). `forced_wind` selects which forced part enters M0, M2b and M3b.

If a window lies inside the record, the observed change is added (`observed`).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.analogues.preselection import selected_models
from climate_scenarios_2050.core.config import select
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate.wind import to_wind_grid
from climate_scenarios_2050.linear_model.fit import Training, fit_case, sliding_means
from climate_scenarios_2050.methods.wind import member_weights, mmem_change, sfcwind_change, weighted_mean_std
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices.compute import index_output, open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)


def forecast_output(cfg, scenarios, wind_variable, windows) -> StepOutput:
    return StepOutput(
        cfg, "forecast", sections=["analogues", "linear_model", "forced_wind", "split"],
        extra={"scenarios": list(scenarios), "wind_variable": wind_variable, "windows": [list(w) for w in windows], "target": select(cfg, ["target"])["target"] | {"windows": None}, "matching_index": cfg["obs"]["matching_index"], "training_index": cfg["obs"]["training_index"]},
        upstream={"indices": index_output(cfg).hash, "split": {s: split_output(cfg, s).hash for s in scenarios}, "obs_wind": obs.wind_output(cfg, wind_variable).hash},
    )


def internal_up_to_init(series: xr.DataArray, basis: xr.DataArray, init: int, min_fit_years: int, lag: int = 0) -> xr.DataArray:
    """series minus its forced part, with coefficients fitted on the years up to init"""
    years = series.year.values
    design = fc.design_matrix(basis, years, lag=lag)
    coef = fc.fit_coefficients(series, design, fc.up_to_init_exclusions(years, [init]).isel(init_year=0, drop=True), min_fit_years)
    return series - fc.forced_part(design, coef), coef


def analogue_selection(cfg, split, internal: xr.DataArray, init: int):
    """selected members (indices into `internal`) and their lags at the init year"""
    series = open_obs_index(cfg, cfg["obs"]["matching_index"]).dropna("year")
    obs_internal, _ = internal_up_to_init(series, split.basis, init, cfg["split"]["min_fit_years"])
    pool = internal
    if cfg["analogues"]["preselection"] is not None:
        models = selected_models(cfg["analogues"]["preselection"], internal["model"].values, internal, obs_internal, cfg["paths"]["inputs"]["subsampling"])
        pool = internal.isel(member=np.isin(internal["model"].values, models))
    sel = matching.select_members(obs_internal.values, series.year.values, pool.values, pool.year.values, init, cfg["analogues"])
    return pool.member.values[sel["member"]].astype(str), sel["lag"], sel


def past_change_spread(wind: xr.DataArray, window, reference_length: int, init: int, gap: int = 0) -> xr.DataArray:
    """
    root mean square of the observed changes [units of wind] of earlier, complete windows, with the same
    distance (`gap` years) between the end of the reference and the init year
    """
    a, b = window
    changes = []
    for past in range(int(wind.year.min()) + reference_length - 1 + gap, init - b + 1):
        end = past - gap
        changes.append(wind.sel(year=slice(past + a, past + b)).mean("year") - wind.sel(year=slice(end - reference_length + 1, end)).mean("year"))
    if len(changes) < 5:
        return xr.full_like(wind.isel(year=0, drop=True), np.nan)
    return np.sqrt((xr.concat(changes, dim="sample") ** 2).mean("sample"))


def forecast_scenario(cfg, scenario, wind_variable, init, windows, reference_end=None) -> xr.Dataset:
    tcfg = cfg["target"]
    lag, ref_len = tcfg["predictor_lag"], tcfg["reference_length"]
    min_fit = cfg["split"]["min_fit_years"]
    labels = [window_label(w) for w in windows]
    split = SplitResults(cfg, scenario)
    wind = obs.open_obs_wind(cfg, wind_variable).transpose("year", ...)
    last = int(wind.year.max())
    if reference_end is None:
        reference_end = min(init, last)
        if reference_end < init:
            log.warning(f"the {wind_variable} record ends in {last}: reference period {reference_end - ref_len + 1}-{reference_end} instead of the {ref_len} years up to init {init}")
    if reference_end > min(init, last) or reference_end - ref_len + 1 < int(wind.year.min()):
        raise ValueError(f"reference period ending in {reference_end} not available ({wind_variable} record {int(wind.year.min())}-{last}, init {init})")
    gap = init - reference_end
    in_ref = (wind.year >= reference_end - ref_len + 1) & (wind.year <= reference_end)
    wind_ref = wind.isel(year=in_ref.values).mean("year")
    to_units = xr.ones_like(wind_ref) if tcfg["change"] == "absolute" else 100 / wind_ref.where(wind_ref > 0)
    from_percent = wind_ref / 100 if tcfg["change"] == "absolute" else xr.ones_like(wind_ref)

    # members
    internal = split.cmip6_internal().transpose("member", "year")
    weights = member_weights(internal["model_id"].values, cfg["cmip6"]["weighting"])
    selected, lags, sel = analogue_selection(cfg, split, internal, init)
    log.info(f"{scenario}: {len(selected)} analogue members of {sel['n_valid']} valid, best score {sel['score'][0]:.2f}")
    at_selected = internal.sel(member=list(selected))

    # training data: internal index and wind up to init
    training = Training(cfg, scenario, wind_variable)
    index = open_obs_index(cfg, training.index_name).dropna("year")
    index_internal, _ = internal_up_to_init(index, split.basis, init, min_fit)
    wind_internal, wind_coef = internal_up_to_init(wind, split.basis, init, min_fit, lag=lag)
    pair = training.years[training.years <= reference_end]
    x_train = index_internal.sel(year=pair - lag).values
    y_train = wind_internal.sel(year=pair).transpose("year", ...).values.reshape(len(pair), -1)
    internal_ref = wind_internal.isel(year=in_ref.values).mean("year")

    future = np.arange(reference_end - ref_len + 1, init + max(w[1] for w in windows) + 1)
    design = fc.design_matrix(split.basis, future, lag=lag)
    forced_ref = fc.forced_part(design.sel(year=slice(reference_end - ref_len + 1, reference_end)), wind_coef).mean("year")

    # sfcWind
    field = None
    if wind_variable == "wsp100":
        try:
            field = cmip6.open_field(cfg, "sfcWind", "annual", scenario)["sfcWind"].load().transpose("member", "year", "lat", "lon")
        except FileNotFoundError:
            log.info(f"no sfcWind for {scenario}")
    mmem = split.sfcwind_mmem() if field is not None else None

    outputs = []
    for window, label in zip(windows, labels):
        a, b = window
        length = b - a + 1
        slope, intercept, sigma, n = fit_case(x_train, y_train, length, None, cfg["linear_model"]["min_samples"])
        reshape = lambda v: xr.DataArray(v.reshape(wind_ref.shape), dims=wind_ref.dims, coords=wind_ref.coords)
        slope, intercept, sigma = reshape(slope), reshape(intercept), reshape(sigma)

        leads = np.arange(a - lag, b - lag + 1)
        def predictor(members, member_lags=None):
            values = matching.align_leads(members.values, members.year.values, init, leads, member_lags)
            return np.where(np.isfinite(values).all(axis=1), values.mean(axis=1), np.nan)
        all_mean, all_std = weighted_mean_std(predictor(internal), weights)
        ana_mean, ana_std = weighted_mean_std(predictor(at_selected, lags), np.full(len(selected), 1 / len(selected)))

        forced_nao = (fc.forced_part(design.sel(year=slice(init + a, init + b)), wind_coef).mean("year") - forced_ref) * to_units
        forced_cmip = None
        if mmem is not None:
            forced_cmip = to_wind_grid(mmem_change(mmem, [init], [window], ref_len, gap).isel(init_year=0, window=0, drop=True), wind_ref) * from_percent
        forced = {"nao_mmem": forced_nao, "cmip6_sfcwind": forced_cmip, "none": xr.zeros_like(forced_nao)}[cfg["forced_wind"]]
        if forced is None:
            forced = forced_nao

        internal_m0 = -internal_ref * to_units
        # M0 has no predictor: spread of the training window means around their mean
        x_windows = sliding_means(x_train, length)
        spread_m0 = np.sqrt(sigma**2 + (slope * np.nanstd(x_windows)) ** 2) * to_units
        methods = {"M0": (internal_m0, spread_m0, internal_m0)}
        spread_m1 = past_change_spread(wind, window, ref_len, init, gap) * to_units
        methods["M1"] = (xr.zeros_like(wind_ref), spread_m1, None)
        for name, x_mean, x_std in [("M2b", all_mean, all_std), ("M3b", ana_mean, ana_std)]:
            part = (slope * x_mean + intercept - internal_ref) * to_units
            methods[name] = (part, np.sqrt((slope * x_std) ** 2 + sigma**2) * to_units, part)
        if field is not None:
            position = {m: i for i, m in enumerate(field.member.values.astype(str))}
            found = [m in position for m in selected]
            rows = [position[m] for m in selected[found]]
            changes = sfcwind_change(field.values, field.year.values, init, window, ref_len, reference_end=reference_end)
            mean, std = weighted_mean_std(changes, member_weights(field["model_id"].values, cfg["cmip6"]["weighting"]))
            changes_a = sfcwind_change(field.values[rows], field.year.values, init, window, ref_len, np.asarray(lags)[found], reference_end)
            mean_a, std_a = weighted_mean_std(changes_a, np.full(len(rows), 1 / max(len(rows), 1)))
            for name, m, s in [("M2a", mean, std), ("M3a", mean_a, std_a)]:
                grid = {"lat": field.lat, "lon": field.lon}
                on_wind = lambda v: to_wind_grid(xr.DataArray(v, dims=("lat", "lon"), coords=grid), wind_ref) * from_percent
                methods[name] = (on_wind(m), on_wind(s), None)

        names = ["M0", "M1", "M2a", "M2b", "M3a", "M3b"]
        names = [n for n in names if n in methods]
        full = {}
        for name in names:
            mean, spread, part = methods[name]
            full[name] = (mean + forced if part is not None else mean, spread, part if part is not None else xr.full_like(wind_ref, np.nan))
        stack = lambda i: xr.concat([full[n][i].drop_vars("mode", errors="ignore") for n in names], dim=xr.DataArray(names, dims="fc_method", name="fc_method"))
        ds = xr.Dataset({"mean": stack(0), "spread": stack(1), "internal": stack(2), "forced_nao": forced_nao.drop_vars("mode", errors="ignore")})
        if forced_cmip is not None:
            ds["forced_cmip6"] = forced_cmip
        if init + b <= last:
            ds["observed"] = (wind.sel(year=slice(init + a, init + b)).mean("year") - wind_ref) * to_units
        ds["n_training_windows"] = n
        outputs.append(ds.expand_dims(window=[label]))

    result = xr.concat(outputs, dim="window").expand_dims(scenario=[scenario])
    result["reference_wind"] = wind_ref
    result.attrs.update({"init_year": init, "units": "%" if tcfg["change"] == "relative" else "units of wind", "reference_period": f"{reference_end - ref_len + 1}-{reference_end}", "n_analogues": len(selected)})
    return result


def site_table(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    rows = []
    for scenario in point.scenario.values:
        for window in point.window.values:
            sub = point.sel(scenario=scenario, window=window)
            for name in sub.fc_method.values:
                row = {"scenario": scenario, "window": window, "fc_method": name, "mean": float(sub["mean"].sel(fc_method=name)), "spread": float(sub["spread"].sel(fc_method=name))}
                if "observed" in sub:
                    row["observed"] = float(sub["observed"])
                rows.append(row)
    return pd.DataFrame(rows)


def run(cfg, init: int, scenarios=None, wind_variable=None, windows=None, site=None, reference_wind=None, reference_end=None):
    wind_variable = wind_variable or cfg["obs"]["wind_variable"]
    windows = [tuple(w) for w in (windows or cfg["target"]["windows"])]
    available = [s for s in (scenarios or cfg["cmip6"]["scenarios"]) if split_output(cfg, s).exists("basis.nc")]
    out = forecast_output(cfg, available, wind_variable, windows).prepare()
    log.info(f"forecast init {init} -> {out.dir}")
    tag = "" if reference_end is None else f"_ref{reference_end}"
    fl = out.path(f"forecast_init{init}{tag}.nc")
    if fl.exists():
        log.info(f"{fl} exists, reading it")
        with xr.open_dataset(fl) as stored:
            ds = stored.load()
    else:
        parts = [forecast_scenario(cfg, s, wind_variable, init, windows, reference_end) for s in available]
        reference = parts[0]["reference_wind"]
        attrs = parts[0].attrs
        # scenarios without sfcWind lack M2a/M3a and the CMIP6 forced part: NaN there
        names = [m for m in ["M0", "M1", "M2a", "M2b", "M3a", "M3b"] if any(m in p.fc_method.values for p in parts)]
        parts = [p.drop_vars("reference_wind").reindex(fc_method=names) for p in parts]
        for ii, part in enumerate(parts):
            if "forced_cmip6" not in part and any("forced_cmip6" in q for q in parts):
                parts[ii] = part.assign(forced_cmip6=xr.full_like(part["forced_nao"], np.nan))
        ds = xr.concat(parts, dim="scenario")
        ds["reference_wind"] = reference
        ds.attrs = attrs
        write_netcdf(ds, fl, cfg, compress=True)
    if site is None:
        return out, ds, None
    table = site_table(ds, *site)
    if reference_wind is not None:
        if ds.attrs["units"] != "%":
            raise ValueError("a measured reference wind needs relative changes (target.change: relative)")
        for col in [c for c in ["mean", "spread", "observed"] if c in table]:
            table[f"{col}_abs"] = table[col] * reference_wind / 100
    name = f"site_{site[0]:.2f}N_{site[1]:.2f}E_init{init}{tag}"
    table.to_csv(out.path(f"{name}.csv"), index=False)
    log.info(f"wrote {out.path(name + '.csv')}")
    from climate_scenarios_2050.plotting.forecast import plot_site
    plot_site(cfg, out, table, name, ds.attrs)
    return out, ds, table
