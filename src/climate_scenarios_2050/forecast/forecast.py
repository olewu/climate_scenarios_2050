"""
Step 6: forecast of the change of a target for an arbitrary initialization year (`<root>/forecast/<hash>/`).

Only what is known at init enters: observations up to the init year for the forced-signal coefficients,
the linear model and the analogue selection (the runs' own future is part of the CMIP6 data).

Methods, per scenario, as (mean, spread) of the change of each target window against the observed
reference (the `reference_length` years up to init), relative [%] or absolute as the target's `change`.
If the target record ends before the init year (the index can be more recent than the target data), the reference is the last `reference_length` years of the
record instead (`reference_end`, also selectable explicitly), and all methods use these same years:
  M0   return to the long-term mean (no index information): internal reference removed, forced part added
  M1   persistence: no change, spread of the past observed changes
  M2a  the target's CMIP6 variable in all members (e.g. sfcWind for wsp100; where it exists)
  M2b  linear model on the index of all members
  M3a  the target's CMIP6 variable in the analogue members
  M3b  linear model on the index of the analogue members
The components of the linear-model methods are stored as well: `internal`, `forced_index` and `forced_cmip6`
(the latter where the CMIP6 variable exists). `forced_target` selects which forced part enters M0, M2b and M3b.

If a window lies inside the record, the observed change is added (`observed`).

Member values are stored per scenario in `members_init<year>[_ref<year>]_<scenario>.nc` (distributions
of the site figure): the predictor window mean of every member (`x_all`, with weights) and of the
analogue members (`x_analogue`), slope and residual spread of the linear model in the units of the
change (`slope_u`, `sigma_u`), and the CMIP6 change of every member on the CMIP6 grid (`sfc_change`). M2b/M3b are
then the mixture of N(mean + slope_u (x_m - weighted mean of x), sigma_u) over the members.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.analogues.preselection import selected_models
from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import select, target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate.target import to_target_grid
from climate_scenarios_2050.linear_model.fit import Training, fit_case, sliding_means
from climate_scenarios_2050.methods.target import CMIP6_AGGREGATION, cmip6_mmem, member_change, member_weights, mmem_change, open_cmip6_target, weighted_mean_std
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices.compute import index_output, open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)


def forecast_output(cfg, scenarios, target_variable, windows) -> StepOutput:
    spec = target_spec(cfg, target_variable)
    upstream = {"indices": index_output(cfg).hash, "split": {s: split_output(cfg, s).hash for s in scenarios}, "obs_target": obs.target_output(cfg, target_variable).hash}
    if spec.get("cmip6"):
        upstream["cmip6_fields"] = {s: cmip6.field_id(cfg, spec["cmip6"], CMIP6_AGGREGATION, s) for s in scenarios}
    return StepOutput(
        cfg, "forecast", sections=["analogues", "linear_model", "forced_target", "split"],
        extra={"scenarios": list(scenarios), "target_variable": target_variable, "spec": spec, "windows": [list(w) for w in windows], "target": select(cfg, ["target"])["target"] | {"windows": None}, "matching_index": cfg["obs"]["matching_index"], "training_index": cfg["obs"]["training_index"]},
        upstream=upstream,
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
    root mean square of the observed changes [units of the target] of earlier, complete windows, with the same
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


def forecast_scenario(cfg, scenario, target_variable, init, windows, reference_end=None) -> xr.Dataset:
    tcfg = cfg["target"]
    spec = target_spec(cfg, target_variable)
    relative = spec["change"] == "relative"
    lag, ref_len = tcfg["predictor_lag"], tcfg["reference_length"]
    min_fit = cfg["split"]["min_fit_years"]
    labels = [window_label(w) for w in windows]
    split = SplitResults(cfg, scenario)
    wind = obs.open_obs_target(cfg, target_variable).transpose("year", ...)
    last = int(wind.year.max())
    if reference_end is None:
        reference_end = min(init, last)
        if reference_end < init:
            log.warning(f"the {target_variable} record ends in {last}: reference period {reference_end - ref_len + 1}-{reference_end} instead of the {ref_len} years up to init {init}")
    if reference_end > min(init, last) or reference_end - ref_len + 1 < int(wind.year.min()):
        raise ValueError(f"reference period ending in {reference_end} not available ({target_variable} record {int(wind.year.min())}-{last}, init {init})")
    gap = init - reference_end
    in_ref = (wind.year >= reference_end - ref_len + 1) & (wind.year <= reference_end)
    wind_ref = wind.isel(year=in_ref.values).mean("year")
    to_units = 100 / wind_ref.where(wind_ref > 0) if relative else xr.ones_like(wind_ref)

    # members
    internal = split.cmip6_internal().transpose("member", "year")
    weights = member_weights(internal["model_id"].values, cfg["cmip6"]["weighting"])
    selected, lags, sel = analogue_selection(cfg, split, internal, init)
    log.info(f"{scenario}: {len(selected)} analogue members of {sel['n_valid']} valid, best score {sel['score'][0]:.2f}")
    at_selected = internal.sel(member=list(selected))

    # training data: internal index and wind up to init
    training = Training(cfg, scenario, target_variable)
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

    # the target's CMIP6 variable (M2a/M3a, forced part b)
    field = open_cmip6_target(cfg, spec["cmip6"], scenario) if spec.get("cmip6") else None
    mmem = cmip6_mmem(field, cfg["cmip6"]["weighting"]) if field is not None else None

    outputs, member_parts = [], []
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
        x_all, x_ana = predictor(internal), predictor(at_selected, lags)
        all_mean, all_std = weighted_mean_std(x_all, weights)
        ana_mean, ana_std = weighted_mean_std(x_ana, np.full(len(selected), 1 / len(selected)))
        members = xr.Dataset({
            "x_all": ("member", x_all), "x_analogue": ("analogue", x_ana),
            "slope_u": (slope * to_units).astype("float32"), "sigma_u": (sigma * to_units).astype("float32"),
        })

        forced_index = (fc.forced_part(design.sel(year=slice(init + a, init + b)), wind_coef).mean("year") - forced_ref) * to_units
        forced_cmip = None
        if mmem is not None:
            forced_cmip = to_target_grid(mmem_change(mmem, [init], [window], ref_len, gap, relative).isel(init_year=0, window=0, drop=True), wind_ref)
        forced = {"index_mmem": forced_index, "cmip6_mmem": forced_cmip, "none": xr.zeros_like(forced_index)}[cfg["forced_target"]]
        if forced is None:
            forced = forced_index

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
            changes = member_change(field.values, field.year.values, init, window, ref_len, reference_end=reference_end, relative=relative)
            mean, std = weighted_mean_std(changes, member_weights(field["model_id"].values, cfg["cmip6"]["weighting"]))
            changes_a = member_change(field.values[rows], field.year.values, init, window, ref_len, np.asarray(lags)[found], reference_end, relative)
            mean_a, std_a = weighted_mean_std(changes_a, np.full(len(rows), 1 / max(len(rows), 1)))
            members["sfc_change"] = (("sfc_member", "lat", "lon"), changes)
            members["sfc_change_analogue"] = (("sfc_analogue", "lat", "lon"), changes_a)
            for name, m, s in [("M2a", mean, std), ("M3a", mean_a, std_a)]:
                grid = {"lat": field.lat, "lon": field.lon}
                on_wind = lambda v: to_target_grid(xr.DataArray(v, dims=("lat", "lon"), coords=grid), wind_ref)
                methods[name] = (on_wind(m), on_wind(s), None)

        names = ["M0", "M1", "M2a", "M2b", "M3a", "M3b"]
        names = [n for n in names if n in methods]
        full = {}
        for name in names:
            mean, spread, part = methods[name]
            full[name] = (mean + forced if part is not None else mean, spread, part if part is not None else xr.full_like(wind_ref, np.nan))
        stack = lambda i: xr.concat([full[n][i].drop_vars("mode", errors="ignore") for n in names], dim=xr.DataArray(names, dims="fc_method", name="fc_method"))
        ds = xr.Dataset({"mean": stack(0), "spread": stack(1), "internal": stack(2), "forced_index": forced_index.drop_vars("mode", errors="ignore")})
        if forced_cmip is not None:
            ds["forced_cmip6"] = forced_cmip
        if init + b <= last:
            ds["observed"] = (wind.sel(year=slice(init + a, init + b)).mean("year") - wind_ref) * to_units
        ds["n_training_windows"] = n
        outputs.append(ds.expand_dims(window=[label]))
        member_parts.append(members.drop_vars("mode", errors="ignore").expand_dims(window=[label]))

    result = xr.concat(outputs, dim="window").expand_dims(scenario=[scenario])
    result["reference_value"] = wind_ref
    result.attrs.update({"init_year": init, "target_variable": target_variable, "units": "%" if relative else wind.attrs.get("units", "units of the target"), "reference_period": f"{reference_end - ref_len + 1}-{reference_end}", "n_analogues": len(selected)})

    members = xr.concat(member_parts, dim="window")
    members = members.assign_coords(member=internal.member.values.astype(str), analogue=np.asarray(selected).astype(str))
    members["weight_all"] = ("member", weights)
    members["analogue_lag"] = ("analogue", np.asarray(lags))
    if field is not None:
        members = members.assign_coords(sfc_member=field.member.values.astype(str), sfc_analogue=np.asarray(selected)[found].astype(str), lat=field.lat.values, lon=field.lon.values)
        members["sfc_weight"] = ("sfc_member", member_weights(field["model_id"].values, cfg["cmip6"]["weighting"]))
        members.attrs["sfc_units"] = "%" if relative else field.attrs.get("units", "")
    members.attrs.update({"init_year": init, "scenario": scenario, "change": spec["change"]})
    return result, members


#---------------distributions at a site--------------#
def _kde_bandwidth(values, weights):
    """Scott's rule with the effective number of members"""
    w = weights / weights.sum()
    mean = (w * values).sum()
    std = np.sqrt((w * (values - mean) ** 2).sum())
    n_eff = 1 / (w**2).sum()
    return max(1.06 * std * n_eff ** (-1 / 5), 1e-6)


def site_components(point: xr.Dataset, members, name: str, reference_value: float):
    """
    predictive distribution of one method at one site as a Gaussian mixture (mu, sigma, weight);
    `point`: the forecast at the site for one scenario and window, `members`: the member values
    (same scenario and window) or None
    """
    mean, spread = float(point["mean"].sel(fc_method=name)), float(point["spread"].sel(fc_method=name))
    single = (np.array([mean]), np.array([spread]), np.array([1.0]))
    if members is None or not np.isfinite(mean):
        return single
    site = {"latitude": float(point.latitude), "longitude": float(point.longitude)}
    if name in ("M2b", "M3b"):
        x = members["x_all" if name == "M2b" else "x_analogue"].values
        w = members["weight_all"].values if name == "M2b" else np.ones(len(x))
        ok = np.isfinite(x)
        x, w = x[ok], w[ok] / w[ok].sum()
        slope = float(members["slope_u"].sel(site, method="nearest"))
        sigma = float(members["sigma_u"].sel(site, method="nearest"))
        return mean + slope * (x - (w * x).sum()), np.full(len(x), sigma), w
    if name in ("M2a", "M3a") and "sfc_change" in members:
        var = "sfc_change" if name == "M2a" else "sfc_change_analogue"
        values = members[var].interp(lat=site["latitude"], lon=site["longitude"]).values
        w = members["sfc_weight"].values if name == "M2a" else np.ones(len(values))
        ok = np.isfinite(values)
        values, w = values[ok], w[ok] / w[ok].sum()
        # variance-preserving kernel density: members shrunk towards the mean so that the
        # mixture has the members' weighted variance (= the tabulated spread)
        h = _kde_bandwidth(values, w)
        center = (w * values).sum()
        var = (w * (values - center) ** 2).sum()
        shrink = np.sqrt(max(1 - h**2 / var, 0.0)) if var > 0 else 1.0
        return center + shrink * (values - center), np.full(len(values), h), w
    return single


def mixture_quantiles(mu, sigma, w, quantiles=(0.05, 0.5, 0.95)):
    from scipy.special import ndtr

    grid = np.linspace((mu - 5 * sigma).min(), (mu + 5 * sigma).max(), 4001)
    cdf = (w[:, None] * ndtr((grid[None, :] - mu[:, None]) / sigma[:, None])).sum(axis=0)
    return [float(np.interp(q, cdf, grid)) for q in quantiles]


def site_table(ds: xr.Dataset, lat: float, lon: float, members=None):
    """table of the forecasts at a site and the mixture components of each (scenario, window, method)"""
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    rows, components = [], {}
    for scenario in point.scenario.values:
        mem = None if members is None else members.get(str(scenario))
        for window in point.window.values:
            sub = point.sel(scenario=scenario, window=window)
            reference_value = float(sub["reference_value"]) if "reference_value" in sub else np.nan
            for name in sub.fc_method.values:
                row = {"scenario": scenario, "window": window, "fc_method": name, "mean": float(sub["mean"].sel(fc_method=name)), "spread": float(sub["spread"].sel(fc_method=name))}
                if np.isfinite(row["mean"]):
                    comp = site_components(sub, None if mem is None else mem.sel(window=window), str(name), reference_value)
                    components[(str(scenario), str(window), str(name))] = comp
                    row.update(dict(zip(["q05", "q50", "q95"], mixture_quantiles(*comp))))
                if "observed" in sub:
                    row["observed"] = float(sub["observed"])
                rows.append(row)
    return pd.DataFrame(rows), components


def run(cfg, init: int, scenarios=None, target_variable=None, windows=None, site=None, reference_value=None, reference_end=None):
    target_variable = target_variable or cfg["obs"]["target_variable"]
    windows = [tuple(w) for w in (windows or cfg["target"]["windows"])]
    available = [s for s in (scenarios or cfg["cmip6"]["scenarios"]) if split_output(cfg, s).exists("basis.nc")]
    out = forecast_output(cfg, available, target_variable, windows).prepare()
    log.info(f"forecast init {init} -> {out.dir}")
    tag = "" if reference_end is None else f"_ref{reference_end}"
    fl = out.path(f"forecast_init{init}{tag}.nc")
    member_files = {s: out.path(f"members_init{init}{tag}_{s}.nc") for s in available}
    missing_members = [s for s in available if not member_files[s].exists()]
    # an existing forecast is read, never rewritten; member files of older forecasts are added
    computed = {}
    for scenario in (available if not fl.exists() else missing_members):
        computed[scenario] = forecast_scenario(cfg, scenario, target_variable, init, windows, reference_end)
        if not member_files[scenario].exists():
            write_netcdf(computed[scenario][1], member_files[scenario], cfg, compress=True)
    if fl.exists():
        log.info(f"{fl} exists, reading it")
        with xr.open_dataset(fl) as stored:
            ds = stored.load()
    else:
        parts = [computed[s][0] for s in available]
        reference = parts[0]["reference_value"]
        attrs = parts[0].attrs
        # scenarios without the CMIP6 variable lack M2a/M3a and the CMIP6 forced part: NaN there
        names = [m for m in ["M0", "M1", "M2a", "M2b", "M3a", "M3b"] if any(m in p.fc_method.values for p in parts)]
        parts = [p.drop_vars("reference_value").reindex(fc_method=names) for p in parts]
        for ii, part in enumerate(parts):
            if "forced_cmip6" not in part and any("forced_cmip6" in q for q in parts):
                parts[ii] = part.assign(forced_cmip6=xr.full_like(part["forced_index"], np.nan))
        ds = xr.concat(parts, dim="scenario")
        ds["reference_value"] = reference
        ds.attrs = attrs
        write_netcdf(ds, fl, cfg, compress=True)
    if site is None:
        return out, ds, None
    members = {s: xr.open_dataset(member_files[s]) for s in available}
    site = (site[0], dom.point_lon(site[1], cfg["region"]["domain"]))
    table, components = site_table(ds, *site, members=members)
    if reference_value is not None:
        if ds.attrs["units"] != "%":
            raise ValueError("--reference-value converts relative changes to absolute ones; this target has absolute changes already")
        for col in [c for c in ["mean", "spread", "q05", "q50", "q95", "observed"] if c in table]:
            table[f"{col}_abs"] = table[col] * reference_value / 100
    name = f"site_{site[0]:.2f}N_{site[1]:.2f}E_init{init}{tag}"
    table.to_csv(out.path(f"{name}.csv"), index=False)
    log.info(f"wrote {out.path(name + '.csv')}")
    from climate_scenarios_2050.plotting.forecast import plot_site
    plot_site(cfg, out, table, components, name, ds.attrs)
    return out, ds, table
