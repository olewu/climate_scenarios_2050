"""
Step 6: forecast of the change of a target for an arbitrary initialization year (`<root>/forecast/<hash>/`).

Only what is known at init enters: observations up to the init year for the forced-signal coefficients,
the linear model and the analogue selection (the runs' own future is part of the CMIP6 data).

Methods, per scenario, as (mean, spread) of the change of each target window against the observed
reference (the `reference_length` years up to init), relative [%] or absolute as the target's `change`.
If the target record ends before the init year (the index can be more recent than the target data), the reference is the last `reference_length` years of the
record instead (`reference_end`, also selectable explicitly), and all methods use these same years:
  M0   return to the long-term mean (no index information): internal reference removed, forced part added;
       spread: RMS of its errors at earlier, complete windows (= their internal window means), as M0c/M0t
  M1   persistence: no change, spread of the past observed changes
  M0c  observations only (`evaluation.obs_methods`): return to the mean of the observed target up to the reference end
  M0t  observations only: linear trend fitted to the observed target up to the reference end, read at the target
       window, minus the reference mean; M0c/M0t spreads: RMS of their errors at earlier, complete windows (same
       gap between reference end and init), each predicted from the years before it
  M2a  the target's CMIP6 variable in all members (e.g. sfcWind for wsp100; where it exists)
  M2b  linear model on the index of all members
  M3a  the target's CMIP6 variable in the analogue members
  M3b  linear model on the index of the analogue members
The components of the linear-model methods are stored as well: `internal`, `forced_index` (the forced part of
the target's split: index MMEM regression, or the scaled MMEM of its CMIP6 variable with `targets.<name>.forced:
cmip6`) and `forced_cmip6` (the plain CMIP6 MMEM change, where the CMIP6 variable exists). `forced_target` selects which forced part enters M0, M2b and M3b.

If a window lies inside the record, the observed change is added (`observed`).

Member values are stored per scenario in `members_init<year>[_ref<year>]_<scenario>.nc` (distributions
of the site figure): the predictor window mean of every member (`x_all`, with weights) and of the
analogue members (`x_analogue`), (member, predictor), slopes and residual spread of the linear model
in the units of the change (`slope_u` (predictor, ...), `sigma_u`), and the CMIP6 change of every member on the CMIP6 grid (`sfc_change`). M2b/M3b are
then the mixture of N(mean + slope_u . (x_m - weighted mean of x), sigma_u) over the members. With several
indices (`predictor.indices`) the analogues are matched jointly (`analogues.match_weights`) and the linear
model has several predictors (`linear_model.predictors`).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.analogues.preselection import selected_models
from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import hindcast_indices, lm_predictors, match_weights, primary_index, select, target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate.target import to_target_grid
from climate_scenarios_2050.linear_model.fit import Training, fit_case, sliding_means
from climate_scenarios_2050.methods.target import CMIP6_AGGREGATION, cmip6_mmem, member_change, member_weights, mmem_change, open_cmip6_target, weighted_mean_cov, weighted_mean_std
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices.compute import index_output, open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)

METHODS = ["M0", "M1", "M0c", "M0t", "M2a", "M2b", "M3a", "M3b"]


def forecast_output(cfg, scenarios, target_variable, windows) -> StepOutput:
    spec = target_spec(cfg, target_variable)
    extra_methods = {}
    ocfg = cfg["evaluation"].get("obs_methods") or {}
    if ocfg.get("enabled"):   # only recorded when used, so other settings keep their hashes
        extra_methods = {"obs_methods": {"min_years": ocfg.get("min_years", 20)}}
    if cfg["evaluation"].get("m0_spread", "own_errors") == "own_errors":
        extra_methods["m0_spread"] = "own_errors"
    upstream = {"indices": index_output(cfg).hash, "split": {s: split_output(cfg, s).hash for s in scenarios}, "obs_target": obs.target_output(cfg, target_variable).hash}
    names = hindcast_indices(cfg)
    if names != [primary_index(cfg)]:
        upstream["indices"] = {n: index_output(cfg, n).hash for n in names}
        upstream["split"] = {s: {n: split_output(cfg, s, n).hash for n in names} for s in scenarios}
    if spec.get("cmip6"):
        upstream["cmip6_fields"] = {s: cmip6.field_id(cfg, spec["cmip6"], CMIP6_AGGREGATION, s) for s in scenarios}
    return StepOutput(
        cfg, "forecast", sections=["analogues", "linear_model", "forced_target", "split"],
        extra={"scenarios": list(scenarios), "target_variable": target_variable, "spec": spec, "windows": [list(w) for w in windows], "target": select(cfg, ["target"])["target"] | {"windows": None}, "matching_index": cfg["obs"]["matching_index"], "training_index": cfg["obs"]["training_index"], **extra_methods},
        upstream=upstream,
    )


def internal_up_to_init(series: xr.DataArray, basis: xr.DataArray, init: int, min_fit_years: int, lag: int = 0, design: xr.DataArray = None) -> xr.DataArray:
    """series minus its forced part, with coefficients fitted on the years up to init (`design`: instead of the basis)"""
    years = series.year.values
    design = fc.design_matrix(basis, years, lag=lag) if design is None else design
    coef = fc.fit_coefficients(series, design, fc.up_to_init_exclusions(years, [init]).isel(init_year=0, drop=True), min_fit_years)
    return series - fc.forced_part(design, coef), coef


def member_pool(cfg, scenario: str):
    """
    internal CMIP6 series (index, member, year) of the indices in `hindcast_indices` (primary, matched,
    predictors) for the members that have all of them, the primary index's internal series, and the splits
    """
    names = hindcast_indices(cfg)
    splits = {n: SplitResults(cfg, scenario, n) for n in names}
    internals = {n: s.cmip6_internal().transpose("member", "year") for n, s in splits.items()}
    common = sorted(set.intersection(*[set(i.member.values.astype(str)) for i in internals.values()]))
    first = internals[names[0]].sel(member=common)
    stack = np.stack([internals[n].sel(member=common).reindex(year=first.year).values for n in names])
    return names, stack, first, splits


def analogue_selection(cfg, names, stack, first, splits, init: int):
    """selected members (names) and their lags at the init year, matched jointly on `analogues.match_weights`"""
    weights = match_weights(cfg)
    matched = list(weights)
    min_fit = cfg["split"]["min_fit_years"]
    obs_internal = {}
    for n in matched:
        series = open_obs_index(cfg, cfg["obs"]["matching_index"], n).dropna("year")
        obs_internal[n], _ = internal_up_to_init(series, splits[n].basis, init, min_fit)
    obs_years = obs_internal[matched[0]].year.values
    for n in matched[1:]:
        obs_years = np.intersect1d(obs_years, obs_internal[n].year.values)
    rows = np.ones(first.sizes["member"], dtype=bool)
    if cfg["analogues"]["preselection"] is not None:
        primary = primary_index(cfg)
        models = selected_models(cfg["analogues"]["preselection"], first["model"].values, first, obs_internal.get(primary), cfg["paths"]["inputs"]["subsampling"])
        rows = np.isin(first["model"].values, models)
    model = np.stack([stack[names.index(n)][rows] for n in matched])
    obs_values = np.stack([obs_internal[n].sel(year=obs_years).values for n in matched])
    sel = matching.select_members(obs_values, obs_years, model, first.year.values, init, cfg["analogues"], np.array([weights[n] for n in matched]))
    return first.member.values[rows][sel["member"]].astype(str), sel["lag"], sel


def quadratic(slope: xr.DataArray, cov: xr.DataArray) -> xr.DataArray:
    """slope' C slope over the predictors (slope: predictor, ...; cov: predictor, predictor2)"""
    cov = cov.assign_coords(predictor=slope.predictor.values, predictor2=slope.predictor.values)
    return (slope * (cov * slope.rename(predictor="predictor2")).sum("predictor2")).sum("predictor")


def past_change_spread(wind: xr.DataArray, window, reference_length: int, init: int, gap: int = 0, reference: bool = True) -> xr.DataArray:
    """
    root mean square of the observed changes [units of the target] of earlier, complete windows, with the same
    distance (`gap` years) between the end of the reference and the init year; `reference=False`: of the window
    means themselves (e.g. of the internal series: the errors of M0)
    """
    a, b = window
    changes = []
    for past in range(int(wind.year.min()) + reference_length - 1 + gap, init - b + 1):
        end = past - gap
        win = wind.sel(year=slice(past + a, past + b))
        if win.sizes["year"] < b - a + 1 or not bool(win.notnull().any()):
            continue
        ref = wind.sel(year=slice(end - reference_length + 1, end)).mean("year") if reference else 0
        changes.append(win.mean("year") - ref)
    if len(changes) < 5:
        return xr.full_like(wind.isel(year=0, drop=True), np.nan)
    return np.sqrt((xr.concat(changes, dim="sample") ** 2).mean("sample"))


def obs_reference_forecast(wind: xr.DataArray, window, reference_length: int, reference_end: int, init: int, min_years: int):
    """
    M0c/M0t (observations only) for one window, in units of the target: {name: (change, spread)} over space.
    Fit on the years up to `reference_end`; spread: RMS of the same predictions for earlier, complete windows
    (reference ending `init - reference_end` years before their own init), each fitted on the years before it.
    """
    a, b = window
    years = wind.year.values
    values = wind.transpose("year", ...).values.reshape(len(years), -1)
    space = wind.isel(year=0, drop=True)

    def predict(end, start):
        """(M0c, M0t) change predictions for a reference ending in `end` and a window starting `start` years after it"""
        keep = years <= end
        if keep.sum() < min_years:
            return None
        y = values[keep]
        t = years[keep] - years[keep].mean()
        const = y.mean(axis=0)
        slope = (t[:, None] * (y - const)).sum(axis=0) / (t**2).sum()
        ref = values[(years > end - reference_length) & (years <= end)].mean(axis=0)
        t_win = (end + start + end + start + (b - a)) / 2 - years[keep].mean()
        return const - ref, const + slope * t_win - ref

    gap = init - reference_end
    current = predict(reference_end, gap + a)
    if current is None:
        return {}
    errors = []
    for past in range(int(years.min()) + reference_length - 1 + gap, init - b + 1):
        end = past - gap
        pred = predict(end, gap + a)
        if pred is None:
            continue
        in_win = (years >= past + a) & (years <= past + b)
        if in_win.sum() < b - a + 1:
            continue
        change = values[in_win].mean(axis=0) - values[(years > end - reference_length) & (years <= end)].mean(axis=0)
        errors.append([change - pred[0], change - pred[1]])
    out = {}
    for k, name in enumerate(("M0c", "M0t")):
        spread = np.sqrt(np.mean([e[k] ** 2 for e in errors], axis=0)) if len(errors) >= 5 else np.full(values.shape[1], np.nan)
        wrap = lambda v: xr.DataArray(v.reshape(space.shape), dims=space.dims, coords=space.coords)
        out[name] = (wrap(current[k]), wrap(spread))
    return out


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

    # members: pool of all members with every needed index, analogues matched jointly
    names, stack, first, splits = member_pool(cfg, scenario)
    weights = member_weights(first["model_id"].values, cfg["cmip6"]["weighting"])
    selected, lags, sel = analogue_selection(cfg, names, stack, first, splits, init)
    log.info(f"{scenario}: {len(selected)} analogue members of {sel['n_valid']} valid, best score {sel['score'][0]:.2f}")
    predictors = lm_predictors(cfg)
    member_rows = {m: i for i, m in enumerate(first.member.values.astype(str))}
    selected_rows = np.array([member_rows[m] for m in selected], dtype=int)
    pred_stack = np.stack([stack[names.index(n)] for n in predictors])          # (predictor, member, year)
    member_years = first.year.values

    # training data: internal indices and target up to init
    training = Training(cfg, scenario, target_variable)
    wind_internal, wind_coef = internal_up_to_init(wind, split.basis, init, min_fit, lag=lag, design=split.target_design(target_variable, wind.year.values))
    pair = training.years[training.years <= reference_end]
    x_train = np.stack([
        internal_up_to_init(open_obs_index(cfg, training.index_name, n).dropna("year"), splits[n].basis, init, min_fit)[0].sel(year=pair - lag).values
        for n in predictors
    ], axis=1)
    y_train = wind_internal.sel(year=pair).transpose("year", ...).values.reshape(len(pair), -1)
    internal_ref = wind_internal.isel(year=in_ref.values).mean("year")

    future = np.arange(reference_end - ref_len + 1, init + max(w[1] for w in windows) + 1)
    design = split.target_design(target_variable, future)
    forced_ref = fc.forced_part(design.sel(year=slice(reference_end - ref_len + 1, reference_end)), wind_coef).mean("year")

    # the target's CMIP6 variable (M2a/M3a, forced part b)
    field = open_cmip6_target(cfg, spec["cmip6"], scenario) if spec.get("cmip6") else None
    mmem = cmip6_mmem(field, cfg["cmip6"]["weighting"]) if field is not None else None

    outputs, member_parts = [], []
    for window, label in zip(windows, labels):
        a, b = window
        length = b - a + 1
        slope, intercept, sigma, n = fit_case(x_train, y_train, length, None, cfg["linear_model"]["min_samples"], float(cfg["linear_model"].get("ridge") or 0.0))
        reshape = lambda v: xr.DataArray(v.reshape(wind_ref.shape), dims=wind_ref.dims, coords=wind_ref.coords)
        slope = xr.DataArray(slope.reshape(len(predictors), *wind_ref.shape), dims=("predictor", *wind_ref.dims), coords={"predictor": predictors, **wind_ref.coords})
        intercept, sigma = reshape(intercept), reshape(sigma)

        leads = np.arange(a - lag, b - lag + 1)
        def predictor_means(rows=slice(None), member_lags=None):
            """window means (member, predictor)"""
            out = []
            for k in range(len(predictors)):
                values = matching.align_leads(pred_stack[k][rows], member_years, init, leads, member_lags)
                out.append(np.where(np.isfinite(values).all(axis=1), values.mean(axis=1), np.nan))
            return np.stack(out, axis=1)
        x_all, x_ana = predictor_means(), predictor_means(selected_rows, lags)
        all_mean, all_cov = weighted_mean_cov(x_all, weights)
        ana_mean, ana_cov = weighted_mean_cov(x_ana, np.full(len(selected), 1 / len(selected)))
        members = xr.Dataset({
            "x_all": (("member", "predictor"), x_all), "x_analogue": (("analogue", "predictor"), x_ana),
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
        if cfg["evaluation"].get("m0_spread", "own_errors") == "own_errors":
            # RMS of M0's errors at earlier, complete windows: its error is the internal window mean (the prediction is
            # internal = 0, the forced part cancels on the full target)
            spread_m0 = past_change_spread(wind_internal, window, ref_len, init, gap, reference=False) * to_units
        else:
            # from the linear model: spread of the predicted training window means around their mean plus the residual
            x_windows = sliding_means(x_train, length)
            x_windows = x_windows[np.isfinite(x_windows).all(axis=1)]
            explained_var = xr.DataArray(np.cov(x_windows.T, bias=True).reshape(len(predictors), len(predictors)), dims=("predictor", "predictor2"))
            spread_m0 = np.sqrt(sigma**2 + quadratic(slope, explained_var)) * to_units
        methods = {"M0": (internal_m0, spread_m0, internal_m0)}
        spread_m1 = past_change_spread(wind, window, ref_len, init, gap) * to_units
        methods["M1"] = (xr.zeros_like(wind_ref), spread_m1, None)
        ocfg = cfg["evaluation"].get("obs_methods") or {}
        if ocfg.get("enabled"):
            for name, (change, spread) in obs_reference_forecast(wind, window, ref_len, reference_end, init, ocfg.get("min_years", 20)).items():
                methods[name] = (change * to_units, spread * to_units, None)
        for name, x_mean, x_cov in [("M2b", all_mean, all_cov), ("M3b", ana_mean, ana_cov)]:
            x_mean = xr.DataArray(x_mean, dims="predictor", coords={"predictor": predictors})
            x_cov = xr.DataArray(x_cov, dims=("predictor", "predictor2"))
            part = ((slope * x_mean).sum("predictor") + intercept - internal_ref) * to_units
            methods[name] = (part, np.sqrt(quadratic(slope, x_cov) + sigma**2) * to_units, part)
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

        names = [n for n in METHODS if n in methods]
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
    members = members.assign_coords(member=first.member.values.astype(str), analogue=np.asarray(selected).astype(str), predictor=predictors)
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
        x = x[:, None] if x.ndim == 1 else x                                  # (member, predictor)
        w = members["weight_all"].values if name == "M2b" else np.ones(len(x))
        ok = np.isfinite(x).all(axis=1)
        x, w = x[ok], w[ok] / w[ok].sum()
        slope = np.atleast_1d(members["slope_u"].sel(site, method="nearest").values)
        sigma = float(members["sigma_u"].sel(site, method="nearest"))
        return mean + (x - w @ x) @ slope, np.full(len(x), sigma), w
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
        names = [m for m in METHODS if any(m in p.fc_method.values for p in parts)]
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
