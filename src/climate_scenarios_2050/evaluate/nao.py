"""
Step 3b: evaluation of the index hindcasts (`<root>/evaluate_nao/<hash>/`, figures below `figures`).

Two views, for the analogue hindcast, the full CMIP6 ensemble ("climatological" reference), the
AR(1) and sine + trend baselines:

- `continuous`: the old evaluation (`evaluate_full_synth.py`): centred running means of the index,
  correlation and RMSE skill over all inits per lead time, with block bootstrap intervals and
  phase-randomization p-values. Observations are smoothed with the same *centred* running mean (the old
  trailing mean was 4 years off). Verified against the index split with the full-period coefficients.
- `windows`: change of the target-window mean relative to the 20 seasons before init. Verified against
  the index split with the target window and reference left out, forecasts use their own reference
  (observed for the baselines). Deterministic and probabilistic scores.

Null test: the 95th percentile over the synthetic series of the same scores.
"""

import logging

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from climate_scenarios_2050.analogues.generate import hindcast_output, init_years, lead_axis, matching_obs, open_hindcast
from climate_scenarios_2050.analogues.matching import align_leads, running_mean
from climate_scenarios_2050.baselines import nao as baselines
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate import metrics
from climate_scenarios_2050.evaluate.significance import score_with_uncertainty
from climate_scenarios_2050.preprocess.indices.compute import open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults

log = logging.getLogger(__name__)


def evaluation_output(cfg, scenario) -> StepOutput:
    # the significance test and the sites of the wind evaluation are not used here
    return StepOutput(cfg, "evaluate_nao", sections=["evaluation", "baselines"], upstream={"hindcast_nao": hindcast_output(cfg, scenario).hash}, exclude={"evaluation": ["significance", "sites"]})


def lead_window_change(hc: xr.DataArray, windows, reference_length: int) -> xr.DataArray:
    """change of the window means relative to the reference (leads -reference_length+1 ... 0)"""
    ref = hc.sel(lead=slice(-reference_length + 1, 0))
    ref_mean = ref.mean("lead").where(ref.notnull().all("lead"))
    out = []
    for a, b in windows:
        win = hc.sel(lead=slice(a, b))
        out.append((win.mean("lead").where(win.notnull().all("lead")) - ref_mean).expand_dims(window=[window_label((a, b))]))
    return xr.concat(out, dim="window")


def obs_window_change(series, design, coef, windows, reference_length) -> xr.DataArray:
    """observed change per (init_year, window), each with its own leave-out split"""
    out = xr.full_like(coef.isel(term=0, drop=True), np.nan)
    years = series.year.values
    for iw, (a, b) in enumerate(windows):
        for ii, init in enumerate(coef.init_year.values):
            this = coef.isel(init_year=ii, window=iw)
            if not bool(this.notnull().all()):
                continue
            internal = series - fc.forced_part(design, this)
            win = internal.sel(year=slice(init + a, init + b))
            ref = internal.sel(year=slice(init - reference_length + 1, init))
            if win.sizes["year"] == b - a + 1 and ref.sizes["year"] == reference_length:
                out[ii, iw] = float(win.mean() - ref.mean())
    return out.rename("obs_change")


def build_products(cfg, scenario, split, leads) -> dict:
    """all hindcasts (init_year, member, lead) that are verified"""
    hindcast = open_hindcast(cfg, scenario)
    inits = hindcast.init_year.values
    products = {"analogue": hindcast["index"].rename(rank="member")}

    internal = split.cmip6_internal().transpose("member", "year")
    n_members = internal.sizes["member"]
    full = np.stack([align_leads(internal.values, internal.year.values, int(init), leads) for init in inits])
    products["cmip6_all"] = xr.DataArray(full, dims=("init_year", "member", "lead"), coords={"init_year": inits, "member": np.arange(n_members), "lead": leads})

    series, forced_at = matching_obs(cfg, split)
    obs_at_init = lambda init: series - forced_at(init)
    bcfg = cfg["baselines"]
    for name, ar_cfg in bcfg["ar1"].items():
        ar_cfg = {"members": bcfg["members"], **ar_cfg}
        products[name] = baselines.ar1_hindcast(obs_at_init, inits, leads, ar_cfg, bcfg["seed"])
    products["sine_trend"] = baselines.fft_hindcast(obs_at_init, inits, leads, bcfg["sine_trend"])
    products["persistence"] = baselines.persistence_hindcast(obs_at_init, inits, leads, cfg["target"]["reference_length"])
    return products


def _continuous_one(product, name, lead, verification, Nyr, n_boot, seed, smoothed):
    fcst = product.sel(lead=lead).mean("member")
    obs = verification.sel(init_year=fcst.init_year, lead=lead)
    block = max(int(round(fcst.sizes["init_year"] ** (1 / 3))), Nyr)
    out = {"product": name, "lead": lead}
    for measure in ["corr", "nrmse"]:
        res = score_with_uncertainty(obs.values, fcst.values, block, n_boot, measure, seed)
        out.update({f"{measure}_{k}": v for k, v in res.items()})
    return out


def continuous_evaluation(cfg, products, verification, leads_eval) -> xr.Dataset:
    ncfg = cfg["evaluation"]["nao_continuous"]
    Nyr = ncfg["running_mean"]
    boot = cfg["evaluation"]["bootstrap"]
    smoothed = {}
    for name, product in products.items():
        if cfg["baselines"]["ar1"].get(name, {}).get("fit_on") == "running_mean":
            smoothed[name] = product  # already a forecast of the smoothed index
        else:
            smoothed[name] = product.rolling(lead=Nyr, center=True).mean()
    rows = Parallel(n_jobs=cfg["resources"]["n_jobs"])(
        delayed(_continuous_one)(smoothed[name], name, int(lead), verification, Nyr, boot["n"], boot["seed"], None) for name in products for lead in leads_eval
    )
    keys = [k for k in rows[0] if k not in ["product", "lead"]]
    names = list(products)
    data = {k: (("product", "lead"), np.array([[next(r[k] for r in rows if r["product"] == n and r["lead"] == int(l)) for l in leads_eval] for n in names], dtype=float)) for k in keys}
    return xr.Dataset(data, coords={"product": names, "lead": leads_eval})


def obs_running_verification(internal_full, inits, leads, Nyr) -> xr.DataArray:
    """centred running mean of the observations as (init_year, lead), like the smoothed hindcasts"""
    smooth = running_mean(internal_full.values, Nyr, Nyr)[0]
    return xr.DataArray(align_leads(smooth[None, :].repeat(len(inits), axis=0), internal_full.year.values, 0, leads, inits), dims=("init_year", "lead"), coords={"init_year": inits, "lead": leads})


def hindcast_overview(cfg, products, internal_full, name) -> xr.Dataset:
    """
    ensemble mean of the analogue hindcast (init_year, lead), smoothed like the continuous evaluation,
    and the observed internal index with the same centred running mean (visual verification)
    """
    Nyr = cfg["evaluation"]["nao_continuous"]["running_mean"]
    analogue = products["analogue"].rolling(lead=Nyr, center=True).mean().mean("member")
    observed = running_mean(internal_full.values, Nyr, Nyr)[0]
    ds = xr.Dataset({
        "analogue_mean": analogue,
        "observed": xr.DataArray(observed, dims="year", coords={"year": internal_full.year.values}),
        "observed_raw": xr.DataArray(internal_full.values, dims="year", coords={"year": internal_full.year.values}),
    })
    ds.attrs.update({"running_mean": Nyr, "matching_window": str(cfg["analogues"]["window"]), "observations": name, "index": cfg["predictor"]["name"]})
    return ds


def window_evaluation(cfg, products, obs_change, windows) -> xr.Dataset:
    ref_len = cfg["target"]["reference_length"]
    quantiles = cfg["evaluation"]["quantiles"]
    boot = cfg["evaluation"]["bootstrap"]
    results = []
    for name, product in products.items():
        if cfg["baselines"]["ar1"].get(name, {}).get("fit_on") == "running_mean":
            continue
        change = lead_window_change(product, windows, ref_len)
        row = []
        for (a, b), label in zip(windows, change.window.values):
            fcst = change.sel(window=label)
            obs = obs_change.sel(window=label).sel(init_year=fcst.init_year)
            scores = metrics.deterministic(fcst.mean("member"), obs).drop_vars("window", errors="ignore")
            if fcst.sizes["member"] > 1:
                # non-overlapping target windows only; zero change (persistence) as reference
                prob = metrics.probabilistic(fcst, obs, b - a + 1, quantiles, reference_mean=0.0)
                scores = scores.merge(prob.drop_vars("window", errors="ignore"))
            block = max(int(round(fcst.sizes["init_year"] ** (1 / 3))), b - a + 1)
            unc = score_with_uncertainty(obs.values, fcst.mean("member").values, block, boot["n"], "corr", boot["seed"])
            scores = scores.assign(corr_p=unc["p"], corr_ci_low=unc["ci_low"], corr_ci_high=unc["ci_high"])
            row.append(scores.expand_dims(window=[label]))
        results.append(xr.concat(row, dim="window").expand_dims(product=[name]))
    # deterministic products have no probabilistic scores: same variables, NaN
    template = next(res for res in results if "crps" in res)
    for ii, res in enumerate(results):
        if "crps" not in res:
            missing = {v: xr.full_like(template[v].isel(product=0, drop=True), np.nan).expand_dims(product=res["product"].values) for v in template.data_vars if v not in res}
            results[ii] = res.assign_coords(quantile=template["quantile"]).assign(missing)
    return xr.concat(results, dim="product")


def synthetic_null(cfg, synth: xr.Dataset, windows, leads_eval) -> xr.Dataset:
    """95th percentile over the synthetic series of the continuous and window skill"""
    Nyr = cfg["evaluation"]["nao_continuous"]["running_mean"]
    ref_len = cfg["target"]["reference_length"]
    mean = synth["mean"]
    inits = mean.init_year.values
    leads = mean.lead.values
    obs = synth["synthetic_obs"].transpose("syn", "year")
    syn_ids, years = obs.syn.values, obs.year.values

    smooth_obs = running_mean(obs.values, Nyr, Nyr)
    aligned = np.stack([align_leads(smooth_obs, years, int(init), leads) for init in inits], axis=1)
    verif = xr.DataArray(aligned, dims=("syn", "init_year", "lead"), coords={"syn": syn_ids, "init_year": inits, "lead": leads})
    smooth_fc = mean.rolling(lead=Nyr, center=True).mean()
    det = metrics.deterministic(smooth_fc, verif)
    continuous = det[["corr", "nrmse"]].sel(lead=leads_eval).quantile(0.95, dim="syn").drop_vars("quantile")

    at_leads = np.stack([align_leads(obs.values, years, int(init), leads) for init in inits], axis=1)
    obs_aligned = xr.DataArray(at_leads, dims=mean.dims, coords=mean.coords)
    fc_change = lead_window_change(mean, windows, ref_len)
    obs_change = lead_window_change(obs_aligned, windows, ref_len)
    win = metrics.deterministic(fc_change, obs_change)[["corr", "nrmse"]].quantile(0.95, dim="syn").drop_vars("quantile")
    return continuous.rename({"corr": "continuous_corr_p95", "nrmse": "continuous_nrmse_p95"}).merge(win.rename({"corr": "window_corr_p95", "nrmse": "window_nrmse_p95"}))


def run_scenario(cfg, scenario):
    out = evaluation_output(cfg, scenario).prepare()
    log.info(f"NAO evaluation {scenario} -> {out.dir}")
    split = SplitResults(cfg, scenario)
    name = cfg["obs"]["matching_index"]
    leads = lead_axis(cfg)
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    series = open_obs_index(cfg, name).dropna("year")
    design = fc.design_matrix(split.basis, series.year.values)
    products = build_products(cfg, scenario, split, leads)
    inits = products["analogue"].init_year.values

    ncfg = cfg["evaluation"]["nao_continuous"]
    leads_eval = np.arange(ncfg["lead_times"][0], ncfg["lead_times"][1] + 1)
    internal_full = series - fc.forced_part(design, split.obs_coef(name, "coef_full"))
    verification = obs_running_verification(internal_full, inits, leads, ncfg["running_mean"])
    if not out.exists("continuous.nc"):
        write_netcdf(continuous_evaluation(cfg, products, verification, leads_eval), out.path("continuous.nc"), cfg)
    if not out.exists("hindcast_overview.nc"):
        write_netcdf(hindcast_overview(cfg, products, internal_full, name), out.path("hindcast_overview.nc"), cfg)

    if not out.exists("windows.nc"):
        obs_change = obs_window_change(series, design, split.obs_coef(name, "coef_window_lag0"), windows, cfg["target"]["reference_length"])
        write_netcdf(window_evaluation(cfg, products, obs_change, windows), out.path("windows.nc"), cfg)

    hc_out = hindcast_output(cfg, scenario)
    if hc_out.exists("synthetic.nc") and not out.exists("null.nc"):
        write_netcdf(synthetic_null(cfg, open_hindcast(cfg, scenario, "synthetic.nc"), windows, leads_eval), out.path("null.nc"), cfg)

    from climate_scenarios_2050.plotting.nao import plot_evaluation
    plot_evaluation(cfg, out, scenario)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        run_scenario(cfg, scenario)
