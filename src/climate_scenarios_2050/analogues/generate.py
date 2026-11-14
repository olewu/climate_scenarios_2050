"""
Step 3: analogue hindcasts of the predictor index (`<root>/hindcast_index/<hash>/`, per scenario).

`hindcast.nc`: `index` (init_year, rank, lead): internal variability of the selected members,
relative to the init year (lead = year - init, negative leads are the members' past, needed for the
reference period), `member`, `score`, `lag` (init_year, rank), `n_valid` (init_year).
`synthetic.nc`: the same selection against synthetic AR(1) observations (null test).
"""

import logging

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.analogues.preselection import selected_models
from climate_scenarios_2050.analogues.synthetic import fit_ar1_ensemble, select_synthetic
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess.indices.compute import open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)


def hindcast_output(cfg, scenario: str) -> StepOutput:
    return StepOutput(
        cfg, "hindcast_index", sections=["analogues"],
        extra={"scenario": scenario, "matching_index": cfg["obs"]["matching_index"], "reference_length": cfg["target"]["reference_length"], "predictor_lag": cfg["target"]["predictor_lag"]},
        upstream={"split": split_output(cfg, scenario).hash},
    )


def lead_axis(cfg) -> np.ndarray:
    return np.arange(-cfg["target"]["reference_length"] - cfg["target"]["predictor_lag"], cfg["analogues"]["max_lead"] + 1)


def init_years(cfg, obs_years: np.ndarray) -> np.ndarray:
    """inits with a complete matching window"""
    acfg = cfg["analogues"]
    first = int(obs_years.min()) + (acfg["filter"].get("length", 41) if acfg["window"] == "all" else acfg["window"] - 1)
    if acfg.get("first_init") is not None:
        first = max(first, acfg["first_init"])
    return np.arange(first, int(obs_years.max()) + 1)


def matching_obs(cfg, split: SplitResults):
    """observed series and, per init, the coefficients removing its forced part (known at init)"""
    name = cfg["obs"]["matching_index"]
    series = open_obs_index(cfg, name).dropna("year")
    design = fc.design_matrix(split.basis, series.year.values)
    if cfg["split"]["obs_leakage"] == "full_period":
        coef = split.obs_coef(name, "coef_full")
        forced = fc.forced_part(design, coef)
        return series, lambda init: forced
    coef = split.obs_coef(name, "coef_up_to_init")
    return series, lambda init: fc.forced_part(design, coef.sel(init_year=init))


def member_pool(cfg, internal: xr.DataArray, obs_full: xr.DataArray) -> xr.DataArray:
    name = cfg["analogues"]["preselection"]
    if name is None:
        return internal
    models = selected_models(name, internal["model"].values, internal, obs_full, cfg["paths"]["inputs"]["subsampling"])
    log.info(f"pre-selection {name}: {len(models)} models")
    return internal.isel(member=np.isin(internal["model"].values, models))


def _one_init(init, obs_values, obs_years, model, years, members, leads, acfg):
    sel = matching.select_members(obs_values, obs_years, model, years, init, acfg)
    return {
        "init": init,
        "member": members[sel["member"]],
        "score": sel["score"],
        "lag": sel["lag"],
        "n_valid": sel["n_valid"],
        "index": matching.align_leads(model[sel["member"]], years, init, leads, sel["lag"]),
    }


def run_scenario(cfg, scenario: str):
    out = hindcast_output(cfg, scenario).prepare()
    log.info(f"analogue hindcast {scenario} -> {out.dir}")
    split = SplitResults(cfg, scenario)
    series, forced_at = matching_obs(cfg, split)
    internal = split.cmip6_internal()
    pool = member_pool(cfg, internal, series - forced_at(int(series.year.max())))
    pool = pool.transpose("member", "year")
    model, years, members = pool.values, pool.year.values, pool.member.values.astype(str)
    leads = lead_axis(cfg)
    inits = init_years(cfg, series.year.values)
    acfg = cfg["analogues"]

    if not out.exists("hindcast.nc"):
        n_jobs = cfg["resources"]["n_jobs"] if acfg["selection"] != "subsample" else 1
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_init)(int(init), (series - forced_at(int(init))).values, series.year.values, model, years, members, leads, acfg) for init in inits
        )
        n_rank = max(len(r["member"]) for r in results)
        shape = (len(inits), n_rank)
        index = np.full((*shape, len(leads)), np.nan, dtype="float32")
        score, lag = np.full(shape, np.nan), np.zeros(shape, dtype="int32")
        names = np.full(shape, "", dtype=f"U{max(len(m) for m in members)}")
        for ii, res in enumerate(results):
            n = len(res["member"])
            index[ii, :n] = res["index"]
            score[ii, :n], lag[ii, :n], names[ii, :n] = res["score"], res["lag"], res["member"]
        ds = xr.Dataset(
            {
                "index": (("init_year", "rank", "lead"), index),
                "score": (("init_year", "rank"), score),
                "lag": (("init_year", "rank"), lag),
                "member": (("init_year", "rank"), names),
                "n_valid": ("init_year", [r["n_valid"] for r in results]),
            },
            coords={"init_year": inits, "rank": np.arange(n_rank), "lead": leads},
        )
        write_netcdf(ds, out.path("hindcast.nc"), cfg)
        log.info(f"{scenario}: {len(inits)} inits ({inits[0]}-{inits[-1]}), {int(ds['n_valid'].min())}-{int(ds['n_valid'].max())} members in the pool")

    null_cfg = cfg["synthetic_null"]
    if null_cfg["enabled"] and not out.exists("synthetic.nc"):
        if acfg["selection"] != "subsample" or acfg["measure"] == "two_stage":
            log.warning("the synthetic null test is only implemented for selection=subsample with CORR or RMSE, skipping")
        else:
            full = series - fc.forced_part(fc.design_matrix(split.basis, series.year.values), split.obs_coef(cfg["obs"]["matching_index"], "coef_full"))
            synth = fit_ar1_ensemble(full, null_cfg)
            write_netcdf(select_synthetic(synth, model, years, inits, leads, acfg), out.path("synthetic.nc"), cfg)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        run_scenario(cfg, scenario)


def open_hindcast(cfg, scenario: str, name: str = "hindcast.nc") -> xr.Dataset:
    with xr.open_dataset(hindcast_output(cfg, scenario).path(name)) as ds:
        return ds.load()
