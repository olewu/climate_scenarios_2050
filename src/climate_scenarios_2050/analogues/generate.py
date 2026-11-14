"""
Step 3: analogue hindcasts of the predictor index (`<root>/hindcast_index/<hash>/`, per scenario).

`hindcast.nc`: `index` (init_year, rank, lead): internal variability of the selected members,
relative to the init year (lead = year - init, negative leads are the members' past, needed for the
reference period), `member`, `score`, `lag` (init_year, rank), `n_valid` (init_year).
`synthetic.nc`: the same selection against synthetic AR(1) observations (null test).

Several indices (`predictor.indices`): members are matched jointly on the indices of
`analogues.match_weights` (`matching.select_members`); the selected members' trajectories of every index
the later steps need (`hindcast_indices`: primary, matched, linear-model predictors) are stored as
`index_<name>`, `index` is the primary one. The pool are the members that have all these indices. The
synthetic null test is only run when the matching uses a single index.
"""

import logging

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.analogues.preselection import selected_models
from climate_scenarios_2050.analogues.synthetic import fit_ar1_ensemble, select_synthetic
from climate_scenarios_2050.core.config import hindcast_indices, index_cfg, match_weights, primary_index
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess.indices.compute import open_obs_index
from climate_scenarios_2050.split import forced as fc
from climate_scenarios_2050.split.run import SplitResults, split_output

log = logging.getLogger(__name__)


def hindcast_output(cfg, scenario: str) -> StepOutput:
    extra = {"scenario": scenario, "matching_index": cfg["obs"]["matching_index"], "reference_length": cfg["target"]["reference_length"], "predictor_lag": cfg["target"]["predictor_lag"]}
    names = hindcast_indices(cfg)
    if names == [primary_index(cfg)]:
        upstream = {"split": split_output(cfg, scenario).hash}   # one index: as a single-index configuration
    else:
        upstream = {"split": {name: split_output(cfg, scenario, name).hash for name in names}}
        extra["match_weights"] = match_weights(cfg)
    return StepOutput(cfg, "hindcast_index", sections=["analogues"], extra=extra, upstream=upstream)


def hindcast_index(ds: xr.Dataset, name: str) -> xr.DataArray:
    """trajectories (init_year, rank, lead) of one index in a hindcast dataset"""
    return ds[f"index_{name}"] if f"index_{name}" in ds else ds["index"]


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


def _one_init(init, obs_values, obs_years, model, years, members, leads, acfg, weights, stored):
    """`obs_values` (matched index, year), `model` (index, member, year), `stored`: model rows to keep"""
    sel = matching.select_members(obs_values, obs_years, model[: len(weights)], years, init, acfg, weights)
    return {
        "init": init,
        "member": members[sel["member"]],
        "score": sel["score"],
        "lag": sel["lag"],
        "n_valid": sel["n_valid"],
        "index": [matching.align_leads(model[k][sel["member"]], years, init, leads, sel["lag"]) for k in stored],
    }


def index_pool(cfg, scenario: str, names: list):
    """
    internal CMIP6 series (index, member, year) of the members that have all the indices, the observed
    series and the forced part known at init per index (`matching_obs`), and the primary split
    """
    splits = {name: SplitResults(cfg, scenario, name) for name in names}
    internals = {name: split.cmip6_internal().transpose("member", "year") for name, split in splits.items()}
    common = sorted(set.intersection(*[set(i.member.values.astype(str)) for i in internals.values()]))
    for name, internal in internals.items():
        if internal.sizes["member"] != len(common):
            log.info(f"{name}: {internal.sizes['member'] - len(common)} members without all indices left out")
    first = internals[names[0]].sel(member=common)
    years = first.year.values
    stack = np.stack([internals[n].sel(member=common).reindex(year=years).values for n in names])
    obs = {name: matching_obs(index_cfg(cfg, name), splits[name]) for name in names}
    return stack, first, obs, splits


def run_scenario(cfg, scenario: str):
    out = hindcast_output(cfg, scenario).prepare()
    log.info(f"analogue hindcast {scenario} -> {out.dir}")
    weights = match_weights(cfg)
    matched = list(weights)
    names = list(dict.fromkeys([*matched, *hindcast_indices(cfg)]))   # matched first
    primary = primary_index(cfg)
    stack, first, obs, splits = index_pool(cfg, scenario, names)
    series, forced_at = obs[primary]
    # the pre-selection works on the primary index
    keep = member_pool(cfg, first, series - forced_at(int(series.year.max()))).member.values
    rows = np.isin(first.member.values, keep)
    model, members = stack[:, rows], first.member.values[rows].astype(str)
    years = first.year.values
    leads = lead_axis(cfg)
    obs_years = series.year.values
    for name in matched:
        obs_years = np.intersect1d(obs_years, obs[name][0].year.values)
    inits = init_years(cfg, obs_years)
    acfg = cfg["analogues"]
    w = np.array([weights[n] for n in matched])
    stored = list(range(len(names)))

    def obs_at(init):
        return np.stack([(obs[n][0] - obs[n][1](init)).sel(year=obs_years).values for n in matched])

    if not out.exists("hindcast.nc"):
        n_jobs = cfg["resources"]["n_jobs"] if acfg["selection"] != "subsample" else 1
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_init)(int(init), obs_at(int(init)), obs_years, model, years, members, leads, acfg, w, stored) for init in inits
        )
        n_rank = max(len(r["member"]) for r in results)
        shape = (len(inits), n_rank)
        index = np.full((len(names), *shape, len(leads)), np.nan, dtype="float32")
        score, lag = np.full(shape, np.nan), np.zeros(shape, dtype="int32")
        member_names = np.full(shape, "", dtype=f"U{max(len(m) for m in members)}")
        for ii, res in enumerate(results):
            n = len(res["member"])
            for k in stored:
                index[k, ii, :n] = res["index"][k]
            score[ii, :n], lag[ii, :n], member_names[ii, :n] = res["score"], res["lag"], res["member"]
        dims = ("init_year", "rank", "lead")
        ds = xr.Dataset(
            {
                "index": (dims, index[names.index(primary)]),
                **{f"index_{name}": (dims, index[k]) for k, name in enumerate(names)},
                "score": (("init_year", "rank"), score),
                "lag": (("init_year", "rank"), lag),
                "member": (("init_year", "rank"), member_names),
                "n_valid": ("init_year", [r["n_valid"] for r in results]),
            },
            coords={"init_year": inits, "rank": np.arange(n_rank), "lead": leads},
        )
        ds.attrs.update({"indices": " ".join(names), "matched": " ".join(f"{n}:{weights[n]:.3g}" for n in matched)})
        write_netcdf(ds, out.path("hindcast.nc"), cfg)
        log.info(f"{scenario}: {len(inits)} inits ({inits[0]}-{inits[-1]}), {int(ds['n_valid'].min())}-{int(ds['n_valid'].max())} members in the pool, matched on {matched}")

    null_cfg = cfg["synthetic_null"]
    if null_cfg["enabled"] and not out.exists("synthetic.nc"):
        if acfg["selection"] != "subsample" or acfg["measure"] == "two_stage":
            log.warning("the synthetic null test is only implemented for selection=subsample with CORR or RMSE, skipping")
        elif len(matched) > 1:
            log.warning("the synthetic null test is only implemented for matching on one index, skipping")
        else:
            name = matched[0]
            split = splits[name]
            full = obs[name][0] - fc.forced_part(fc.design_matrix(split.basis, obs[name][0].year.values), split.obs_coef(cfg["obs"]["matching_index"], "coef_full"))
            synth = fit_ar1_ensemble(full, null_cfg)
            write_netcdf(select_synthetic(synth, model[names.index(name)], years, inits, leads, acfg), out.path("synthetic.nc"), cfg)
    return out


def run(cfg, scenarios=None):
    for scenario in scenarios or cfg["cmip6"]["scenarios_hindcast"]:
        run_scenario(cfg, scenario)


def open_hindcast(cfg, scenario: str, name: str = "hindcast.nc") -> xr.Dataset:
    with xr.open_dataset(hindcast_output(cfg, scenario).path(name)) as ds:
        return ds.load()
