"""
Variance partition of projected window changes (`pixi run partition [--config ...]`): how much of the spread
of the change of an N-year window mean, relative to the reference period before init, comes from internal
variability, model uncertainty and scenario uncertainty (Hawkins & Sutton 2009), per lead time. One output
folder (`<root>/partition/<hash>/partition.nc`) per index and per target and region.

For a series x of every CMIP6 member in every scenario s (forced part F of its model m plus its internal
part I, from the forced/internal split) and a window of N years starting at lead l after the init year Y:

  Δ[x] = mean(x, years Y+l ... Y+l+N-1) - mean(x, the `target.reference_length` years up to Y)

  internal  I = mean_s mean_m mean_{r in m} Δ[I_r]^2      (model-weighted, cmip6.weighting: member -> over members)
  model     M = mean_s var_m Δ[F_m,s]                     (ddof 1, models with a forced part in s)
  scenario  S = var_s mean_m Δ[F_m,s]                     (ddof 1, the multi-model mean change per scenario)

and the fractions I/T, M/T, S/T with T = I + M + S. Relative targets (`targets.<t>.change: relative`): Δ in % of
the reference-period mean of the model's forced part in that scenario.

- indices: the split of each scenario as used everywhere else (`split/<hash>/cmip6_index.nc`); the forced part is
  the configured split method (`split.method`, e.g. the scaled MMEM per model)
- targets: the area mean (cos lat) of the target's CMIP6 field (`targets.<t>.cmip6`, annual means) over each box
  of `partition.regions`, split per scenario like an index (regression of the single-model mean on the
  multi-model mean of the box mean, `split` settings); targets without a CMIP6 field (energy) and boxes outside
  `region.domain` (the extent of the CMIP6 target fields) are skipped

Scenarios: those of `cmip6.scenarios` whose split (indices) or preprocessed field (targets) exists.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import active_targets, index_cfg, index_names, target_spec
from climate_scenarios_2050.core.io import StepOutput, write_netcdf

log = logging.getLogger(__name__)

SOURCES = ["internal", "model", "scenario"]
AGGREGATION = "annual"
SPLIT_KEYS = ["method", "polynomial_degree", "lowpass", "min_model_fraction"]
CMIP6_UNITS = {"tas": "K", "tos": "K", "sfcWind": "m/s", "psl": "Pa", "ua": "m/s", "pr": "kg m-2 s-1"}   # when the stores have none


#---------------computation--------------#
def window_change(values: np.ndarray, years: np.ndarray, init: int, length: int, leads: np.ndarray, reference_length: int) -> np.ndarray:
    """
    (..., lead) change of the `length`-year window mean starting at Y + lead against the `reference_length`
    years up to Y, for values (..., year); NaN where a window or the reference is not complete
    """
    years = np.asarray(years)
    in_ref = (years > init - reference_length) & (years <= init)
    ref = values[..., in_ref].mean(-1) if in_ref.sum() == reference_length else np.full(values.shape[:-1], np.nan)
    out = np.full(values.shape[:-1] + (len(leads),), np.nan)
    for k, lead in enumerate(leads):
        sel = (years >= init + lead) & (years < init + lead + length)
        if sel.sum() == length:
            out[..., k] = values[..., sel].mean(-1) - ref
    return out


def reference_mean(values: np.ndarray, years: np.ndarray, init: int, reference_length: int) -> np.ndarray:
    in_ref = (np.asarray(years) > init - reference_length) & (np.asarray(years) <= init)
    return values[..., in_ref].mean(-1)


def partition(cases: dict, weighting: str = "model") -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # leads without any complete window
        return _partition(cases, weighting)


def _partition(cases: dict, weighting: str) -> dict:
    """
    cases: scenario -> {"forced": (model, lead) change of each model's forced part, "internal": (member, lead)
    change of each member's internal part, "member_model": (member,) row of the member's model in "forced"}.
    Returns the variances (lead,) of the three sources, the multi-model mean change per scenario and counts.
    """
    internal, model, mmem = [], [], {}
    for scenario, case in cases.items():
        forced, ichange, rows = case["forced"], case["internal"], np.asarray(case["member_model"])
        sq = ichange**2
        if weighting == "member":
            internal.append(np.nanmean(sq, axis=0))
        else:
            per_model = [np.nanmean(sq[rows == r], axis=0) for r in np.unique(rows)]
            internal.append(np.nanmean(per_model, axis=0))
        valid = np.isfinite(forced)
        with np.errstate(invalid="ignore", divide="ignore"):
            n = valid.sum(0)
            mean = np.where(n > 0, np.nansum(forced, axis=0) / np.maximum(n, 1), np.nan)
            var = np.where(n > 1, np.nansum((forced - mean) ** 2, axis=0) / np.maximum(n - 1, 1), np.nan)
        model.append(var)
        mmem[scenario] = mean
    stack = np.array(list(mmem.values()))
    scen = np.var(stack, axis=0, ddof=1) if len(stack) > 1 else np.full(stack.shape[1:], np.nan)
    return {"internal": np.nanmean(internal, axis=0), "model": np.nanmean(model, axis=0), "scenario": scen, "scenario_mean": mmem}


def scenario_case(internal: xr.DataArray, forced: xr.DataArray, init, length, leads, reference_length, relative: bool) -> dict:
    """the window changes of one scenario: internal (member, year) with a model_id coordinate, forced (model_id, year)"""
    years = internal.year.values
    forced = forced.sel(year=years)
    models = list(forced.model_id.values)
    member_model = np.array([models.index(m) for m in internal["model_id"].values])
    fvals = forced.transpose("model_id", "year").values
    dF = window_change(fvals, years, init, length, leads, reference_length)
    dI = window_change(internal.transpose("member", "year").values, years, init, length, leads, reference_length)
    if relative:
        scale = reference_mean(fvals, years, init, reference_length)   # (model,)
        with np.errstate(invalid="ignore", divide="ignore"):
            dF = 100 * dF / scale[:, None]
            dI = 100 * dI / scale[member_model][:, None]
    return {"forced": dF, "internal": dI, "member_model": member_model}


def partition_dataset(splits: dict, pcfg: dict, reference_length: int, weighting: str, relative: bool, attrs: dict) -> xr.Dataset:
    """splits: scenario -> (internal (member, year), forced (model_id, year)); all window lengths"""
    attrs = dict(attrs)
    units = "%" if relative else attrs.pop("units", "")
    attrs.pop("units", None)
    init = int(pcfg["init"])
    last = min(int(internal.year.max()) for internal, _ in splits.values())
    max_lead = last - init if pcfg.get("max_lead") is None else int(pcfg["max_lead"])
    leads = np.arange(1, max_lead + 1)
    lengths = [int(n) for n in pcfg["window_lengths"]]
    scenarios = list(splits)
    var = np.full((len(lengths), len(leads), len(SOURCES)), np.nan)
    smean = np.full((len(lengths), len(leads), len(scenarios)), np.nan)
    for j, length in enumerate(lengths):
        cases = {s: scenario_case(i, f, init, length, leads, reference_length, relative) for s, (i, f) in splits.items()}
        res = partition(cases, weighting)
        var[j] = np.stack([res[k] for k in SOURCES], axis=-1)
        smean[j] = np.stack([res["scenario_mean"][s] for s in scenarios], axis=-1)
    total = var.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = var / total[..., None]
    centre = init + leads[None, :] + (np.array(lengths)[:, None] - 1) / 2
    ds = xr.Dataset(
        {
            "variance": (("window_length", "lead", "source"), var),
            "fraction": (("window_length", "lead", "source"), frac),
            "total": (("window_length", "lead"), total),
            "scenario_mean": (("window_length", "lead", "scenario"), smean),
            "n_models": ("scenario", [splits[s][1].sizes["model_id"] for s in scenarios]),
            "n_members": ("scenario", [splits[s][0].sizes["member"] for s in scenarios]),
        },
        coords={"window_length": lengths, "lead": leads, "source": SOURCES, "scenario": scenarios,
                "centre_year": (("window_length", "lead"), centre)},
    )
    ds.attrs.update({"init": init, "reference_period": f"{init - reference_length + 1}-{init}", "weighting": weighting,
                     "units": units, **attrs,
                     "definition": "variance of the change of the window mean (lead = first year after init) against the reference period"})
    return ds


#---------------inputs and outputs--------------#
def partition_indices(cfg) -> list:
    return list(cfg["partition"].get("indices") or index_names(cfg))


def partition_targets(cfg) -> list:
    """targets with a CMIP6 field (configured, or the active ones)"""
    names = cfg["partition"].get("targets") or active_targets(cfg)
    return [t for t in names if target_spec(cfg, t).get("cmip6")]


def _common(cfg) -> dict:
    return {"init": cfg["partition"]["init"], "reference_length": cfg["target"]["reference_length"], "weighting": cfg["cmip6"]["weighting"]}


def index_scenarios(cfg, name) -> list:
    from climate_scenarios_2050.split.run import split_output

    return [s for s in cfg["cmip6"]["scenarios"] if split_output(index_cfg(cfg, name), s).exists("cmip6_index.nc")]


def index_partition_output(cfg, name, scenarios=None) -> StepOutput:
    from climate_scenarios_2050.split.run import split_output

    scenarios = index_scenarios(cfg, name) if scenarios is None else scenarios
    return StepOutput(cfg, "partition", sections=["partition"], exclude={"partition": ["regions", "indices", "targets"]},
                      extra={"kind": "index", "index": name, **_common(cfg)},
                      upstream={f"split_{s}": split_output(index_cfg(cfg, name), s).hash for s in scenarios})


def target_scenarios(cfg, target) -> list:
    from climate_scenarios_2050.preprocess import cmip6

    store, _ = cmip6.field_store(cfg, target_spec(cfg, target)["cmip6"], AGGREGATION)
    return [s for s in cfg["cmip6"]["scenarios"] if (Path(store) / s).is_dir() and any((Path(store) / s).iterdir())]


def target_partition_output(cfg, target, region, scenarios=None) -> StepOutput:
    from climate_scenarios_2050.preprocess import cmip6

    spec = target_spec(cfg, target)
    scenarios = target_scenarios(cfg, target) if scenarios is None else scenarios
    return StepOutput(cfg, "partition", sections=["partition"], exclude={"partition": ["regions", "indices", "targets"]},
                      extra={"kind": "target", "target": target, "cmip6": spec["cmip6"], "change": spec["change"],
                             "region": {region: cfg["partition"]["regions"][region]},
                             "split": {k: cfg["split"][k] for k in SPLIT_KEYS if k in cfg["split"]}, **_common(cfg)},
                      upstream={f"cmip6_{s}": cmip6.field_id(cfg, spec["cmip6"], AGGREGATION, s) for s in scenarios})


def partition_regions(cfg) -> list:
    """boxes of `partition.regions` inside `region.domain` (the CMIP6 target fields only cover the domain)"""
    domain = cfg["region"]["domain"]
    out = []
    for name, box in cfg["partition"]["regions"].items():
        if dom.crosses_dateline(box) or dom.crosses_dateline(domain):
            inside = True   # not checked
        else:
            inside = domain["lon"][0] <= box["lon"][0] and box["lon"][1] <= domain["lon"][1] and domain["lat"][0] <= box["lat"][0] and box["lat"][1] <= domain["lat"][1]
        if inside:
            out.append(name)
        else:
            log.warning(f"partition.regions.{name} {box} is not inside region.domain {domain}: skipped")
    return out


def outputs(cfg) -> list:
    """(label, StepOutput) of every partition case of the configuration"""
    out = [(f"index {n}", index_partition_output(cfg, n)) for n in partition_indices(cfg)]
    for t in partition_targets(cfg):
        for region in partition_regions(cfg):
            out.append((f"target {t} {region}", target_partition_output(cfg, t, region)))
    return out


def box_mean(field: xr.DataArray, box: dict) -> xr.DataArray:
    """cos-lat weighted mean (member, year) of a field (member, year, lat, lon) over a box"""
    sub = dom.select(field, box)
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(f"box {box} is outside the field ({float(field.lat.min())}-{float(field.lat.max())}N, {float(field.lon.min())}-{float(field.lon.max())}E)")
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w.fillna(0)).mean(("lat", "lon"), skipna=True)


def target_splits(cfg, target, region, scenarios) -> dict:
    """per scenario: internal and forced part of the box mean, split like an index"""
    from climate_scenarios_2050.core.config import field_variable
    from climate_scenarios_2050.preprocess import cmip6
    from climate_scenarios_2050.split import forced as fc

    field = target_spec(cfg, target)["cmip6"]
    box = cfg["partition"]["regions"][region]
    out = {}
    for s in scenarios:
        ds = cmip6.open_field(cfg, field, AGGREGATION, s)
        series = box_mean(ds[field_variable(cfg, field)].transpose("member", "year", "lat", "lon"), box).load()
        mmem = fc.multi_model_mean(series, cfg["cmip6"]["weighting"], cfg["split"]["min_model_fraction"])
        split = fc.split_ensemble(series, fc.make_basis(mmem, cfg["split"]))
        internal = split["internal"].assign_coords(model_id=series["model_id"])
        out[s] = (internal, split["forced"])
        log.info(f"{target} {region} {s}: {series.sizes['member']} members of {split['forced'].sizes['model_id']} models")
    return out


def field_units(cfg, field, scenario) -> str:
    """units of a CMIP6 field (lazy open, no data read)"""
    from climate_scenarios_2050.core.config import field_variable
    from climate_scenarios_2050.preprocess import cmip6

    variable = field_variable(cfg, field)
    try:
        units = str(cmip6.open_field(cfg, field, AGGREGATION, scenario)[variable].attrs.get("units", ""))
    except FileNotFoundError:
        units = ""
    return units or CMIP6_UNITS.get(variable, "")


def index_splits(cfg, name, scenarios) -> dict:
    from climate_scenarios_2050.split.run import SplitResults

    out = {}
    for s in scenarios:
        res = SplitResults(cfg, s, index=name)
        out[s] = (res.cmip6_internal(), res.cmip6_forced())
    return out


def run_case(cfg, out: StepOutput, load, attrs, relative):
    if out.exists("partition.nc"):
        log.info(f"{attrs['name']}: done ({out.dir})")
        return out
    out.prepare()
    splits = load()
    if len(splits) < 2:
        log.warning(f"{attrs['name']}: fewer than two scenarios ({list(splits)}), no scenario uncertainty")
    ds = partition_dataset(splits, cfg["partition"], cfg["target"]["reference_length"], cfg["cmip6"]["weighting"], relative, attrs)
    write_netcdf(ds, out.path("partition.nc"), cfg)
    log.info(f"{attrs['name']}: {out.dir}")
    return out


def run(cfg, plot=True):
    from climate_scenarios_2050.plotting.partition import plot_partition

    done = []
    for name in partition_indices(cfg):
        scenarios = index_scenarios(cfg, name)
        if not scenarios:
            log.warning(f"index {name}: no split of any scenario yet (pixi run split)")
            continue
        out = index_partition_output(cfg, name, scenarios)
        done.append(run_case(cfg, out, lambda: index_splits(cfg, name, scenarios), {"kind": "index", "name": name, "units": "index units (std)"}, relative=False))
    skipped = [t for t in (cfg["partition"].get("targets") or active_targets(cfg)) if not target_spec(cfg, t).get("cmip6")]
    if skipped:
        log.info(f"no CMIP6 field, no partition: {skipped}")
    for t in partition_targets(cfg):
        scenarios = target_scenarios(cfg, t)
        if not scenarios:
            log.warning(f"target {t}: no preprocessed {target_spec(cfg, t)['cmip6']} field yet")
            continue
        spec = target_spec(cfg, t)
        for region in partition_regions(cfg):
            box = cfg["partition"]["regions"][region]
            out = target_partition_output(cfg, t, region, scenarios)
            attrs = {"kind": "target", "name": f"{t} {region}", "target": t, "cmip6": spec["cmip6"], "region": region,
                     "box": f"lon {box['lon'][0]}..{box['lon'][1]}, lat {box['lat'][0]}..{box['lat'][1]}", "units": field_units(cfg, spec["cmip6"], scenarios[0])}
            done.append(run_case(cfg, out, lambda t=t, region=region, scenarios=scenarios: target_splits(cfg, t, region, scenarios), attrs, relative=spec["change"] == "relative"))
    if plot:
        for out in done:
            plot_partition(cfg, out)
    return done
