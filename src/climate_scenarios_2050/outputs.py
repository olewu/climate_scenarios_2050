"""
Output folders of a configuration: `pixi run outputs [--config ...] [--step ...]` lists, per step and case
(scenario, index, target), the hash folder the steps use for this configuration and whether it exists.

Only hashes are computed (as in the steps themselves); nothing is written or computed. The CMIP6 member
lists are read from the archive's file names, as for the hashes of the steps.

Status: `done` = files besides config.yaml, `empty` = folder with config.yaml only, `-` = not there (not run
with this configuration yet). The forecast hash covers the scenarios whose split exists (as in the forecast step)
and the windows (`--window`, default: target.windows); one folder holds the forecasts of all init years.
"""

from pathlib import Path

from climate_scenarios_2050.core.config import active_targets, hindcast_indices, index_cfg, index_names, predictor_field, used_fields
from climate_scenarios_2050.core.io import read_only

# pipeline step -> output steps (folders) it writes
STEP_FOLDERS = {
    "preprocess": ["cmip6", "obs_fields", "obs_target", "indices", "indices_cmip6"],
    "split": ["split"],
    "hindcast-index": ["hindcast_index"],
    "evaluate-index": ["evaluate_index"],
    "linear-model": ["linear_model"],
    "hindcast-target": ["hindcast_target"],
    "evaluate-target": ["evaluate_target"],
    "forecast": ["forecast"],
    "partition": ["partition"],
}
FIGURES = {"evaluate_index", "linear_model", "evaluate_target", "forecast", "partition"}   # steps with a figure folder of the same hash


def _status(folder: Path) -> str:
    if not folder.exists():
        return "-"
    return "done" if any(p.name != "config.yaml" for p in folder.iterdir()) else "empty"


def _cases(cfg, folder, scenarios, targets, windows):
    """(case label, StepOutput or (path, hash)) of one output step"""
    from climate_scenarios_2050.analogues.generate import hindcast_output
    from climate_scenarios_2050.evaluate.index import evaluation_output
    from climate_scenarios_2050.evaluate.target import target_evaluation_output
    from climate_scenarios_2050.forecast.forecast import forecast_output
    from climate_scenarios_2050.linear_model.fit import linear_model_output
    from climate_scenarios_2050.methods.target import target_hindcast_output
    from climate_scenarios_2050.preprocess import cmip6, obs
    from climate_scenarios_2050.preprocess.indices.compute import cmip6_index_output, index_output
    from climate_scenarios_2050.split.run import split_output

    all_sc = [s for s in cfg["cmip6"]["scenarios"] if not scenarios or s in scenarios]
    hc_sc = [s for s in cfg["cmip6"]["scenarios_hindcast"] if not scenarios or s in scenarios]
    names = index_names(cfg)
    if folder == "cmip6":
        for field in used_fields(cfg):
            for agg in cfg["preprocess"]["fields"][field]["aggregations"]:
                store = cmip6.field_store(cfg, field, agg)[0]
                yield f"{field} {agg}", store
    elif folder == "obs_fields":
        for field in dict.fromkeys(predictor_field(index_cfg(cfg, n)) for n in names):
            yield field, obs.obs_field_output(cfg, field)
    elif folder == "obs_target":
        for tv in targets:
            yield tv, obs.target_output(cfg, tv)
    elif folder == "indices":
        for n in names:
            yield n, index_output(cfg, n)
    elif folder == "indices_cmip6":
        for s in all_sc:
            for n in names:
                yield f"{s} {n}", cmip6_index_output(cfg, s, n)
    elif folder == "split":
        for s in all_sc:
            for n in names:
                yield f"{s} {n}", split_output(cfg, s, n)
    elif folder == "hindcast_index":
        for s in hc_sc:
            yield s, hindcast_output(cfg, s)
    elif folder == "evaluate_index":
        for s in hc_sc:
            for n in hindcast_indices(cfg):
                yield f"{s} {n}", evaluation_output(cfg, s, n)
    elif folder == "linear_model":
        for s in all_sc:
            for tv in targets:
                yield f"{s} {tv}", linear_model_output(cfg, s, tv)
    elif folder == "hindcast_target":
        for s in hc_sc:
            for tv in targets:
                yield f"{s} {tv}", target_hindcast_output(cfg, s, tv)
    elif folder == "evaluate_target":
        for s in hc_sc:
            for tv in targets:
                yield f"{s} {tv}", target_evaluation_output(cfg, s, tv)
    elif folder == "partition":
        from climate_scenarios_2050 import partition

        yield from partition.outputs(cfg)
    elif folder == "forecast":
        # as in the forecast step: the scenarios whose split exists
        available = [s for s in all_sc if split_output(cfg, s).exists("basis.nc")]
        if available:
            for tv in targets:
                yield tv, forecast_output(cfg, available, tv, windows)


def run(cfg, steps=None, scenarios=None, target_variable=None, windows=None, print_fn=print):
    targets = [target_variable] if target_variable else active_targets(cfg)
    windows = [tuple(w) for w in (windows or cfg["target"]["windows"])]
    roots = cfg["paths"]["outputs"]
    folders = [f for step in (steps or STEP_FOLDERS) for f in STEP_FOLDERS[step]]
    print_fn(f"outputs: {roots['root']}\nCMIP6 stores: {roots['cmip6_processed']}\nfigures: {roots['figures']}")
    rows = []
    with read_only():
        for folder in folders:
            print_fn(f"\n{folder}")
            for case, out in _cases(cfg, folder, scenarios, targets, windows):
                path, digest = (out, out.name) if isinstance(out, Path) else (out.dir, out.hash)
                status = _status(path)
                fig = ""
                if folder in FIGURES and (Path(roots["figures"]) / folder / digest).exists():
                    fig = "  (figures)"
                print_fn(f"  {case:<32} {digest}  {status:<5}{fig}")
                rows.append({"folder": folder, "case": case, "hash": digest, "status": status, "path": str(path)})
    return rows
