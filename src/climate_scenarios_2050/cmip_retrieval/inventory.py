"""
Inventory of the CMIP6 monthly data on ESGF and in the local archive, and the download plan.

- remote: one row per dataset (experiment, model, member, grid) in its latest version, from the
  dataset records of all index nodes (size and number of files, data nodes holding a copy)
- local: one row per (experiment, model, member, grid) found below the archive root
  (<root>/<experiment>/<model>/<variable>/<variable>_<table>_<model>_<experiment>_<member>_<grid>_<range>.nc)
- plan: per variable, every member that has `historical` and at least one of the scenarios (on ESGF or
  locally) - such a member can be used in the pipeline. Experiments of these members that are not in the
  archive are downloaded, on the grid the member already has locally, else the native grid `gn` if
  offered. Members already in the archive are never touched. Variables in `download.subsets` count as
  local if their subset files exist.

The tables are written to `<outputs.root>/cmip6_inventory/<date>/`, a new folder per run.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from climate_scenarios_2050.cmip_retrieval import esgf
from climate_scenarios_2050.core.config import cmip6_table

log = logging.getLogger(__name__)

FILE_PATTERN = re.compile(
    r"^(?P<variable>[A-Za-z0-9]+)_(?P<table>[A-Za-z0-9]+)_(?P<model>[A-Za-z0-9-]+)_(?P<experiment>[A-Za-z0-9-]+)_"
    r"(?P<member>r\d+i\d+p\d+f\d+)_(?P<grid>g[a-z0-9]+)(?:_(?P<start>\d{6})-(?P<end>\d{6}))?\.nc$"
)
KEY = ["variable", "experiment", "model", "member", "grid"]
PLEV19 = 19   # levels of the Amon pressure level variables


def subset_dir(variable, tag) -> str:
    return f"{variable}_{tag}"


def subset_file(name, tag) -> str:
    return name.removesuffix(".nc") + f"_{tag}.nc"


def subset_fraction(spec) -> float:
    """rough size of a subset relative to the full file: one of 19 levels, part of the latitudes"""
    return 1 / PLEV19 * (spec["lat"][1] - spec["lat"][0]) / 180


def inventory_dir(cfg) -> Path:
    base = Path(cfg["paths"]["outputs"]["root"]) / "cmip6_inventory"
    path = base / f"{datetime.now():%Y%m%d}"
    if path.exists():
        path = base / f"{datetime.now():%Y%m%d-%H%M%S}"
    path.mkdir(parents=True)
    return path


def latest_inventory_dir(cfg) -> Path:
    base = Path(cfg["paths"]["outputs"]["root"]) / "cmip6_inventory"
    found = sorted(p for p in base.glob("*") if (p / "plan.csv").exists())
    if not found:
        raise FileNotFoundError(f"no inventory below {base}, run `pixi run cmip6-inventory` first")
    return found[-1]


def remote_datasets(variables, experiments, tables=None, nodes=None) -> tuple:
    """`tables`: {variable: CMIP6 table}, default Amon"""
    """(DataFrame of the latest version of every dataset, {(variable, experiment): {node: error}})"""
    rows, failures = [], {}
    for variable in variables:
        for experiment in experiments:
            docs, failed = esgf.search_all(nodes, "Dataset", table_id=(tables or {}).get(variable, "Amon"), variable_id=variable, experiment_id=experiment)
            if failed:
                failures[(variable, experiment)] = failed
            for doc in docs:
                if doc.get("retracted") in (True, "true"):
                    continue
                rows.append({
                    "variable": variable,
                    "experiment": experiment,
                    "model": esgf.first(doc.get("source_id")),
                    "member": esgf.first(doc.get("variant_label")),
                    "grid": esgf.first(doc.get("grid_label")),
                    "version": int(str(esgf.first(doc.get("version"))).lstrip("v")),
                    "size": int(esgf.first(doc.get("size")) or 0),
                    "n_files": int(esgf.first(doc.get("number_of_files")) or 0),
                    "end": _year(doc.get("datetime_stop")),
                    "data_node": esgf.first(doc.get("data_node")),
                    "index_node": doc["index_node"],
                    "dataset_id": doc["instance_id"],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, failures
    # latest version of each dataset; collect the data nodes holding it
    df = df[df["version"] == df.groupby(KEY)["version"].transform("max")]
    agg = df.groupby(KEY, as_index=False).agg(
        version=("version", "first"), size=("size", "max"), n_files=("n_files", "max"), end=("end", "max"), dataset_id=("dataset_id", "first"),
        data_nodes=("data_node", lambda x: " ".join(sorted(set(x)))), index_nodes=("index_node", lambda x: " ".join(sorted(set(x)))),
    )
    return agg, failures


def _year(value):
    """year of an ESGF datetime ("2100-12-16T12:00:00Z"); NaN if unknown (not all index nodes have it)"""
    value = esgf.first(value)
    return float(str(value)[:4]) if value else np.nan


def local_datasets(root, variables, experiments, tables=None, subsets=None) -> pd.DataFrame:
    """files below root; variables in `subsets` are looked up in their subset folders"""
    rows = []
    root = Path(root)
    subsets = subsets or {}
    for experiment in experiments:
        for variable in variables:
            tag = subsets.get(variable, {}).get("tag")
            folder = subset_dir(variable, tag) if tag else variable
            table = (tables or {}).get(variable, "Amon")
            for fl in root.glob(f"{experiment}/*/{folder}/{variable}_{table}_*.nc"):
                name = fl.name.replace(f"_{tag}.nc", ".nc") if tag else fl.name
                match = FILE_PATTERN.match(name)
                if match is None or match["experiment"] != experiment:
                    continue
                size = fl.stat().st_size
                if size == 0:
                    continue
                rows.append({**{k: match[k] for k in KEY}, "start": match["start"], "end": match["end"], "size": size, "file": str(fl)})
    df = pd.DataFrame(rows, columns=[*KEY, "start", "end", "size", "file"])
    out = df.groupby(KEY, as_index=False).agg(n_files=("file", "size"), size=("size", "sum"), start=("start", "min"), end=("end", "max"))
    out["end"] = pd.to_numeric(out["end"].str[:4], errors="coerce")
    return out


def is_short(df: pd.DataFrame, scenario_end) -> pd.Series:
    """scenario runs ending before scenario_end (unknown end: not short)"""
    return (df["experiment"] != "historical") & df["end"].notna() & (df["end"] < scenario_end)


def plan(remote: pd.DataFrame, local: pd.DataFrame, scenarios, prefer_grid="gn", scenario_end=2055) -> pd.DataFrame:
    """
    one row per (variable, experiment, model, member) of usable members, with `status`
    local | download | missing | short, and the grid and size of the dataset to download.
    Scenario runs ending before `scenario_end` (e.g. CNRM-CM6-1 ssp245 r7-r10, 2015-2020) are
    `short`: not downloaded, and they do not make a member usable.
    """
    members = ["variable", "model", "member"]
    remote_ok = remote[~is_short(remote, scenario_end)]
    local_ok = local[~is_short(local, scenario_end)]
    available = pd.concat([remote_ok[[*members, "experiment"]], local_ok[[*members, "experiment"]]]).drop_duplicates()
    has = available.assign(flag=True).pivot_table(index=members, columns="experiment", values="flag", aggfunc="any", fill_value=False)
    for exp in ["historical", *scenarios]:
        if exp not in has:
            has[exp] = False
    usable = has[has["historical"] & has[list(scenarios)].any(axis=1)].reset_index()[members]

    local_grid = local.groupby(members)["grid"].first()
    rows = []
    for _, m in usable.iterrows():
        key = (m["variable"], m["model"], m["member"])
        for exp in ["historical", *scenarios]:
            row = {"variable": key[0], "experiment": exp, "model": key[1], "member": key[2]}
            here = local[(local["variable"] == key[0]) & (local["model"] == key[1]) & (local["member"] == key[2]) & (local["experiment"] == exp)]
            if not here.empty:
                short = bool(is_short(here, scenario_end).all())
                rows.append({**row, "status": "short" if short else "local", "grid": here["grid"].iloc[0], "size": int(here["size"].sum()), "end": here["end"].max()})
                continue
            offered = remote[(remote["variable"] == key[0]) & (remote["model"] == key[1]) & (remote["member"] == key[2]) & (remote["experiment"] == exp)]
            if offered.empty:
                rows.append({**row, "status": "missing", "grid": None, "size": 0})
                continue
            if is_short(offered, scenario_end).all():
                rows.append({**row, "status": "short", "grid": None, "size": 0, "end": offered["end"].max()})
                continue
            offered = offered[~is_short(offered, scenario_end)]
            grids = list(offered["grid"])
            grid, note = local_grid.get(key), ""
            if grid not in grids:
                # the other experiments of the member are on a grid not offered here: the member cannot be joined
                note = f"local grid {grid} not offered" if grid is not None else ""
                grid = prefer_grid if prefer_grid in grids else sorted(grids)[0]
            pick = offered[offered["grid"] == grid].iloc[0]
            rows.append({**row, "status": "download", "grid": grid, "size": int(pick["size"]), "end": pick["end"], "n_files": int(pick["n_files"]),
                         "version": int(pick["version"]), "dataset_id": pick["dataset_id"], "data_nodes": pick["data_nodes"], "note": note})
    return pd.DataFrame(rows)


def summary(planned: pd.DataFrame, local: pd.DataFrame, subsets=None) -> pd.DataFrame:
    """
    members and GB per (variable, experiment): in the archive, usable after the download, to download
    (for subset variables a rough estimate from the full size)
    """
    gb = 1e9
    factor = planned["variable"].map({v: subset_fraction(spec) for v, spec in (subsets or {}).items()}).fillna(1.0)
    planned = planned.assign(size=planned["size"] * factor)
    grouped = planned.groupby(["variable", "experiment"])
    out = pd.DataFrame({
        "members_local_all": local.groupby(["variable", "experiment"]).size(),
        "members_local_usable": grouped["status"].apply(lambda s: (s == "local").sum()),
        "members_download": grouped["status"].apply(lambda s: (s == "download").sum()),
        "members_after": grouped["status"].apply(lambda s: s.isin(["local", "download"]).sum()),
        "models_after": planned[planned["status"].isin(["local", "download"])].groupby(["variable", "experiment"])["model"].nunique(),
        "members_short": grouped["status"].apply(lambda s: (s == "short").sum()),
        "GB_local": local.groupby(["variable", "experiment"])["size"].sum() / gb,
        "GB_download": planned[planned["status"] == "download"].groupby(["variable", "experiment"])["size"].sum() / gb,
    }).fillna(0)
    return out.astype({c: int for c in out if not c.startswith("GB")}).round(1)


def run(cfg, variables=None, experiments=None) -> Path:
    dcfg = cfg["download"]
    variables = variables or dcfg["variables"]
    scenarios = [e for e in (experiments or dcfg["experiments"]) if e != "historical"]
    experiments = ["historical", *scenarios]
    out = inventory_dir(cfg)
    log.info(f"CMIP6 inventory of {variables} {experiments} -> {out}")

    tables = {v: cmip6_table(cfg, v) for v in variables}
    remote, failures = remote_datasets(variables, experiments, tables, dcfg["index_nodes"])
    remote.to_csv(out / "remote.csv", index=False)
    if failures:
        pd.DataFrame([{"variable": v, "experiment": e, "node": n, "error": err} for (v, e), f in failures.items() for n, err in f.items()]).to_csv(out / "failed_queries.csv", index=False)
        log.warning(f"{len(failures)} queries failed on at least one index node, see {out / 'failed_queries.csv'}")
    local = local_datasets(cfg["paths"]["inputs"]["cmip6"], variables, experiments, tables, dcfg.get("subsets"))
    local.to_csv(out / "local.csv", index=False)
    planned = plan(remote, local, scenarios, dcfg["prefer_grid"], cfg["cmip6"]["full_period"][1])
    planned.to_csv(out / "plan.csv", index=False)
    table = summary(planned, local, dcfg.get("subsets"))
    table.to_csv(out / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info(f"\n{table}\ntotal to download: {table['GB_download'].sum():.0f} GB")
    return out
