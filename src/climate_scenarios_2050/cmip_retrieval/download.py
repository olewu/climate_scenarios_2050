"""
Download of the datasets with status `download` in an inventory plan.

Per dataset, the file records (with SHA256 checksums and the HTTP urls of all copies) are searched on
the index nodes. Files outside `cmip6.period` (e.g. beyond 2100) are skipped. Every file is downloaded
on its own: written to `<target>/.partial/`, checked against its checksum and only then moved to
`<target>/<experiment>/<model>/<variable>/`. A failing copy is retried from the next data node; a file
that fails everywhere is logged and the download continues with the next file. Existing files are never
overwritten, files are not merged (the preprocessing joins the files of a member).

Variables listed in `download.subsets` (e.g. `ua`) are not downloaded as whole files: the subset (one
pressure level, a latitude band) is read through OPeNDAP and written as
`<experiment>/<model>/<variable>_<tag>/<file name without .nc>_<tag>.nc`. There is no checksum for a
subset; instead every month of the file's range must be present and no time step may be empty.

Every file is appended to `<inventory>/download_log.jsonl` as soon as it is done; a rerun skips files
already in place.
"""

import hashlib
import json
import logging
import os
import shutil
import functools
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from climate_scenarios_2050.cmip_retrieval import esgf
from climate_scenarios_2050.core.config import cmip6_table
from climate_scenarios_2050.cmip_retrieval.inventory import FILE_PATTERN, latest_inventory_dir, subset_dir, subset_file

log = logging.getLogger(__name__)
CHUNK = 1 << 20
TIME_CHUNK = 120   # months per OPeNDAP request


def dataset_files(row, nodes, table, opendap=False) -> list:
    """[{name, size, checksum, checksum_type, urls}] of one planned dataset (all copies merged)"""
    facets = {"table_id": table, "variable_id": row["variable"], "experiment_id": row["experiment"], "source_id": row["model"],
              "variant_label": row["member"], "grid_label": row["grid"]}
    version = f"v{int(row['version'])}"
    files = {}
    for node in nodes:
        try:
            docs = esgf.search(node, "File", **facets)
        except Exception as exc:
            log.warning(f"file search on {node} failed for {row['dataset_id']}: {exc!r}")
            continue
        for doc in docs:
            if doc.get("retracted") in (True, "true") or f".{version}." not in doc["instance_id"]:
                continue
            name = esgf.first(doc.get("title"))
            entry = files.setdefault(name, {"name": name, "size": int(esgf.first(doc.get("size")) or 0), "checksum": esgf.first(doc.get("checksum")),
                                            "checksum_type": (esgf.first(doc.get("checksum_type")) or "").upper(), "urls": [], "opendap": []})
            entry["urls"] += [u for u in esgf.http_urls(doc) if u not in entry["urls"]]
            entry["opendap"] += [u for u in esgf.opendap_urls(doc) if u not in entry["opendap"]]
        if files and all(f["opendap"] if opendap else f["urls"] for f in files.values()):
            break  # copies found; the other index nodes are only asked if needed
    return list(files.values())


def in_period(name, period) -> bool:
    match = FILE_PATTERN.match(name)
    if match is None or match["start"] is None:
        return True
    return int(match["start"][:4]) <= period[1] and int(match["end"][:4]) >= period[0]


def _checksum(path, kind) -> str:
    digest = hashlib.sha256() if kind == "SHA256" else hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_resuming(url, partial: Path, size, timeout, attempts):
    """download url to partial; broken connections are resumed with a range request (like wget -c)"""
    partial.unlink(missing_ok=True)
    for attempt in range(attempts):
        have = partial.stat().st_size if partial.exists() else 0
        if size and have >= size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                if have and r.status_code != 206:   # server ignores ranges: start again
                    have = 0
                with open(partial, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
            if not size or partial.stat().st_size >= size:
                return
        except requests.HTTPError:
            raise   # 4xx/5xx: try the next copy
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            log.info(f"{partial.name}: {exc.__class__.__name__} after {partial.stat().st_size if partial.exists() else 0} bytes, resuming")
    raise IOError(f"incomplete after {attempts} attempts")


def fetch(file, destination: Path, partial_dir: Path, timeout=(30, 300), attempts=5) -> dict:
    """download one file to destination via partial_dir; returns a log record"""
    record = {"file": file["name"], "destination": str(destination), "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    if destination.exists():
        return {**record, "status": "exists"}
    if file["checksum_type"] not in ("SHA256", "MD5"):
        return {**record, "status": "failed", "error": f"unknown checksum type {file['checksum_type']!r}"}
    partial = partial_dir / file["name"]
    errors = []
    for url in file["urls"]:
        try:
            _download_resuming(url, partial, file.get("size"), timeout, attempts)
            if _checksum(partial, file["checksum_type"]) != file["checksum"]:
                raise ValueError("checksum mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # never replace a file that appeared in the meantime
            if destination.exists():
                partial.unlink()
                return {**record, "status": "exists"}
            shutil.move(str(partial), str(destination))
            return {**record, "status": "done", "url": url, "size": os.path.getsize(destination)}
        except Exception as exc:
            errors.append(f"{url}: {exc!r}")
            partial.unlink(missing_ok=True)
    return {**record, "status": "failed", "error": " | ".join(errors) or "no http url"}


def _months(name) -> int:
    match = FILE_PATTERN.match(name)
    start, end = int(match["start"]), int(match["end"])
    return (end // 100 - start // 100) * 12 + end % 100 - start % 100 + 1


def read_subset(url, variable, spec) -> xr.Dataset:
    """one pressure level and a latitude band of one file through OPeNDAP"""
    with xr.open_dataset(url, decode_times=False) as ds:
        plev = ds["plev"].values
        target = spec["plev"] if np.nanmax(plev) > 2000 else spec["plev"] / 100   # Pa or hPa
        level = np.flatnonzero(np.abs(plev - target) < 1e-3 * target)
        if level.size != 1:
            raise ValueError(f"level {target} not in {plev}")
        lat = ds["lat"].values
        if lat.ndim != 1:
            raise ValueError("2-d latitude not supported")
        rows = np.flatnonzero((lat >= spec["lat"][0]) & (lat <= spec["lat"][1]))
        rows = slice(int(rows.min()), int(rows.max()) + 1)
        n = ds.sizes["time"]
        parts = [ds[variable].isel(plev=int(level[0]), lat=rows, time=slice(t, t + TIME_CHUNK)).load() for t in range(0, n, TIME_CHUNK)]
        out = xr.concat(parts, dim="time").to_dataset()
        for name in ["time_bnds", "lat_bnds", "lon_bnds"]:
            if name in ds:
                sel = {"lat": rows} if name == "lat_bnds" and "lat" in ds[name].dims else {}
                out[name] = ds[name].isel(sel).load()
        out.attrs = dict(ds.attrs)
    return xr.decode_cf(out, use_cftime=True)


def fetch_subset(file, destination: Path, partial_dir: Path, variable, spec, attempts=3) -> dict:
    record = {"file": file["name"], "destination": str(destination), "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    if destination.exists():
        return {**record, "status": "exists"}
    partial = partial_dir / destination.name
    errors = []
    for url in file["opendap"]:
        for attempt in range(attempts):
            try:
                ds = read_subset(url, variable, spec)
                if ds.sizes["time"] != _months(file["name"]):
                    raise ValueError(f"{ds.sizes['time']} time steps, expected {_months(file['name'])}")
                if bool(ds[variable].isnull().all(dim=[d for d in ds[variable].dims if d != "time"]).any()):
                    raise ValueError("time steps without data")
                ds.attrs.update({"subset": spec["tag"], "subset_source": url, "subset_source_sha256": str(file["checksum"])})
                ds.to_netcdf(partial, encoding={variable: {"zlib": True, "complevel": 1, "dtype": "float32"}})
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    partial.unlink()
                    return {**record, "status": "exists"}
                shutil.move(str(partial), str(destination))
                return {**record, "status": "done", "url": url, "size": os.path.getsize(destination)}
            except Exception as exc:
                partial.unlink(missing_ok=True)
                errors.append(f"{url} (attempt {attempt + 1}): {exc!r}")
                if isinstance(exc, ValueError):
                    break   # content problem: no point in retrying this copy
    return {**record, "status": "failed", "error": " | ".join(errors) or "no opendap url"}


def _log(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def run(cfg, target=None, inventory=None, variables=None, experiments=None, max_datasets=None) -> pd.DataFrame:
    """download the planned datasets; `target` defaults to paths.inputs.cmip6"""
    dcfg = cfg["download"]
    target = Path(target or cfg["paths"]["inputs"]["cmip6"])
    inventory = Path(inventory) if inventory else latest_inventory_dir(cfg)
    planned = pd.read_csv(inventory / "plan.csv")
    todo = planned[planned["status"] == "download"]
    if variables:
        todo = todo[todo["variable"].isin(variables)]
    if experiments:
        # scenarios bring the historical runs of their members along
        scenarios = [e for e in experiments if e != "historical"]
        if scenarios:
            members = planned[planned["experiment"].isin(scenarios) & (planned["status"] != "missing")][["variable", "model", "member"]].drop_duplicates()
            hist = todo[todo["experiment"] == "historical"].merge(members)
            todo = pd.concat([todo[todo["experiment"].isin(scenarios)], hist]).sort_values(["variable", "model", "member", "experiment"])
        else:
            todo = todo[todo["experiment"] == "historical"]
    if max_datasets:
        todo = todo.head(max_datasets)
    partial_dir = target / ".partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    log_file = inventory / "download_log.jsonl"
    subsets = dcfg.get("subsets") or {}
    log.info(f"{len(todo)} datasets ({todo['size'].sum() / 1e9:.1f} GB as full files) from plan {inventory} -> {target}, log {log_file}")

    period = cfg["cmip6"]["period"]
    results = []
    lock = threading.Lock()

    def finished(job, base):
        """log every file as soon as it is done (also if the run is interrupted later)"""
        try:
            record = {**base, **job.result()}
        except Exception as exc:
            record = {**base, "status": "failed", "error": repr(exc)}
        with lock:
            _log(log_file, record)
            results.append(record)
            if len(results) % 200 == 0:
                log.info(f"{len(results)} files done")
        if record["status"] == "failed":
            log.warning(f"failed: {record.get('file', record['dataset_id'])}: {record['error'][:300]}")

    with ProcessPoolExecutor(dcfg["workers"]) as pool:
        for _, row in todo.iterrows():
            try:
                files = dataset_files(row, dcfg["index_nodes"], cmip6_table(cfg, row["variable"]), opendap=row["variable"] in subsets)
            except Exception as exc:
                files, error = [], repr(exc)
            else:
                error = "no files found"
            base = {"dataset_id": row["dataset_id"], "variable": row["variable"], "experiment": row["experiment"], "model": row["model"], "member": row["member"]}
            if not files:
                record = {**base, "status": "failed", "error": error}
                with lock:
                    _log(log_file, record)
                    results.append(record)
                continue
            for file in files:
                if not in_period(file["name"], period):
                    continue
                folder = target / row["experiment"] / row["model"]
                if row["variable"] in subsets:
                    spec = subsets[row["variable"]]
                    destination = folder / subset_dir(row["variable"], spec["tag"]) / subset_file(file["name"], spec["tag"])
                    job = pool.submit(fetch_subset, file, destination, partial_dir, row["variable"], spec)
                else:
                    destination = folder / row["variable"] / file["name"]
                    job = pool.submit(fetch, file, destination, partial_dir)
                job.add_done_callback(functools.partial(finished, base=base))
        # leaving the block waits for all files

    df = pd.DataFrame(results)
    if not df.empty:
        log.info(f"\n{df.groupby(['variable', 'experiment', 'status']).size().unstack(fill_value=0)}")
    return df
