"""
Download of the datasets with status `download` in an inventory plan.

Per dataset, the file records (with SHA256 checksums and the HTTP urls of all copies) are searched on
all index nodes and merged (one index node may list only some of the copies). Files outside `cmip6.period` (e.g. beyond 2100) are skipped. Every file is downloaded
on its own: written to `<target>/.partial/`, checked against its checksum and only then moved to
`<target>/<experiment>/<model>/<variable>/`. A failing copy is retried from the next data node; a file
that fails everywhere is logged and the download continues with the next file. Existing files are never
overwritten, files are not merged (the preprocessing joins the files of a member).

Variables listed in `download.subsets` (e.g. `ua`) are not downloaded as whole files: the subset (one
pressure level, a latitude band) is read through OPeNDAP and written as
`<experiment>/<model>/<variable>_<tag>/<file name without .nc>_<tag>.nc`. There is no checksum for a
subset; instead every month of the file's range must be present and no time step may be empty.
If no OPeNDAP copy works (the DKRZ OPeNDAP service answered 404 for every file in 2026-11), the full file
is downloaded over HTTP with its checksum, cut locally and deleted (`subset_method: http+cut` in the file).

Every file is appended to `<inventory>/download_log.jsonl` as soon as it is done; a rerun skips files
already in place.

Unreachable servers: a connection that fails before any data arrived is not retried (only a broken
transfer is resumed), and a server (HTTP or OPeNDAP service of a host) that fails `download.skip_host_after`
times in a row is paused for `download.skip_host_minutes` and then probed again (shared between the workers).
Files whose servers are paused are deferred and retried in further rounds after the pause (up to
`download.deferred_rounds`, as long as a round brings files in), and only then logged as failed. Requests to
one server are spaced by `download.min_request_interval` seconds (rate limits). A dead server
otherwise costs minutes per file (connect timeouts x attempts x copies) for thousands of files.
"""

import hashlib
import json
import logging
import os
import shutil
import functools
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from urllib.parse import urlparse

from climate_scenarios_2050.cmip_retrieval import esgf
from climate_scenarios_2050.core.config import cmip6_table
from climate_scenarios_2050.cmip_retrieval.inventory import FILE_PATTERN, latest_inventory_dir, subset_dir, subset_file

log = logging.getLogger(__name__)
CHUNK = 1 << 20
TIME_CHUNK = 120   # months per OPeNDAP request


def dataset_files(row, nodes, table) -> list:
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
            if doc.get("retracted") in (True, "true") or version not in doc["instance_id"].split("."):
                continue
            name = esgf.first(doc.get("title"))
            entry = files.setdefault(name, {"name": name, "size": int(esgf.first(doc.get("size")) or 0), "checksum": esgf.first(doc.get("checksum")),
                                            "checksum_type": (esgf.first(doc.get("checksum_type")) or "").upper(), "urls": [], "opendap": []})
            entry["urls"] += [u for u in esgf.http_urls(doc) if u not in entry["urls"]]
            entry["opendap"] += [u for u in esgf.opendap_urls(doc) if u not in entry["opendap"]]
        # no early stop: an index node may know only some copies (CEDA lists EC-Earth3 only on
        # esg-dn1.nsc.liu.se, ORNL also has the ORNL and ALCF replicas), so all nodes are asked
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


# errors that mean "server not reachable / not answering" (counted per server, see HostHealth)
CONNECTION_ERRORS = (requests.ConnectionError, requests.Timeout)
OPENDAP_SERVER_ERRORS = ("I/O failure", "DAP failure", "Access failure", "file not found", "timed out", "Timeout")


class HostHealth:
    """
    connection failures per server, shared between the worker processes (a Manager dict) or local (plain
    dict). After `limit` failures in a row a server is paused for `pause` seconds; afterwards it is tried
    again (a failed probe pauses it again, a success makes it normal again). Servers that are down for
    days cost one probe per pause. Files whose servers are all paused are deferred, not failed (see `run`).

    `interval`: minimum time between two requests to the same server (all workers together). The EC-Earth
    server esg-dn1.nsc.liu.se stopped answering twice after ~18,000 small files in ~30 min (2026-11-18/19),
    most likely a rate limit.
    """

    def __init__(self, store=None, limit=5, pause=900, interval=0.0, turns=None, lock=None):
        self.store = {} if store is None else store
        self.limit = limit
        self.pause = pause
        self.interval = interval
        self.turns = {} if turns is None else turns
        self.lock = lock or threading.Lock()

    def wait_turn(self, url):
        """sleep until this worker may send the next request to the server of `url`"""
        if not self.interval:
            return
        host = urlparse(url).netloc
        with self.lock:
            now = time.time()
            turn = max(now, self.turns.get(host, 0.0))
            self.turns[host] = turn + self.interval
        if turn > now:
            time.sleep(turn - now)

    def next_probe(self) -> float:
        """seconds until the first paused server may be probed again (0: none paused)"""
        now = time.time()
        waits = [until - now for n, until, _ in self.store.values() if n >= self.limit]
        return max(0.0, min(waits)) if waits else 0.0

    @staticmethod
    def key(url, service):
        return f"{service}:{urlparse(url).netloc}"

    def _get(self, key):
        return self.store.get(key, (0, 0.0, 0))   # failures in a row, paused until (epoch s), number of pauses

    def alive(self, url, service) -> bool:
        n, until, _ = self._get(self.key(url, service))
        return n < self.limit or time.time() >= until

    def failed(self, url, service):
        key = self.key(url, service)
        n, until, pauses = self._get(key)
        n += 1
        if n >= self.limit:
            until, pauses = time.time() + self.pause, pauses + 1
            if pauses == 1:
                log.warning(f"{key}: {n} connection failures in a row, paused for {self.pause / 60:.0f} min (then probed again)")
        self.store[key] = (n, until, pauses)

    def ok(self, url, service):
        key = self.key(url, service)
        n, _, pauses = self._get(key)
        if n >= self.limit:
            log.warning(f"{key}: reachable again")
        if n:
            self.store[key] = (0, 0.0, pauses)

    def dead(self) -> list:
        """servers paused at least once, with the number of pauses (still failing at the end: marked)"""
        return sorted(f"{k} ({p} pauses{', still unreachable' if n >= self.limit else ''})" for k, (n, _, p) in self.store.items() if p)


def _download_resuming(url, partial: Path, size, timeout, attempts):
    """
    download url to partial; broken connections are resumed with a range request (like wget -c); an
    attempt that got no data at all (connect timeout, refused, ...) is not repeated
    """
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
            got = (partial.stat().st_size if partial.exists() else 0) - have
            if attempt == attempts - 1 or (isinstance(exc, CONNECTION_ERRORS) and got <= 0):
                raise
            log.info(f"{partial.name}: {exc.__class__.__name__} after {partial.stat().st_size if partial.exists() else 0} bytes, resuming")
    raise IOError(f"incomplete after {attempts} attempts")


def fetch(file, destination: Path, partial_dir: Path, timeout=(30, 300), attempts=5, health: HostHealth = None) -> dict:
    """download one file to destination via partial_dir; returns a log record"""
    health = health or HostHealth()
    record = {"file": file["name"], "destination": str(destination), "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    if destination.exists():
        return {**record, "status": "exists"}
    if file["checksum_type"] not in ("SHA256", "MD5"):
        return {**record, "status": "failed", "error": f"unknown checksum type {file['checksum_type']!r}"}
    partial = partial_dir / file["name"]
    errors, paused = [], False
    for url in file["urls"]:
        if not health.alive(url, "http"):
            errors.append(f"{url}: server paused (unreachable)")
            paused = True
            continue
        try:
            try:
                health.wait_turn(url)
                _download_resuming(url, partial, file.get("size"), timeout, attempts)
            except CONNECTION_ERRORS:
                health.failed(url, "http")
                paused = True   # server not reachable now: retry this file in a later round as well
                raise
            health.ok(url, "http")
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
    # a copy on a paused server may still work later: deferred, not failed
    return {**record, "status": "deferred" if paused else "failed", "error": " | ".join(errors) or "no http url"}


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


def _write_subset(ds, file, source, partial: Path, destination: Path, variable, spec, record, method) -> dict:
    """check a subset (every month present, no empty time step) and move it into place"""
    if ds.sizes["time"] != _months(file["name"]):
        raise ValueError(f"{ds.sizes['time']} time steps, expected {_months(file['name'])}")
    if bool(ds[variable].isnull().all(dim=[d for d in ds[variable].dims if d != "time"]).any()):
        raise ValueError("time steps without data")
    ds.attrs.update({"subset": spec["tag"], "subset_source": source, "subset_source_sha256": str(file["checksum"]), "subset_method": method})
    ds.to_netcdf(partial, encoding={variable: {"zlib": True, "complevel": 1, "dtype": "float32"}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        partial.unlink()
        return {**record, "status": "exists"}
    shutil.move(str(partial), str(destination))
    return {**record, "status": "done", "url": source, "method": method, "size": os.path.getsize(destination)}


def fetch_subset(file, destination: Path, partial_dir: Path, variable, spec, attempts=3, health: HostHealth = None) -> dict:
    """
    subset through OPeNDAP; if no copy works (e.g. the DKRZ OPeNDAP service answers 404 for every file),
    the full file is downloaded over HTTP (checksum verified), cut locally and deleted again
    """
    health = health or HostHealth()
    record = {"file": file["name"], "destination": str(destination), "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    if destination.exists():
        return {**record, "status": "exists"}
    partial = partial_dir / f"subset_{destination.name}"
    errors, paused = [], False
    for url in file["opendap"]:
        if not health.alive(url, "opendap"):
            errors.append(f"{url}: OPeNDAP service paused (unreachable)")
            paused = True
            continue
        for attempt in range(attempts):
            try:
                health.wait_turn(url)
                ds = read_subset(url, variable, spec)
            except Exception as exc:
                errors.append(f"{url} (attempt {attempt + 1}): {exc!r}")
                if any(text in str(exc) for text in OPENDAP_SERVER_ERRORS):
                    health.failed(url, "opendap")
                    break   # server/service problem: no point in retrying this copy
                continue
            health.ok(url, "opendap")
            try:
                return _write_subset(ds, file, url, partial, destination, variable, spec, record, "opendap")
            except Exception as exc:
                partial.unlink(missing_ok=True)
                errors.append(f"{url}: {exc!r}")
                break   # content problem: no point in retrying this copy
    if not file["urls"]:
        return {**record, "status": "deferred" if paused else "failed", "error": " | ".join(errors) or "no opendap or http url"}
    # fallback: whole file over HTTP, then cut
    full_dir = partial_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full = full_dir / file["name"]
    try:
        got = fetch(file, full, partial_dir, health=health)
        if got["status"] != "done":
            errors.append(f"http fallback: {got.get('error', got['status'])}")
            status = "deferred" if paused or got["status"] == "deferred" else "failed"
            return {**record, "status": status, "error": " | ".join(errors)}
        try:
            return {**_write_subset(read_subset(str(full), variable, spec), file, got["url"], partial, destination, variable, spec, record, "http+cut"),
                    "opendap_errors": len(errors)}
        except Exception as exc:
            partial.unlink(missing_ok=True)
            errors.append(f"cut of {got['url']}: {exc!r}")
            return {**record, "status": "failed", "error": " | ".join(errors)}
    finally:
        full.unlink(missing_ok=True)


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
    results, deferred, jobs = [], [], []
    handled = [0]   # callbacks run after `wait` may already have returned: count them
    lock = threading.Lock()

    def record_result(record):
        with lock:
            _log(log_file, record)
            results.append(record)
            if len(results) % 200 == 0:
                # successes are only in the log file; the terminal shows failures, so give the counts here
                counts = pd.Series([r["status"] for r in results]).value_counts().to_dict()
                log.info(f"{len(results)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" (current: {record['variable']} {record['experiment']} {record['model']})")
        if record["status"] == "failed":
            log.warning(f"failed: {record.get('file', record['dataset_id'])}: {record['error'][:300]}")

    def finished(job, base, task):
        """log every file as soon as it is done (also if the run is interrupted later); keep deferred files"""
        try:
            record = {**base, **job.result()}
        except Exception as exc:
            record = {**base, "status": "failed", "error": repr(exc)}
        if record["status"] == "deferred":
            with lock:
                deferred.append((task, base, record))
        else:
            record_result(record)
        with lock:
            handled[0] += 1

    def wait_all():
        wait(jobs)
        while handled[0] < len(jobs):
            time.sleep(0.1)

    def submit(pool, task, base):
        fn, args = task
        job = pool.submit(fn, *args, health=health)
        job.add_done_callback(functools.partial(finished, base=base, task=task))
        jobs.append(job)

    manager = Manager()
    health = HostHealth(manager.dict(), dcfg.get("skip_host_after", 5), 60 * dcfg.get("skip_host_minutes", 15),
                        dcfg.get("min_request_interval", 0.0), manager.dict(), manager.Lock())
    with manager:
        with ProcessPoolExecutor(dcfg["workers"]) as pool:
            for _, row in todo.iterrows():
                try:
                    files = dataset_files(row, dcfg["index_nodes"], cmip6_table(cfg, row["variable"]))
                except Exception as exc:
                    files, error = [], repr(exc)
                else:
                    error = "no files found"
                base = {"dataset_id": row["dataset_id"], "variable": row["variable"], "experiment": row["experiment"], "model": row["model"], "member": row["member"]}
                if not files:
                    record_result({**base, "status": "failed", "error": error})
                    continue
                for file in files:
                    if not in_period(file["name"], period):
                        continue
                    folder = target / row["experiment"] / row["model"]
                    if row["variable"] in subsets:
                        spec = subsets[row["variable"]]
                        destination = folder / subset_dir(row["variable"], spec["tag"]) / subset_file(file["name"], spec["tag"])
                        submit(pool, (fetch_subset, (file, destination, partial_dir, row["variable"], spec)), base)
                    else:
                        destination = folder / row["variable"] / file["name"]
                        submit(pool, (fetch, (file, destination, partial_dir)), base)
            wait_all()
            # files whose servers were paused: another round once a pause is over, as long as rounds bring
            # files in (servers that stay down cost one pause at the end of the run)
            rounds = dcfg.get("deferred_rounds", 8)
            for round_ in range(1, rounds + 1):
                if not deferred:
                    break
                with lock:
                    batch, deferred[:] = list(deferred), []
                pause = health.next_probe()
                log.info(f"round {round_}/{rounds}: {len(batch)} files deferred (servers paused), next probe in {pause / 60:.0f} min")
                time.sleep(pause + 1)
                before = sum(r["status"] == "done" for r in results)
                jobs.clear()
                handled[0] = 0
                for task, base, _ in batch:
                    submit(pool, task, base)
                wait_all()
                if sum(r["status"] == "done" for r in results) == before:
                    log.info(f"round {round_}: no file came in, servers still unreachable")
                    break
            for _, base, record in deferred:
                record_result({**base, **record, "status": "failed", "error": f"server(s) paused until the end of the run: {record['error']}"})
        dead = health.dead()
    if dead:
        log.warning(f"servers paused after repeated connection failures: {', '.join(dead)}")

    df = pd.DataFrame(results)
    if not df.empty:
        log.info(f"\n{df.groupby(['variable', 'experiment', 'status']).size().unstack(fill_value=0)}")
    return df
