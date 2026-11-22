import hashlib
import http.server
import threading
from functools import partial

import pandas as pd
import pytest

from climate_scenarios_2050.cmip_retrieval import download as dl
from climate_scenarios_2050.cmip_retrieval import inventory as inv


def frame(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def test_plan_selects_usable_members_and_grids():
    cols = ["variable", "experiment", "model", "member", "grid", "version", "size", "n_files", "end", "dataset_id", "data_nodes"]
    remote = frame([
        ["psl", "historical", "A", "r1i1p1f1", "gn", 1, 10, 1, 2014, "a", "n"],
        ["psl", "ssp245", "A", "r1i1p1f1", "gn", 1, 5, 1, 2100, "b", "n"],
        ["psl", "ssp245", "A", "r1i1p1f1", "gr", 1, 4, 1, 2100, "c", "n"],
        ["psl", "ssp245", "A", "r2i1p1f1", "gn", 1, 5, 1, 2100, "d", "n"],    # no historical anywhere: not usable
        ["psl", "ssp370", "B", "r1i1p1f1", "gr", 1, 7, 1, float("nan"), "e", "n"],   # historical only locally (gr); end unknown
        ["psl", "ssp370", "B", "r1i1p1f1", "gn", 1, 6, 1, 2100, "f", "n"],
        ["psl", "historical", "C", "r1i1p1f1", "gn", 1, 10, 1, 2014, "g", "n"],
        ["psl", "ssp245", "C", "r1i1p1f1", "gn", 1, 1, 1, 2020, "h", "n"],     # short run only: member not usable
        ["psl", "ssp245", "B", "r1i1p1f1", "gr", 1, 1, 1, 2020, "i", "n"],     # short run of a usable member
    ], cols)
    local = frame([["psl", "historical", "B", "r1i1p1f1", "gr", 1, 99, "185001", 2014]],
                  ["variable", "experiment", "model", "member", "grid", "n_files", "size", "start", "end"])
    p = inv.plan(remote, local, ["ssp245", "ssp370"]).set_index(["model", "member", "experiment"])
    assert ("A", "r2i1p1f1", "ssp245") not in p.index
    assert p.loc[("A", "r1i1p1f1", "ssp245"), "grid"] == "gn"
    assert p.loc[("A", "r1i1p1f1", "ssp370"), "status"] == "missing"
    assert p.loc[("B", "r1i1p1f1", "historical"), "status"] == "local"
    assert p.loc[("B", "r1i1p1f1", "ssp370"), "grid"] == "gr"     # the member's local grid wins over gn
    assert p.loc[("A", "r1i1p1f1", "historical"), "status"] == "download"
    assert ("C", "r1i1p1f1", "historical") not in p.index
    assert p.loc[("B", "r1i1p1f1", "ssp245"), "status"] == "short"


def test_in_period():
    assert dl.in_period("psl_Amon_M_ssp245_r1i1p1f1_gn_201501-210012.nc", (1850, 2100))
    assert not dl.in_period("psl_Amon_M_ssp245_r1i1p1f1_gn_210101-230012.nc", (1850, 2100))


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_fetch_checks_checksum_and_never_overwrites(server, tmp_path):
    root, url = server
    content = b"netcdf" * 1000
    (root / "f.nc").write_bytes(content)
    partial_dir = tmp_path / ".partial"
    partial_dir.mkdir()
    good = {"name": "f.nc", "checksum": hashlib.sha256(content).hexdigest(), "checksum_type": "SHA256",
            "urls": [f"{url}/missing.nc", f"{url}/f.nc"]}   # first copy fails, second works
    dest = tmp_path / "archive" / "f.nc"
    assert dl.fetch(good, dest, partial_dir)["status"] == "done"
    assert dest.read_bytes() == content

    dest.write_bytes(b"existing")
    assert dl.fetch(good, dest, partial_dir)["status"] == "exists"
    assert dest.read_bytes() == b"existing"

    bad = {**good, "checksum": "0" * 64}
    dest2 = tmp_path / "archive" / "g.nc"
    record = dl.fetch(bad, dest2, partial_dir)
    assert record["status"] == "failed" and "checksum" in record["error"]
    assert not dest2.exists() and not any(partial_dir.iterdir())


def test_subset_names_and_months():
    name = "ua_Amon_M_historical_r1i1p1f1_gn_185001-201412.nc"
    tagged = inv.subset_file(name, "plev850_lat10-80")
    assert tagged == "ua_Amon_M_historical_r1i1p1f1_gn_185001-201412_plev850_lat10-80.nc"
    assert inv.FILE_PATTERN.match(tagged.replace("_plev850_lat10-80.nc", ".nc"))
    assert dl._months(name) == 165 * 12
    assert dl._months("ua_Amon_M_ssp245_r1i1p1f1_gn_201507-201606.nc") == 12


def test_unreachable_server_is_not_retried_and_then_skipped(server, tmp_path, monkeypatch):
    import socket
    import time

    root, url = server
    content = b"netcdf" * 100
    (root / "f.nc").write_bytes(content)
    with socket.socket() as sock:          # a port nobody listens on: connection refused
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    calls = []
    real_get = dl.requests.get
    monkeypatch.setattr(dl.requests, "get", lambda u, **kw: (calls.append(u), real_get(u, **kw))[1])
    partial_dir = tmp_path / ".partial"
    partial_dir.mkdir()
    health = dl.HostHealth(limit=3)
    file = {"name": "f.nc", "checksum": hashlib.sha256(content).hexdigest(), "checksum_type": "SHA256", "urls": [f"{dead}/f.nc", f"{url}/f.nc"]}
    start = time.time()
    for i in range(5):
        assert dl.fetch(file, tmp_path / f"a{i}" / "f.nc", partial_dir, health=health)["status"] == "done"   # second copy
    assert time.time() - start < 10
    dead_calls = [c for c in calls if c.startswith(dead)]
    assert len(dead_calls) == 3                # one attempt per file (no retries), then skipped
    assert health.dead() == [f"http:127.0.0.1:{dead.rsplit(':', 1)[1]} (1 pauses, still unreachable)"]
    only_dead = {**file, "urls": [f"{dead}/f.nc"]}
    record = dl.fetch(only_dead, tmp_path / "b" / "f.nc", partial_dir, health=health)
    assert record["status"] == "deferred" and "paused" in record["error"]   # retried in a later round
    # a working server keeps its count at zero
    assert health.store.get(health.key(url, "http"), 0) == 0


def test_paused_server_is_probed_again(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(dl.time, "time", lambda: clock[0])
    health = dl.HostHealth(limit=2, pause=600)
    url = "http://server.example/f.nc"
    health.failed(url, "http")
    assert health.alive(url, "http")
    health.failed(url, "http")
    assert not health.alive(url, "http")            # paused
    clock[0] += 601
    assert health.alive(url, "http")                # probe allowed
    health.failed(url, "http")
    assert not health.alive(url, "http")            # failed probe: paused again
    clock[0] += 601
    health.ok(url, "http")                          # server back
    assert health.alive(url, "http") and health.store[health.key(url, "http")][0] == 0
    assert health.dead() == ["http:server.example (2 pauses)"]


def test_requests_to_one_server_are_spaced():
    import time

    health = dl.HostHealth(interval=0.2)
    start = time.time()
    for _ in range(3):
        health.wait_turn("http://a.example/f.nc")
    health.wait_turn("http://b.example/f.nc")            # another server: no wait
    assert 0.4 <= time.time() - start < 0.6


def test_run_defers_files_of_paused_servers(server, tmp_path, monkeypatch):
    """files of a dead server are deferred, retried once after the pause and then logged as failed"""
    import json
    import socket

    from climate_scenarios_2050.core.config import load_config

    root, url = server
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    files = []
    for i in range(8):
        content = f"file {i}".encode()
        (root / f"f{i}.nc").write_bytes(content)
        files.append({"name": f"psl_Amon_M_historical_r1i1p1f1_gn_{1850 + i}01-{1850 + i}12.nc", "checksum": hashlib.sha256(content).hexdigest(),
                      "checksum_type": "SHA256", "urls": [f"{dead if i >= 4 else url}/f{i}.nc"], "opendap": []})
    monkeypatch.setattr(dl, "dataset_files", lambda *a, **k: files)
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    pd.DataFrame([{"variable": "psl", "experiment": "historical", "model": "M", "member": "r1i1p1f1", "status": "download", "grid": "gn",
                   "size": 1, "end": 2014, "n_files": 8, "version": 1, "dataset_id": "d", "data_nodes": "n", "note": ""}]).to_csv(inventory / "plan.csv", index=False)
    cfg = load_config()
    cfg["download"].update({"workers": 2, "skip_host_after": 2, "skip_host_minutes": 0.02, "min_request_interval": 0.0})
    df = dl.run(cfg, target=tmp_path / "archive", inventory=inventory)
    assert (df["status"] == "done").sum() == 4 and (df["status"] == "failed").sum() == 4
    logged = [json.loads(line) for line in open(inventory / "download_log.jsonl")]
    assert len(logged) == 8 and not any(r["status"] == "deferred" for r in logged)
