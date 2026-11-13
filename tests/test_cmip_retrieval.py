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
    cols = ["variable", "experiment", "model", "member", "grid", "version", "size", "n_files", "dataset_id", "data_nodes"]
    remote = frame([
        ["psl", "historical", "A", "r1i1p1f1", "gn", 1, 10, 1, "a", "n"],
        ["psl", "ssp245", "A", "r1i1p1f1", "gn", 1, 5, 1, "b", "n"],
        ["psl", "ssp245", "A", "r1i1p1f1", "gr", 1, 4, 1, "c", "n"],
        ["psl", "ssp245", "A", "r2i1p1f1", "gn", 1, 5, 1, "d", "n"],    # no historical anywhere: not usable
        ["psl", "ssp370", "B", "r1i1p1f1", "gr", 1, 7, 1, "e", "n"],    # historical only locally (gr)
        ["psl", "ssp370", "B", "r1i1p1f1", "gn", 1, 6, 1, "f", "n"],
    ], cols)
    local = frame([["psl", "historical", "B", "r1i1p1f1", "gr", 1, 99, "185001", "201412"]],
                  ["variable", "experiment", "model", "member", "grid", "n_files", "size", "start", "end"])
    p = inv.plan(remote, local, ["ssp245", "ssp370"]).set_index(["model", "member", "experiment"])
    assert ("A", "r2i1p1f1", "ssp245") not in p.index
    assert p.loc[("A", "r1i1p1f1", "ssp245"), "grid"] == "gn"
    assert p.loc[("A", "r1i1p1f1", "ssp370"), "status"] == "missing"
    assert p.loc[("B", "r1i1p1f1", "historical"), "status"] == "local"
    assert p.loc[("B", "r1i1p1f1", "ssp370"), "grid"] == "gr"     # the member's local grid wins over gn
    assert p.loc[("A", "r1i1p1f1", "historical"), "status"] == "download"


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
