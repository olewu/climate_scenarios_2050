import json

import numpy as np
import xarray as xr

from climate_scenarios_2050.gallery import export


class FakeOutput:
    """stands in for the StepOutput of an evaluate_target folder"""

    def __init__(self, folder, digest):
        self.dir, self.hash = folder, digest
        self.record = {"extra": {"scenario": "ssp245", "target_variable": "wsp100", "spec": {"change": "relative"}}}

    def path(self, name):
        return self.dir / name

    def exists(self, name):
        return self.path(name).exists()


def test_target_export_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "coastlines", lambda cfg, lat, lon: "coast_test.json")
    rng = np.random.default_rng(0)
    dims = ("target", "fc_method", "window", "latitude", "longitude")
    coords = {"target": ["full", "internal"], "fc_method": ["M0", "M3a"], "window": ["1-5", "1-10"],
              "latitude": [50.0, 50.5, 51.0], "longitude": [0.0, 0.5]}
    corr = rng.uniform(-1, 1, (2, 2, 2, 3, 2))
    corr[0, 1, 0, 1, 1] = np.nan
    crpss = rng.uniform(-5, 1, (2, 2, 2, 3, 2))    # values below -3.2767 are clipped
    src = tmp_path / "out"
    src.mkdir()
    xr.Dataset({"corr": (dims, corr), "crpss": (dims, crpss)}, coords=coords).to_netcdf(src / "scores.nc")
    cfg = {"paths": {"outputs": {"root": str(tmp_path / "root"), "figures": str(tmp_path / "fig")}}}
    stem = export.export_target_evaluation(cfg, FakeOutput(src, "abcd1234"))
    assert stem == "evaluate_target/abcd1234_s_v2"
    data = tmp_path / "fig" / "gallery" / "data"
    idx = json.loads((data / f"{stem}.json").read_text())
    raw = (data / f"{stem}.bin").read_bytes()

    def read(key):
        o, n = idx["maps"][key]
        v = np.frombuffer(raw[o:o + n], "<i2").astype(float).reshape(idx["shape"])
        return np.where(v == idx["missing"], np.nan, v * idx["scale"])

    np.testing.assert_allclose(read("corr|full|M3a|1-5"), corr[0, 1, 0], atol=6e-5)
    np.testing.assert_allclose(read("crpss|internal|M0|1-10"), np.clip(crpss[1, 0, 1], -3.2767, 3.2767), atol=6e-5)
    # an existing export is kept, not rewritten
    before = (data / f"{stem}.bin").stat().st_mtime_ns
    assert export.export_target_evaluation(cfg, FakeOutput(src, "abcd1234")) == stem
    assert (data / f"{stem}.bin").stat().st_mtime_ns == before


def test_index_export_json(tmp_path):
    lead = np.arange(1, 4)
    products = ["analogue", "ar1_fair"]
    score = np.array([[0.5, np.nan, 0.2], [0.1, 0.0, -0.1]])
    src = tmp_path / "out"
    src.mkdir()
    xr.Dataset({"corr_score": (("product", "lead"), score), "corr_n": (("product", "lead"), np.ones((2, 3)))},
               coords={"product": products, "lead": lead}).to_netcdf(src / "continuous.nc")
    cfg = {"paths": {"outputs": {"root": str(tmp_path / "root"), "figures": str(tmp_path / "fig")}}}
    stem = export.export_index_evaluation(cfg, FakeOutput(src, "ffff0000"), "ssp245", "nao")
    assert stem == "evaluate_index/ffff0000_c_v2"
    data = json.loads((tmp_path / "fig" / "gallery" / "data" / f"{stem}.json").read_text())
    assert data["continuous"]["products"] == products and data["continuous"]["lead"] == [1, 2, 3]
    assert data["continuous"]["corr_score"] == [[0.5, None, 0.2], [0.1, 0.0, -0.1]]
    assert "corr_n" not in data["continuous"]
