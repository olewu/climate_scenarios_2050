"""several indices: config views and hashes, joint matching, member statistics"""

import copy

import numpy as np
import pytest
import xarray as xr

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.core.config import hindcast_indices, index_cfg, index_names, lm_predictors, load_config, match_weights, validate
from climate_scenarios_2050.methods.target import weighted_mean_cov, weighted_mean_std

MULTI = ["configs/experiments/nao_ea.yaml"]


def small(cfg, tmp_path):
    cfg["paths"]["outputs"] = {"root": str(tmp_path / "out"), "cmip6_processed": str(tmp_path / "processed"), "figures": str(tmp_path / "fig")}
    cfg["paths"]["inputs"]["cmip6"] = str(tmp_path / "archive")
    return cfg


def test_index_views_and_hashes(tmp_path):
    from climate_scenarios_2050.analogues.generate import hindcast_output
    from climate_scenarios_2050.linear_model.fit import linear_model_output
    from climate_scenarios_2050.preprocess.indices.compute import index_output
    from climate_scenarios_2050.split.run import split_output

    single = small(load_config(), tmp_path)
    multi = small(load_config(MULTI), tmp_path)
    assert index_names(multi) == ["nao", "ea"] and match_weights(multi) == {"nao": 0.5, "ea": 0.5}
    assert lm_predictors(multi) == ["nao", "ea"] and hindcast_indices(multi) == ["nao", "ea"]
    ea = index_cfg(multi, "ea")["predictor"]
    assert ea["mode"] == 2 and ea["sign_reference_point"] == {"lat": 55, "lon": -20, "sign": "negative"} and ea["indices"] is None
    # the NAO of the multi-index configuration is the NAO of the single-index one: shared patterns and splits
    assert index_output(multi, "nao").hash == index_output(single).hash
    assert split_output(multi, "ssp245", "nao").hash == split_output(single, "ssp245").hash
    assert index_output(multi, "ea").hash != index_output(multi, "nao").hash
    # joint matching and two predictors give new hindcast and linear-model outputs
    assert hindcast_output(multi, "ssp245").hash != hindcast_output(single, "ssp245").hash
    assert linear_model_output(multi, "ssp245", "wsp100").hash != linear_model_output(single, "ssp245", "wsp100").hash
    # matching on the NAO only with the EA unused: as the single-index configuration
    nao_only = copy.deepcopy(multi)
    nao_only["analogues"]["match_weights"] = None
    nao_only["linear_model"]["predictors"] = None
    assert hindcast_output(nao_only, "ssp245").hash == hindcast_output(single, "ssp245").hash


def test_validation_of_index_names():
    cfg = load_config(MULTI)
    bad = copy.deepcopy(cfg)
    bad["analogues"]["match_weights"] = {"nao": 1, "pdo": 1}
    with pytest.raises(ValueError, match="match_weights"):
        validate(bad)
    bad = copy.deepcopy(cfg)
    bad["linear_model"]["ridge"] = -1
    with pytest.raises(ValueError, match="ridge"):
        validate(bad)


def series(rng, n_members=60, n_years=200):
    """two independent AR(1)-like indices per member"""
    x = np.zeros((2, n_members, n_years))
    for t in range(1, n_years):
        x[:, :, t] = 0.8 * x[:, :, t - 1] + rng.standard_normal((2, n_members))
    return x


ACFG = {"window": 40, "filter": {"type": "running_mean", "length": 5, "min_periods_frac": 1.0}, "selection": "subsample", "measure": "CORR",
        "M": 10, "min_window_coverage": 1.0, "model_filter_at_init": "full", "max_lag": 3, "two_stage_fraction": 0.5}


def test_joint_matching():
    rng = np.random.default_rng(0)
    model = series(rng)
    years = np.arange(1850, 2050)
    init = 1990
    # observations: member 7 in index 0, member 3 in index 1
    obs = np.stack([model[0, 7], model[1, 3]])
    # weight 1 on index 0: identical to the single-index selection
    one = matching.select_members(obs[0], years, model[0], years, init, ACFG)
    weighted = matching.select_members(obs, years, model, years, init, ACFG, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(one["member"], weighted["member"])
    assert one["member"][0] == 7
    # equal weights: members good in both rank first; the score is the mean of the two correlations
    joint = matching.select_members(obs, years, model, years, init, ACFG, np.array([0.5, 0.5]))
    second = matching.select_members(obs[1], years, model[1], years, init, ACFG)
    assert {7, 3} & set(joint["member"][:3])
    s0 = dict(zip(one["member"], one["score"]))
    s1 = dict(zip(second["member"], second["score"]))
    for m, s in zip(joint["member"], joint["score"]):
        if m in s0 and m in s1:
            assert s == pytest.approx(0.5 * (s0[m] + s1[m]))
    # RMSE: root of the weighted mean of the squared errors
    rmse = {**ACFG, "measure": "RMSE"}
    j = matching.select_members(obs, years, model, years, init, rmse, np.array([0.5, 0.5]))
    a = matching.select_members(obs[0], years, model[0], years, init, {**rmse, "M": 60})
    b = matching.select_members(obs[1], years, model[1], years, init, {**rmse, "M": 60})
    ra, rb = dict(zip(a["member"], a["score"])), dict(zip(b["member"], b["score"]))
    assert j["score"][0] == pytest.approx(np.sqrt(0.5 * ra[j["member"][0]] ** 2 + 0.5 * rb[j["member"][0]] ** 2))
    # shift selection: one lag per member for both indices
    shift = matching.select_members(obs, years, model, years, init, {**ACFG, "selection": "shift_subsample"}, np.array([0.5, 0.5]))
    assert len(shift["lag"]) == ACFG["M"]


def test_weighted_mean_cov():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 2))
    x[3, 1] = np.nan
    w = rng.uniform(0.5, 1.5, 50)
    mean, cov = weighted_mean_cov(x, w)
    m0, s0 = weighted_mean_std(x[np.isfinite(x).all(axis=1), :1], w[np.isfinite(x).all(axis=1)])
    assert mean[0] == pytest.approx(float(m0[0])) and cov[0, 0] == pytest.approx(float(s0[0]) ** 2)
    assert cov[0, 1] == pytest.approx(cov[1, 0])


def test_spread_of_the_linear_model_methods():
    from climate_scenarios_2050.forecast.forecast import quadratic

    slope = xr.DataArray(np.array([[2.0], [1.0]]), dims=("predictor", "point"), coords={"predictor": ["nao", "ea"]})
    cov = xr.DataArray(np.array([[1.0, 0.5], [0.5, 4.0]]), dims=("predictor", "predictor2"))
    # s' C s = 4*1 + 2*2*1*0.5 + 1*4 = 10
    assert float(quadratic(slope, cov)) == pytest.approx(10.0)
