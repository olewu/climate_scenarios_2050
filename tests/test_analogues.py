import numpy as np
import xarray as xr

from climate_scenarios_2050.analogues import matching
from climate_scenarios_2050.baselines import nao as baselines
from climate_scenarios_2050.evaluate import metrics

CFG = {
    "selection": "subsample", "measure": "CORR", "two_stage_fraction": 0.5,
    "filter": {"type": "running_mean", "length": 11, "min_periods_frac": 0.25},
    "window": 40, "M": 3, "max_lag": 5, "model_filter_at_init": "cut", "min_window_coverage": 1.0,
}


def _data(n_members=20, n_years=200, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1850, 1850 + n_years)
    model = rng.normal(size=(n_members, n_years))
    return years, model


def test_running_mean_matches_xarray():
    years, model = _data()
    da = xr.DataArray(model[0], dims="year")
    for length in [8, 41]:
        ours = matching.running_mean(model[0], length, length // 4)[0]
        theirs = da.rolling(year=length, center=True, min_periods=length // 4).mean().values
        np.testing.assert_allclose(ours, theirs)


def test_constant_member_is_not_selected():
    # a series without variance has NaN correlation: it must rank last, not first
    years, model = _data()
    model[0] = 1.0
    sel = matching.select_members(model[1], years, model, years, 1950, CFG)
    assert 0 not in sel["member"]
    assert sel["n_valid"] == 19


def test_best_member_found_and_future_not_used():
    years, model = _data()
    init = 1950
    obs = model[5].copy()
    sel = matching.select_members(obs, years, model, years, init, CFG)
    assert sel["member"][0] == 5
    assert np.isclose(sel["score"][0], 1.0)
    # changing observations and models after init does not change the selection
    obs2, model2 = obs.copy(), model.copy()
    after = years > init
    obs2[after] += 10
    model2[:, after] += np.random.default_rng(1).normal(size=(20, after.sum())) * 5
    sel2 = matching.select_members(obs2, years, model2, years, init, CFG)
    np.testing.assert_array_equal(sel["member"], sel2["member"])


def test_full_filter_option_uses_future_of_the_model():
    years, model = _data()
    cfg = {**CFG, "model_filter_at_init": "full"}
    a = matching.select_members(model[5], years, model, years, 1950, cfg)
    model2 = model.copy()
    model2[:, years > 1950] += np.random.default_rng(1).normal(size=(20, (years > 1950).sum())) * 5
    b = matching.select_members(model[5], years, model2, years, 1950, cfg)
    assert not np.array_equal(a["score"], b["score"])


def test_rmse_centres_every_member_by_its_own_mean():
    # an offset must not matter: every series is an anomaly over the window
    years, model = _data()
    model[7] = model[5] + 3.0
    cfg = {**CFG, "measure": "RMSE"}
    sel = matching.select_members(model[5], years, model, years, 1950, cfg)
    assert set(sel["member"][:2]) == {5, 7}
    assert sel["score"][0] < 1e-6


def test_window_has_requested_length():
    years = np.arange(1850, 2000)
    assert matching.window_mask(years, 1950, 70).sum() == 70
    assert years[matching.window_mask(years, 1950, 70)][-1] == 1950


def test_member_without_window_coverage_is_excluded():
    years, model = _data()
    obs = model[3].copy()
    model[3, years < 1930] = np.nan  # starts within the window
    sel = matching.select_members(obs, years, model, years, 1950, CFG)
    assert 3 not in sel["member"]
    sel2 = matching.select_members(obs, years, model, years, 1950, {**CFG, "min_window_coverage": 0.3})
    assert sel2["member"][0] == 3


def test_shift_finds_the_lag():
    years, model = _data()
    # obs at year t = member 4 at year t + 3
    obs = matching.shifted(model[4:5], 3)[0]
    obs[np.isnan(obs)] = 0
    cfg = {**CFG, "selection": "shift_subsample", "filter": {"type": "running_mean", "length": 5, "min_periods_frac": 1.0}, "model_filter_at_init": "full"}
    sel = matching.select_members(obs, years, model, years, 1950, cfg)
    assert sel["member"][0] == 4 and sel["lag"][0] == 3
    # the hindcast follows the shifted member
    leads = np.array([1, 2])
    out = matching.align_leads(model[[4]], years, 1950, leads, [3])
    np.testing.assert_allclose(out[0], model[4, [104, 105]])


def test_two_stage_returns_M():
    years, model = _data(n_members=30)
    sel = matching.select_members(model[5], years, model, years, 1950, {**CFG, "measure": "two_stage"})
    assert len(sel["member"]) == 3 and 5 in sel["member"]


def test_ar1_hindcast_uses_only_past():
    years = np.arange(1850, 2000)
    series = xr.DataArray(np.random.default_rng(0).normal(size=150), dims="year", coords={"year": years})
    leads = np.arange(-2, 4)
    bcfg = {"members": 5, "fit_window": 70, "fit_on": "annual"}
    a = baselines.ar1_hindcast(lambda i: series, [1950], leads, bcfg, 1)
    future = series.where(series.year <= 1950, 99.0)
    b = baselines.ar1_hindcast(lambda i: future, [1950], leads, bcfg, 1)
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.sel(lead=0, init_year=1950).values, series.sel(year=1950).values)


def test_crps_ensemble_and_gaussian_agree():
    rng = np.random.default_rng(0)
    obs = xr.DataArray(rng.normal(size=200), dims="init_year")
    ens = xr.DataArray(rng.normal(size=(200, 5000)), dims=("init_year", "member"))
    a = float(metrics.crps_ensemble(ens, obs).mean())
    b = float(metrics.crps_gaussian(xr.zeros_like(obs), xr.ones_like(obs), obs).mean())
    assert abs(a - b) < 0.01


def test_brier_skill_of_a_no_skill_forecast_is_not_inflated():
    # slowly varying observations and an ensemble unrelated to them
    rng = np.random.default_rng(0)
    inits = np.arange(1900, 2000)
    slow = np.convolve(rng.normal(size=120), np.ones(20) / 20, mode="valid")[:100]
    obs = xr.DataArray(slow, dims="init_year", coords={"init_year": inits})
    ens = xr.DataArray(rng.normal(scale=float(obs.std()), size=(100, 50)) + float(obs.mean()), dims=("init_year", "member"), coords={"init_year": inits})
    scores = metrics.probabilistic(ens, obs, 20, (0.5, 0.9))
    assert float(scores["bss"].sel(quantile=0.5)) < 0.1


def test_persistence_hindcast():
    years = np.arange(1900, 2000)
    series = xr.DataArray(years.astype(float), dims="year", coords={"year": years})
    out = baselines.persistence_hindcast(lambda i: series, [1950], np.array([-1, 0, 1, 5]), 20)
    # mean of 1931-1950 = 1940.5 at all positive leads, observations before
    assert float(out.sel(init_year=1950, lead=5, member=0)) == 1940.5
    assert float(out.sel(init_year=1950, lead=-1, member=0)) == 1949.0
