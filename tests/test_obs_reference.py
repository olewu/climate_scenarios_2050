import numpy as np

from climate_scenarios_2050.evaluate import versus_m1
from climate_scenarios_2050.evaluate.target import OBS_METHODS, observation_reference_operators


def test_trend_operator_is_exact_for_a_linear_series():
    years = np.arange(1940, 2026)
    obs = 2.0 + 0.03 * (years - 1940)
    inits, windows = [1970, 1990], [(1, 5), (6, 10)]
    ops = observation_reference_operators(years, inits, windows, 20, "up_to_init", 20)
    m0t = ops[OBS_METHODS.index("M0t")] @ obs
    for ii, init in enumerate(inits):
        for iw, (a, b) in enumerate(windows):
            expected = obs[(years >= init + a) & (years <= init + b)].mean() - obs[(years > init - 20) & (years <= init)].mean()
            np.testing.assert_allclose(m0t[ii, iw], expected, atol=1e-10)
    # constant: return to the mean of all years up to init
    m0c = ops[OBS_METHODS.index("M0c")] @ obs
    np.testing.assert_allclose(m0c[0, 0], obs[years <= 1970].mean() - obs[(years > 1950) & (years <= 1970)].mean())


def test_operators_use_no_future_years_and_need_min_years():
    years = np.arange(1940, 2026)
    ops = observation_reference_operators(years, [1965, 1980], [(1, 10)], 20, "up_to_init", 30)
    assert np.isnan(ops[:, 0]).all()                  # 1940-1965: 26 years < 30
    assert (ops[:, 1, 0, years > 1980] == 0).all()    # only years up to init
    lo = observation_reference_operators(years, [1980], [(1, 10)], 20, "leave_out", 20)
    assert (lo[0, 0, 0, (years > 1960) & (years <= 1990)] == -1 / 20 * ((years > 1960) & (years <= 1980))[(years > 1960) & (years <= 1990)]).all()


def test_block_counts_and_ratio_test():
    rng = np.random.default_rng(0)
    counts = versus_m1.block_counts(40, 5, 200, rng)
    assert counts.shape == (200, 40) and (counts.sum(1) == 40).all()
    n, p = 40, 3
    obs = rng.normal(size=(n, p))
    perfect = (obs * 0.1) ** 2            # much smaller squared errors than M1 (obs**2)
    same = obs ** 2
    w = np.ones((n, p))
    v, pl, fm, fp = versus_m1.ratio_test(perfect, obs ** 2, w, counts, np.ones(p))
    assert (v > 0.9).all() and (pl < 0.05).all() and fp < 0.05
    v, pl, fm, fp = versus_m1.ratio_test(same, obs ** 2, w, counts, np.ones(p))
    np.testing.assert_allclose(v, 0, atol=1e-12)
    assert (pl > 0.3).all()


def test_forecast_obs_reference_matches_evaluation_operators():
    import xarray as xr

    from climate_scenarios_2050.forecast.forecast import obs_reference_forecast

    years = np.arange(1940, 2026)
    rng = np.random.default_rng(1)
    obs = 5 + 0.02 * (years - 1940) + rng.normal(0, 0.3, len(years))
    wind = xr.DataArray(obs[:, None, None], dims=("year", "latitude", "longitude"), coords={"year": years, "latitude": [60.0], "longitude": [5.0]})
    init, window = 2000, (1, 10)
    out = obs_reference_forecast(wind, window, 20, init, init, 20)
    ops = observation_reference_operators(years, [init], [window], 20, "up_to_init", 20)
    for k, name in enumerate(OBS_METHODS):
        np.testing.assert_allclose(float(out[name][0]), ops[k, 0, 0] @ obs, atol=1e-10)
        assert float(out[name][1]) > 0
    # an exactly linear series: M0t is perfect, also for the past windows of its spread
    lin = wind.copy(data=(5 + 0.02 * (years - 1940))[:, None, None])
    change, spread = obs_reference_forecast(lin, window, 20, init, init, 20)["M0t"]
    assert abs(float(spread)) < 1e-10
