import numpy as np
import xarray as xr

from climate_scenarios_2050.evaluate import metrics
from climate_scenarios_2050.evaluate import surrogates as sg
from climate_scenarios_2050.evaluate import target as ev


def ar1(n, points, phi, rng):
    x = np.zeros((n, points))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal(points)
    return x


def setup(rng, n_years=90, points=40, info=0.0, ref=20, windows=((1, 5), (1, 10))):
    """internal wind u, M0/M1 and a method with `info` x the future window mean (on internal target)"""
    years = np.arange(1940, 1940 + n_years)
    u = ar1(n_years, points, 0.6, rng) + rng.standard_normal((1, points))  # spatial pattern of means
    u -= u.mean(axis=0)
    inits = np.arange(years[0] + ref - 1, years[-1] - max(b for _, b in windows) + 1)
    win_op, ref_op = sg.averaging_operators(years, inits, windows, ref)
    W = np.einsum("iwy,yp->iwp", win_op, u)
    R = np.einsum("iwy,yp->iwp", ref_op, u)
    target = W - R
    noise = rng.standard_normal(W.shape) * W.std()
    pred_m = info * W + noise * (1 - info) * 0.3 - R
    pred_0 = -R
    return years, u, inits, windows, win_op, ref_op, target, pred_m, pred_0


def make_problem(target, pred_m, pred_0, inits, windows):
    names = ["M0", "M1", "M2b"]
    spread_m = np.full(target.shape, target.std())
    sep = np.stack([metrics.separated(inits, b - a + 1) for a, b in windows]).astype(float)
    pred = np.stack([pred_0, np.zeros_like(target), pred_m])[None]
    spread = np.stack([spread_m, np.full(target.shape, np.nan), spread_m])[None]
    joint = np.ones(pred.shape, dtype=bool)
    return sg.Problem(target[None], pred, spread, joint, names, np.ones((len(inits), target.shape[-1])), sep)


def test_unperturbed_statistics_match_scores():
    rng = np.random.default_rng(0)
    years, u, inits, windows, win_op, ref_op, target, pred_m, pred_0 = setup(rng, info=0.5)
    problem = make_problem(target, pred_m, pred_0, inits, windows)
    stats = problem.statistics()
    da = lambda x: xr.DataArray(x, dims=("init_year", "window", "point"), coords={"init_year": inits, "window": ["1-5", "1-10"]})
    t = da(target)
    ref_spread = ev.persistence_spread(t, windows)
    s_m = ev.gaussian_scores(da(pred_m), da(np.full(target.shape, target.std())), t, ref_spread, (0.5,))
    s_0 = ev.gaussian_scores(da(pred_0), da(np.full(target.shape, target.std())), t, ref_spread, (0.5,))
    np.testing.assert_allclose(stats[0, 0, 2], (s_m["crpss"] - s_0["crpss"]).values, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(stats[1, 0, 2], (s_m["corr"] - s_0["corr"]).values, rtol=1e-4, atol=1e-5)
    assert np.isnan(stats[:, 0, 0]).all()   # M0 itself


def test_coherent_surrogates_keep_spectra_and_cross_spectra():
    rng = np.random.default_rng(1)
    u = ar1(64, 3, 0.7, rng)
    u -= u.mean(axis=0)
    spec = np.fft.rfft(u, axis=0)
    phases = sg.coherent_phases(64, 1, seed=3)[0]
    us = u + sg.perturbation(spec, phases, 64)
    spec_s = np.fft.rfft(us, axis=0)
    np.testing.assert_allclose(np.abs(spec_s), np.abs(spec), atol=1e-9)
    np.testing.assert_allclose(spec_s[:, 0] * np.conj(spec_s[:, 1]), spec[:, 0] * np.conj(spec[:, 1]), atol=1e-8)
    assert not np.allclose(us, u)


def run_test(rng, info, n=199):
    years, u, inits, windows, win_op, ref_op, target, pred_m, pred_0 = setup(rng, info=info)
    problem = make_problem(target, pred_m, pred_0, inits, windows)
    observed, exceed, n_null, field, obs_field = sg._chunk(problem, np.fft.rfft(u, axis=0), sg.coherent_phases(len(years), n, 5), len(years), win_op, ref_op, np.ones(u.shape[1]))
    p = (1 + exceed) / (n_null + 1)
    return p[:, 0, 2]


def test_informative_prediction_is_significant():
    p = run_test(np.random.default_rng(2), info=0.9)
    assert np.median(p) < 0.02


def test_uninformative_prediction_is_not():
    p = run_test(np.random.default_rng(3), info=0.0)
    assert np.mean(p < 0.05) < 0.25


def test_fdr():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    np.testing.assert_array_equal(sg.fdr_mask(p, 0.05), [True, True, False, False, False, False, False, False, False])
    assert not sg.fdr_mask(np.full(5, np.nan), 0.1).any()


def test_leave_out_operators_match_fit():
    """operators reproduce the internal means of the leave-out forced fit"""
    from climate_scenarios_2050.split import forced as fc

    rng = np.random.default_rng(4)
    years = np.arange(1940, 2025)
    windows = [(1, 5), (6, 10)]
    inits = np.arange(1960, 2015)
    basis = xr.DataArray(np.linspace(0, 1, len(years))[:, None] ** [1, 2], dims=("year", "term"), coords={"year": years, "term": ["a", "b"]})
    y = xr.DataArray(rng.standard_normal((len(years), 3)) + 2 * basis.values[:, :1], dims=("year", "point"), coords={"year": years})
    design = fc.design_matrix(basis, years)
    excl = fc.window_exclusions(years, inits, windows, lag=0, reference_length=20)
    coef = fc.fit_coefficients(y, design, excl, 15)
    win, ref = sg.averaging_operators(years, inits, windows, 20)
    win_lo, ref_lo = sg.leave_out_operators(win, ref, design.values, excl.values, 15)
    for ii, iw in [(0, 0), (10, 1), (40, 0)]:
        internal = y - fc.forced_part(design, coef.isel(init_year=ii, window=iw))
        np.testing.assert_allclose(win_lo[ii, iw] @ y.values, win[ii, iw] @ internal.values, atol=1e-10)
        np.testing.assert_allclose(ref_lo[ii, iw] @ y.values, ref[ii, iw] @ internal.values, atol=1e-10)
