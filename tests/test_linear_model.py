import numpy as np

from climate_scenarios_2050.linear_model.fit import fit_case, regress, sliding_means


def test_sliding_means():
    x = np.arange(6.0)
    np.testing.assert_allclose(sliding_means(x, 3), [1, 2, 3, 4])
    x[2] = np.nan
    assert np.isnan(sliding_means(x, 3)[:3]).all() and sliding_means(x, 3)[3] == 4


def test_regress_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = np.stack([2 * x + 1 + 0.1 * rng.normal(size=500), -x + 0.5 * rng.normal(size=500)], axis=1)
    slope, intercept, sigma = regress(x, y)
    np.testing.assert_allclose(slope[0], [2, -1], atol=0.05)
    np.testing.assert_allclose(sigma, [0.1, 0.5], atol=0.05)
    assert abs(intercept[0] - 1) < 0.05


def _series(n=85, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 0.5 * x[:, None] + 0.2 * rng.normal(size=(n, 3))


def test_target_window_does_not_influence_fit():
    x, y = _series()
    years = np.arange(1940, 2025)
    target = (1990, 1999, years)
    a = fit_case(x, y, 10, target)
    x2, y2 = x.copy(), y.copy()
    inside = (years >= 1990) & (years <= 1999)
    x2[inside] += 50
    y2[inside] -= 50
    b = fit_case(x2, y2, 10, target)
    for u, v in zip(a[:3], b[:3]):
        np.testing.assert_allclose(u, v)
    assert a[3] == b[3] == 76 - 19  # sliding windows minus those overlapping the target


def test_too_few_samples():
    x, y = _series(n=25)
    slope, _, _, n = fit_case(x, y, 20)
    assert n == 6 and np.isnan(slope).all()


def test_multiple_regression_and_ridge():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.standard_normal((n, 2))
    x[:, 1] += 0.5 * x[:, 0]                       # correlated predictors
    y = (1.5 * x[:, 0] - 0.7 * x[:, 1])[:, None] + 0.2 * rng.standard_normal((n, 3))
    slope, intercept, sigma = regress(x, y)
    np.testing.assert_allclose(slope, [[1.5] * 3, [-0.7] * 3], atol=0.05)
    np.testing.assert_allclose(sigma, 0.2, atol=0.03)
    # one predictor: the same as a single-column regression
    s1, i1, sg1 = regress(x[:, 0], y)
    s2, i2, sg2 = regress(x[:, :1], y)
    np.testing.assert_allclose(s1, s2) and np.testing.assert_allclose(sg1, sg2)
    # ridge shrinks the slopes towards zero
    shrunk, _, _ = regress(x, y, ridge=1.0)
    assert np.all(np.linalg.norm(shrunk, axis=0) < np.linalg.norm(slope, axis=0))
    # fit_case with two predictors on sliding means
    years = np.arange(1940, 2025)
    xi = rng.standard_normal((len(years), 2))
    yi = (xi @ np.array([1.0, 0.5]))[:, None] + 0.1 * rng.standard_normal((len(years), 1))
    s, _, _, n_samples = fit_case(xi, yi, 5, min_samples=20)
    assert s.shape == (2, 1) and n_samples == len(years) - 4
    np.testing.assert_allclose(s[:, 0], [1.0, 0.5], atol=0.1)
