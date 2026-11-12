import numpy as np
import xarray as xr

from climate_scenarios_2050.evaluate import wind as ev
from climate_scenarios_2050.methods.wind import member_weights, sfcwind_change, weighted_mean_std


def test_member_weights_model_counts_once():
    w = member_weights(["m0", "m0", "m0", "m1"], "model")
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    np.testing.assert_allclose(member_weights(["m0", "m0", "m1"], "member"), [1 / 3] * 3)


def test_weighted_mean_std_ignores_missing_members():
    values = np.array([1.0, 3.0, np.nan])
    mean, std = weighted_mean_std(values, np.array([0.25, 0.25, 0.5]))
    assert mean == 2 and std == 1


def test_sfcwind_change_own_reference():
    years = np.arange(1900, 2000)
    # member 0 constant, member 1 doubled in the window
    values = np.ones((2, 100, 1, 1))
    values[1, (years >= 1951) & (years <= 1955)] = 2
    out = sfcwind_change(values, years, 1950, (1, 5), 20)
    assert out[0, 0, 0] == 0 and out[1, 0, 0] == 100
    # lag: member years shifted by 3
    assert sfcwind_change(values, years, 1947, (1, 5), 20, np.array([0, 3]))[1, 0, 0] == 100
    # reference ending before init: years 1931-1950 for init 1951 (window 1952-1956 not doubled)
    assert sfcwind_change(values, years, 1950, (1, 5), 20, reference_end=1949)[1, 0, 0] == 100
    # window not covered
    assert np.isnan(sfcwind_change(values, years, 1995, (1, 10), 20)).all()


def _case():
    rng = np.random.default_rng(0)
    inits = np.arange(1960, 2010)
    target = xr.DataArray(rng.normal(size=(50, 1, 4)), dims=("init_year", "window", "latitude"), coords={"init_year": inits, "window": ["1-10"], "latitude": np.arange(4.0)})
    return target


def test_persistence_spread_and_reference_crpss():
    target = _case()
    spread = ev.persistence_spread(target, [(1, 10)])
    assert spread.dims == target.dims and float(spread.mean()) > 0.5
    scores = ev.gaussian_scores(xr.zeros_like(target), spread, target, spread, (0.5, 0.9))
    np.testing.assert_allclose(scores["crpss"].values, 0, atol=1e-6)


def test_gaussian_scores_perfect_forecast_beats_reference():
    target = _case()
    spread = ev.persistence_spread(target, [(1, 10)])
    scores = ev.gaussian_scores(target, 0.1 * xr.ones_like(target), target, spread, (0.5,))
    assert (scores["crpss"] > 0.8).all() and (scores["bss"] > 0.8).all() and (scores["corr"] > 0.99).all()
