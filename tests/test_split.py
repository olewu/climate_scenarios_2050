import numpy as np
import xarray as xr

from climate_scenarios_2050.core.filters import lowpass
from climate_scenarios_2050.split import forced as fc


def _basis(years, seed=0):
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.normal(size=len(years)))
    return xr.DataArray(values[None, :], dims=("term", "year"), coords={"term": ["mmem"], "year": years})


def test_coefficients_recovered_and_ddof_consistent():
    years = np.arange(1900, 2000)
    basis = _basis(years)
    rng = np.random.default_rng(1)
    y = xr.DataArray(2.0 + 0.5 * basis.values[0] + rng.normal(size=100), dims="year", coords={"year": years})
    design = fc.design_matrix(basis, years)
    coef = fc.fit_coefficients(y, design, fc.full_period_exclusions(years))
    # OLS slope = cov/var with consistent normalization
    b = basis.isel(term=0)
    assert np.isclose(float(coef.sel(term="mmem")), float(xr.cov(y, b) / b.var(ddof=1)))
    # internal has zero mean and is uncorrelated with the basis over the fit years
    internal = y - fc.forced_part(design, coef)
    assert abs(float(internal.mean())) < 1e-10
    assert abs(float(xr.corr(internal, b))) < 1e-10


def test_excluded_years_do_not_influence_fit():
    years = np.arange(1900, 2000)
    basis = _basis(years)
    rng = np.random.default_rng(2)
    y = xr.DataArray(rng.normal(size=100), dims="year", coords={"year": years})
    design = fc.design_matrix(basis, years)
    windows = [(1, 5), (21, 30)]
    excl = fc.window_exclusions(years, [1950, 1960], windows, lag=0, reference_length=20)
    coef = fc.fit_coefficients(y, design, excl)

    # target window of init 1950, window 21-30 is 1971-1980, reference 1931-1950
    y2 = y.copy()
    y2.loc[dict(year=slice(1971, 1980))] += 100
    y2.loc[dict(year=slice(1931, 1950))] -= 100
    coef2 = fc.fit_coefficients(y2, design, excl)
    a = coef.sel(init_year=1950, window="21-30")
    b = coef2.sel(init_year=1950, window="21-30")
    np.testing.assert_allclose(a.values, b.values)
    # ... but it does influence a case where those years are used
    assert not np.allclose(coef.sel(init_year=1960, window="1-5").values, coef2.sel(init_year=1960, window="1-5").values)


def test_window_exclusions_lag_and_incomplete_cases():
    years = np.arange(1940, 2025)
    excl = fc.window_exclusions(years, [1959, 2000, 2010], [(1, 5), (21, 30)], lag=1)
    e = excl.sel(init_year=2000, window="1-5")
    # lag 1: reference seasons 1980-1999, window seasons 2000-2004
    assert e.sel(year=1980) and e.sel(year=1999) and e.sel(year=2000) and e.sel(year=2004)
    assert not e.sel(year=1979) and not e.sel(year=2005)
    # window 21-30 of init 2010 ends in 2039: not covered, everything excluded
    assert excl.sel(init_year=2010, window="21-30").all()
    # reference of init 1959 with lag 1 starts in 1939: not covered
    assert excl.sel(init_year=1959, window="1-5").all()


def test_up_to_init():
    years = np.arange(1900, 2000)
    e = fc.up_to_init_exclusions(years, [1950])
    assert not e.sel(init_year=1950, year=1950) and e.sel(init_year=1950, year=1951)


def _ensemble():
    """model a: 4 members, forced 1 x signal; model b: 1 member, forced 2 x signal"""
    years = np.arange(1850, 2100)
    signal = np.sin((years - 1850) / 60)
    rng = np.random.default_rng(3)
    data = np.vstack([signal + 0.1 * rng.normal(size=250) for _ in range(4)] + [2 * signal + 0.1 * rng.normal(size=250)])
    return xr.DataArray(
        data, dims=("member", "year"),
        coords={"member": ["m0r1", "m0r2", "m0r3", "m0r4", "m1r1"], "year": years, "model_id": ("member", ["m0"] * 4 + ["m1"])},
    )


def test_weighting():
    index = _ensemble()
    model = fc.multi_model_mean(index, "model")
    member = fc.multi_model_mean(index, "member")
    # model-weighted: (1 + 2)/2 = 1.5 x signal; member-weighted: (4 + 2)/5 = 1.2 x signal
    signal = np.sin((index.year.values - 1850) / 60)
    assert abs(float(np.polyfit(signal, model.values, 1)[0]) - 1.5) < 0.05
    assert abs(float(np.polyfit(signal, member.values, 1)[0]) - 1.2) < 0.05


def test_split_ensemble_single_member_not_zeroed():
    # more models, so that a single member is a small part of the MMEM
    years = np.arange(1850, 2100)
    rng = np.random.default_rng(4)
    signal = np.sin((years - 1850) / 60)
    extra = xr.DataArray(
        np.vstack([signal + 0.1 * rng.normal(size=250) for _ in range(20)]), dims=("member", "year"),
        coords={"member": [f"m{i}r1" for i in range(2, 22)], "year": years, "model_id": ("member", [f"m{i}" for i in range(2, 22)])},
    )
    index = xr.concat([_ensemble(), extra], dim="member")
    basis = fc.make_basis(fc.multi_model_mean(index, "model"), {"method": "mmem"})
    res = fc.split_ensemble(index, basis)
    # internal variability of the single-member model is kept (noise of std 0.1), forced removed
    single = res["internal"].sel(member="m1r1")
    assert 0.05 < float(single.std()) < 0.2
    assert abs(float(xr.corr(single, basis.isel(term=0, drop=True)))) < 0.1
    # scaling factor of model m1 is larger than of m0
    assert float(res["coef"].sel(term="mmem", model_id="m1")) > float(res["coef"].sel(term="mmem", model_id="m0"))


def test_runs_ending_early_keep_coefficients():
    index = _ensemble()
    index = index.where(~((index.member == "m1r1") & (index.year > 2090)))
    basis = fc.make_basis(fc.multi_model_mean(index, "model", min_model_fraction=0.5), {"method": "mmem"})
    res = fc.split_ensemble(index, basis)
    assert np.isfinite(res["coef"].sel(model_id="m1")).all()
    assert res["internal"].sel(member="m1r1", year=2095).isnull()


def test_basis_methods():
    index = _ensemble()
    mmem = fc.multi_model_mean(index, "model")
    poly = fc.make_basis(mmem, {"method": "polynomial", "polynomial_degree": 3})
    assert list(poly.term.values) == ["t1", "t2", "t3"]
    low = fc.make_basis(mmem, {"method": "lowpass_mmem", "lowpass": {"cutoff_period": 5, "order": 4}})
    assert float(low.std()) <= float(mmem.std())


def test_lowpass_pad_and_nans():
    x = np.sin(np.arange(200) / 20) + 0.1 * np.random.default_rng(0).normal(size=200)
    x[:5] = np.nan
    for pad in [None, 2, 3, 10]:  # even pads were rejected by the old filter
        y = lowpass(x, 10, pad=pad)
        assert np.isnan(y[:5]).all() and np.isfinite(y[5:]).all()
