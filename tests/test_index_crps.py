import numpy as np
import xarray as xr

from climate_scenarios_2050.evaluate.index import continuous_crps

CFG = {"evaluation": {"index_continuous": {"running_mean": 1}, "bootstrap": {"n": 200, "seed": 1}}, "baselines": {"ar1": {}}}


def product(values):
    """values (init, member, lead)"""
    n_init, n_mem, n_lead = values.shape
    return xr.DataArray(values, dims=("init_year", "member", "lead"),
                        coords={"init_year": 1950 + np.arange(n_init), "member": np.arange(n_mem), "lead": np.arange(1, n_lead + 1)})


def test_crps_per_lead_and_skill():
    rng = np.random.default_rng(0)
    n_init, n_lead = 60, 3
    obs = rng.normal(size=(n_init, n_lead))
    verification = xr.DataArray(obs, dims=("init_year", "lead"), coords={"init_year": 1950 + np.arange(n_init), "lead": np.arange(1, n_lead + 1)})
    climatology = rng.normal(size=(n_init, 50, n_lead))
    perfect = np.repeat(obs[:, None, :], 5, axis=1)          # every member equals the observation
    products = {"cmip6_all": product(climatology), "analogue": product(perfect), "same": product(climatology.copy()),
                "persistence": product(np.zeros((n_init, 1, n_lead)))}
    ds = continuous_crps(CFG, products, verification, np.arange(1, n_lead + 1))
    assert list(ds["product"].values) == ["cmip6_all", "analogue", "same"]   # single-member products left out
    assert np.allclose(ds["crps"].sel(product="analogue"), 0)
    assert np.allclose(ds["crpss_cmip6"].sel(product="analogue"), 1)
    assert np.allclose(ds["crpss_cmip6"].sel(product="same"), 0)
    assert ds["crpss_cmip6"].sel(product="cmip6_all").isnull().all()
    assert (ds["crpss_cmip6_p"].sel(product="analogue") < 0.05).all()
    low, high = ds["crpss_cmip6_ci_low"].sel(product="same"), ds["crpss_cmip6_ci_high"].sel(product="same")
    assert np.allclose(low, 0) and np.allclose(high, 0)
