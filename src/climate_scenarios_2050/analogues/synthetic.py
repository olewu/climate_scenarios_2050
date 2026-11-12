"""
Null test: the analogue selection applied to synthetic "observations" with no relation to the model
runs (AR(1) around the linear trend of the observed internal index, as in the old
`nao_ar1_lofreq_forecast.py`). Skill against these series shows what the method gets by chance.
"""

import numpy as np
import xarray as xr
from statsmodels.tsa.arima.model import ARIMA

from climate_scenarios_2050.analogues import matching


def fit_ar1_ensemble(series: xr.DataArray, null_cfg: dict, spinup: int = 200) -> xr.DataArray:
    years = series.year.values
    trend = np.polyval(np.polyfit(years - years[0], series.values, 1), years - years[0])
    fitted = ARIMA(series.values - trend, order=(1, 0, 0)).fit()
    rng = np.random.default_rng(null_cfg["seed"])
    sims = fitted.simulate(nsimulations=spinup + len(years), repetitions=null_cfg["n"], random_state=rng).squeeze()[spinup:]
    return xr.DataArray(trend + sims.T, dims=("syn", "year"), coords={"syn": np.arange(null_cfg["n"]), "year": years}, name="synthetic_obs")


def select_synthetic(synth: xr.DataArray, model, years, inits, leads, acfg) -> xr.Dataset:
    """ensemble mean and spread (float32) of the selection for each synthetic series and init"""
    syn_years = synth.year.values
    obs = synth.transpose("syn", "year").values
    mean = np.full((obs.shape[0], len(inits), len(leads)), np.nan, dtype="float32")
    std = np.full_like(mean, np.nan)
    measure = acfg["measure"]
    for ii, init in enumerate(inits):
        known = syn_years <= init
        obs_filtered = matching.apply_filter(obs[:, known], acfg["filter"])
        obs_years = syn_years[known]
        finite_years = obs_years[np.isfinite(obs_filtered[0])]
        mask = matching.window_mask(years, init, acfg["window"]) & np.isin(years, finite_years)
        obs_win = obs_filtered[:, np.isin(obs_years, years[mask])]
        # all synthetic series are compared with the same filtered members
        cut = model.copy()
        cut[:, years > init] = np.nan
        filtered = matching.apply_filter(cut if acfg["model_filter_at_init"] == "cut" else model, acfg["filter"])
        model_win = filtered[:, mask]
        coverage = np.isfinite(model[:, mask]).mean(axis=1)
        model_win = np.where((coverage >= acfg["min_window_coverage"])[:, None], model_win, np.nan)
        at_leads = matching.align_leads(model, years, int(init), leads)
        for ss in range(obs.shape[0]):
            order = matching.rank(matching.scores(obs_win[ss], model_win, measure), measure)[: acfg["M"]]
            mean[ss, ii] = np.nanmean(at_leads[order], axis=0)
            std[ss, ii] = np.nanstd(at_leads[order], axis=0)
    return xr.Dataset(
        {
            "synthetic_obs": synth,
            "mean": (("syn", "init_year", "lead"), mean),
            "std": (("syn", "init_year", "lead"), std),
        },
        coords={"init_year": inits, "lead": leads},
    )
