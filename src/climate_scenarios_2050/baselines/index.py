"""
Statistical baselines for the NAO hindcasts, with the same layout as the analogue hindcasts
(init_year, member, lead). Negative leads hold the observations known at init.

- AR(1) on the annual internal index up to init, around its linear trend (the trend is not
  extrapolated). `fit_window` limits the fit to the last years ("matching": the analogue matching window,
  resolved in `load_config`) or uses all prior data ("all"). The old `nao_ar1_lofreq_forecast.py` fitted the 8-year running mean,
  available with `fit_on: running_mean`: there values are forecasts of the smoothed index.
- sine + trend on the low-pass filtered index plus MA(1) residuals (old `nao_fft_forecast.py`), deterministic
"""

import logging
import warnings

import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA

from climate_scenarios_2050.analogues.matching import running_mean
from climate_scenarios_2050.core.filters import lowpass

log = logging.getLogger(__name__)


def _ar1_members(values, n_steps, n_members, rng):
    """AR(1) ensemble continuing `values` (the series anomalies) for n_steps"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(values, order=(1, 0, 0)).fit()
    const, phi, sigma2 = fitted.params
    out = np.empty((n_members, n_steps))
    state = np.full(n_members, values[-1])
    for step in range(n_steps):
        state = const + phi * (state - const) + rng.normal(scale=np.sqrt(sigma2), size=n_members)
        out[:, step] = state
    return out


def ar1_hindcast(obs_at_init, inits, leads, bcfg, seed) -> xr.DataArray:
    """obs_at_init(init) -> series with dims (year) of the internal index known at init"""
    rng = np.random.default_rng(seed)
    n_members = bcfg["members"]
    out = np.full((len(inits), n_members, len(leads)), np.nan, dtype="float32")
    smooth = bcfg.get("fit_on", "annual") == "running_mean"
    for ii, init in enumerate(inits):
        series = obs_at_init(int(init)).sel(year=slice(None, int(init)))
        if bcfg["fit_window"] != "all":
            series = series.isel(year=slice(-bcfg["fit_window"], None))
        years, values = series.year.values, series.values
        if smooth:
            N = bcfg["running_mean"]
            smoothed = running_mean(values, N, N)[0]
            keep = np.isfinite(smoothed)
            years, values = years[keep], smoothed[keep]
        trend_fit = np.polyfit(years - years[0], values, 1)
        anomalies = values - np.polyval(trend_fit, years - years[0])
        n_steps = int(init) + int(leads.max()) - int(years[-1])
        sims = _ar1_members(anomalies, n_steps, n_members, rng)
        sim_years = years[-1] + np.arange(1, n_steps + 1)
        for il, lead in enumerate(leads):
            year = int(init) + int(lead)
            if year <= years[-1]:
                match = np.flatnonzero(years == year)
                if match.size:
                    out[ii, :, il] = values[match[0]]
            else:
                out[ii, :, il] = sims[:, np.flatnonzero(sim_years == year)[0]]
    return xr.DataArray(out, dims=("init_year", "member", "lead"), coords={"init_year": inits, "member": np.arange(n_members), "lead": leads})


def persistence_hindcast(obs_at_init, inits, leads, reference_length) -> xr.DataArray:
    """the mean of the last `reference_length` observed values continues at all positive leads"""
    out = np.full((len(inits), 1, len(leads)), np.nan, dtype="float32")
    for ii, init in enumerate(inits):
        series = obs_at_init(int(init)).sel(year=slice(None, int(init)))
        level = float(series.isel(year=slice(-reference_length, None)).mean())
        for il, lead in enumerate(leads):
            if lead > 0:
                out[ii, 0, il] = level
            elif int(init) + int(lead) in series.year.values:
                out[ii, 0, il] = float(series.sel(year=int(init) + int(lead)))
    return xr.DataArray(out, dims=("init_year", "member", "lead"), coords={"init_year": inits, "member": [0], "lead": leads})


def _sine_trend(t, amp, phase, freq, slope, intercept):
    return amp * np.sin(2 * np.pi * freq * t + phase) + slope * t + intercept


def fft_hindcast(obs_at_init, inits, leads, bcfg) -> xr.DataArray:
    """sine + trend fit of the low-pass filtered index, plus MA(1) forecast of the residuals"""
    out = np.full((len(inits), 1, len(leads)), np.nan, dtype="float32")
    cutoff = bcfg["cutoff_period"]
    previous = None
    for ii, init in enumerate(inits):
        series = obs_at_init(int(init)).sel(year=slice(None, int(init)))
        if bcfg["fit_window"] != "all":
            series = series.isel(year=slice(-bcfg["fit_window"], None))
        years, values = series.year.values, series.values
        low = lowpass(values, cutoff)
        t = years - 1850.0
        n = len(low)
        spectrum = np.abs(fft(low))[2 : n // 2]
        freqs = np.abs(fftfreq(n, d=1)[2 : n // 2])
        trend = np.polyfit(t, low, 1)
        spread, slope_max, intercept_max = low.std(), np.ptp(low) / n, 1.5 * np.abs(low).max()
        p0 = [0.2 * spread, 0, float(np.clip(freqs[spectrum.argmax()], 0.011, 0.099)), float(np.clip(trend[0], -slope_max, slope_max)), float(np.clip(trend[1], -intercept_max, intercept_max))]
        bounds = ([0, -np.pi, 0.01, -slope_max, -intercept_max], [2 * spread, np.pi, 0.1, slope_max, intercept_max])
        edge = cutoff // 2
        try:
            popt, _ = curve_fit(_sine_trend, t[edge:-edge], low[edge:-edge], p0=p0, bounds=bounds, maxfev=5000)
            previous = popt
        except (RuntimeError, ValueError):
            popt = previous if previous is not None else [0, 0, 0.05, trend[0], trend[1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(values - low, order=(0, 0, 1)).fit()
        residual = fitted.get_forecast(steps=int(leads.max())).predicted_mean
        for il, lead in enumerate(leads):
            if lead > 0:
                year = int(init) + int(lead)
                out[ii, 0, il] = _sine_trend(year - 1850.0, *popt) + residual[int(lead) - 1]
            else:
                match = np.flatnonzero(years == int(init) + int(lead))
                if match.size:
                    out[ii, 0, il] = values[match[0]]
    return xr.DataArray(out, dims=("init_year", "member", "lead"), coords={"init_year": inits, "member": [0], "lead": leads})
