"""
Time conventions used throughout the project.

Seasons
-------
A season is a list of months in seasonal order, e.g. [12, 1, 2, 3] (DJFM). It is labelled by
the calendar year of its first month, i.e. DJFM 2020 = Dec 2020 - Mar 2021 (the convention of
the old `make_season_group`). Annual means ([1, ..., 12]) are labelled by the calendar year.
Monthly values are weighted by the number of days of the month in the data's own calendar and
only complete seasons are kept.

Initialization and lead times
-----------------------------
- init year Y: forecast issued after the DJFM season Y (end of March Y+1)
- lead year k: calendar year Y+k (annual quantities, e.g. wind)
- target window (a, b): calendar years Y+a ... Y+b
- reference period: the `length` calendar years Y-length+1 ... Y
- a seasonal predictor (e.g. DJFM NAO) is paired with the following calendar year
  (DJFM season t-1 is used for wind in year t), i.e. lag = 1: predictor window = seasons
  Y+a-1 ... Y+b-1, predictor reference = seasons Y-length ... Y-1
- verification of the predictor itself uses lag = 0 (seasons Y+a ... Y+b)
"""

import numpy as np
import xarray as xr


#---------------seasons--------------#
def season_labels(time: xr.DataArray, months: list) -> xr.DataArray:
    """year label (year of the season's first month) for every time step"""
    first_month = months[0]
    year = time.dt.year
    month = time.dt.month
    return xr.where(month >= first_month, year, year - 1).rename("year")


def seasonal_mean(da: xr.DataArray, months: list, time_dim: str = "time", weighted: bool = True) -> xr.DataArray:
    """
    day-weighted mean over the months of a season for each season year; seasons with missing
    months (or NaN in any month) are dropped/NaN, so partial seasons never enter
    """
    da = da.isel({time_dim: da[time_dim].dt.month.isin(months).values})
    labels = season_labels(da[time_dim], months)
    if weighted:
        weights = da[time_dim].dt.days_in_month.astype("float64")
    else:
        weights = xr.ones_like(labels, dtype="float64")
    weights = weights.assign_coords(year=labels)
    da = da.assign_coords(year=labels)

    weighted_sum = (da * weights).groupby("year").sum(time_dim, skipna=False)
    weight_sum = weights.groupby("year").sum(time_dim)
    n_months = labels.groupby(labels).count()

    seas = weighted_sum / weight_sum
    complete = n_months.values == len(months)
    return seas.isel(year=complete)


def complete_seasons(time_values, months: list) -> np.ndarray:
    """season years for which all months exist in `time_values` (any datetime-like)"""
    time = xr.DataArray(np.asarray(time_values), dims="time")
    labels = season_labels(time, months)
    keep = time.dt.month.isin(months)
    labels = labels[keep]
    unique, counts = np.unique(labels.values, return_counts=True)
    return unique[counts == len(months)]


#---------------windows--------------#
def window_label(window) -> str:
    return f"{window[0]}-{window[1]}"


def _trailing_mean(series: xr.DataArray, length: int, dim: str = "year") -> xr.DataArray:
    """mean over the `length` years ending at each year label; NaN if any value is missing"""
    years = np.arange(int(series[dim].min()), int(series[dim].max()) + 1)
    series = series.reindex({dim: years})
    return series.rolling({dim: length}, min_periods=length).mean()


def window_means(series: xr.DataArray, windows, lag: int = 0, reference_length: int = 20, dim: str = "year") -> xr.Dataset:
    """
    window means and reference means of an observed (or single continuous) series for
    every possible init year.
    Returns a Dataset with `window_mean` (init_year, window) and `reference_mean` (init_year).
    """
    reference = _trailing_mean(series, reference_length, dim)
    # reference for init Y ends at Y - lag:
    reference = reference.assign_coords({dim: reference[dim] + lag}).rename({dim: "init_year"})

    win_means = []
    for window in windows:
        length = window[1] - window[0] + 1
        trailing = _trailing_mean(series, length, dim)
        # window for init Y ends at Y + b - lag:
        trailing = trailing.assign_coords({dim: trailing[dim] - window[1] + lag}).rename({dim: "init_year"})
        win_means.append(trailing.expand_dims(window=[window_label(window)]))
    win_means = xr.concat(win_means, dim="window", join="outer")

    return xr.Dataset({"window_mean": win_means, "reference_mean": reference}).dropna("init_year", how="all")


def hindcast_window_means(hindcast: xr.DataArray, windows, lag: int = 0, dim: str = "year", init_dim: str = "init_year") -> xr.DataArray:
    """
    window means of a hindcast with dims (init_year, year, ...): for each init year Y and
    window (a, b) the mean over years Y+a-lag ... Y+b-lag (NaN unless all years are present)
    """
    rel = hindcast[dim] - hindcast[init_dim]
    out = []
    for window in windows:
        in_window = (rel >= window[0] - lag) & (rel <= window[1] - lag)
        n_needed = window[1] - window[0] + 1
        vals = hindcast.where(in_window)
        n_valid = vals.notnull().sum(dim)
        mean = vals.mean(dim).where(n_valid == n_needed)
        out.append(mean.expand_dims(window=[window_label(window)]))
    return xr.concat(out, dim="window")


def change(window_mean, reference_mean, kind: str = "relative"):
    """change of a window mean relative to the reference mean (relative in %)"""
    if kind == "relative":
        return (window_mean - reference_mean) / reference_mean * 100
    elif kind == "absolute":
        return window_mean - reference_mean
    raise ValueError(f"unknown change kind {kind}")
