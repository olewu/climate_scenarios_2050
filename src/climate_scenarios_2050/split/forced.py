"""
Forced/internal split.

The forced signal of a series y (a CMIP6 single-model mean, an observed index, observed wind at a
grid point) is estimated by least squares on a set of regressors ("basis"):

    y(t) = c0 + sum_k c_k * basis_k(t) + internal(t)

- method `mmem`: one regressor, the multi-model ensemble mean (as before: scaling the MMEM onto
  each model/observation, c1 = cov(y, MMEM)/var(MMEM))
- method `lowpass_mmem`: one regressor, the low-pass filtered MMEM
- method `polynomial`: powers of time (a trend of arbitrary shape)

Observed targets use the predictor's basis (`targets.<name>.forced: index`, e.g. the NAO MMEM for wind)
or, per grid point, the MMEM of their own CMIP6 variable (`forced: cmip6`, e.g. tas): the design matrix
then has space dims (year, term, lat, lon) and every point is fitted on its own regressor.

The MMEM itself contains no observations; leakage only enters through the coefficients fitted to
observations. They are therefore fitted separately per "case", each with its own set of excluded
years (see the `*_exclusions` functions). A case only stores coefficients, the internal part of
any series follows from `internal = series - design * coef`.
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.filters import lowpass
from climate_scenarios_2050.core.timeconv import window_label

log = logging.getLogger(__name__)


#---------------basis--------------#
def multi_model_mean(index: xr.DataArray, weighting: str = "model", min_model_fraction: float = 0.5) -> xr.DataArray:
    """
    ensemble mean over (member, year) data with a `model_id` coordinate. `model`: mean of the
    single-model means, `member`: mean over all members. Years with fewer than
    `min_model_fraction` of the models are NaN (e.g. the last years, when runs end).
    """
    if weighting not in ["model", "member"]:
        raise ValueError(f"unknown weighting {weighting}")
    smem = index.groupby("model_id").mean("member")
    n_models = smem.notnull().sum("model_id")
    mmem = smem.mean("model_id") if weighting == "model" else index.mean("member")
    return mmem.where(n_models >= min_model_fraction * smem.sizes["model_id"])


def make_basis(mmem: xr.DataArray, split_cfg: dict) -> xr.DataArray:
    """regressors with dims (term, year), defined on the years of the MMEM"""
    method = split_cfg["method"]
    mmem = mmem.dropna("year")
    if method == "mmem":
        terms = {"mmem": mmem.values}
    elif method == "lowpass_mmem":
        terms = {"mmem_lowpass": lowpass(mmem.values, split_cfg["lowpass"]["cutoff_period"], split_cfg["lowpass"]["order"])}
    elif method == "polynomial":
        x = (mmem.year.values - mmem.year.values.mean()) / (np.ptp(mmem.year.values) / 2)
        terms = {f"t{p}": x**p for p in range(1, split_cfg["polynomial_degree"] + 1)}
    else:
        raise ValueError(f"unknown split method {method}")
    return xr.DataArray(
        np.stack(list(terms.values())), dims=("term", "year"),
        coords={"term": list(terms), "year": mmem.year.values}, name="basis",
    )


def design_matrix(basis: xr.DataArray, years, lag: int = 0) -> xr.DataArray:
    """
    design matrix (year, term) on `years` including the intercept; with `lag`, the basis of
    year t-lag is used in year t (annual wind is paired with the previous DJFM season)
    """
    shifted = basis.assign_coords(year=basis.year + lag).reindex(year=np.asarray(years))
    intercept = xr.ones_like(shifted.isel(term=0, drop=True)).expand_dims(term=["intercept"])
    return xr.concat([intercept, shifted], dim="term").transpose("year", "term")


def pointwise_design(field: xr.DataArray, years) -> xr.DataArray:
    """
    design (year, term, space...) with a regressor per grid point: intercept + `field` (year, space...).
    The regressor is centred on its mean over all years of `field` (e.g. tas ~285 K varying by ~1 K: the
    normal equations would be badly conditioned), which only changes the meaning of the intercept.
    """
    field = field.astype("float64")
    shifted = (field - field.mean("year")).reindex(year=np.asarray(years))
    terms = xr.DataArray(["intercept", "mmem_cmip6"], dims="term", name="term")
    return xr.concat([xr.ones_like(shifted), shifted], dim=terms).transpose("year", "term", ...)


def forced_values(design: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """forced part (year, points) of design (year, term) or (year, term, points) and coef (term, points)"""
    if design.ndim == 2:
        return design @ coef
    return np.einsum("ytp,tp->yp", design, coef)


#---------------exclusions (True = year not used for fitting)--------------#
def full_period_exclusions(years) -> xr.DataArray:
    return xr.DataArray(np.zeros(len(years), dtype=bool), dims="year", coords={"year": np.asarray(years)})


def up_to_init_exclusions(years, init_years) -> xr.DataArray:
    """everything after the init year is excluded (real forecast situation)"""
    years = np.asarray(years)
    init_years = np.asarray(init_years)
    return xr.DataArray(years[None, :] > init_years[:, None], dims=("init_year", "year"), coords={"init_year": init_years, "year": years})


def window_exclusions(years, init_years, windows, lag: int = 0, reference_length: int = 20) -> xr.DataArray:
    """
    leave-target-out (`C`): the target window and its reference period are excluded. With `lag`,
    the years are those of a predictor paired with the targets (see `core.timeconv`).
    Cases whose window or reference period is not fully covered by `years` are excluded
    entirely (no coefficients are fitted for them).
    """
    years = np.asarray(years)
    init_years = np.asarray(init_years)
    out = np.zeros((len(init_years), len(windows), len(years)), dtype=bool)
    for iw, (a, b) in enumerate(windows):
        for ii, init in enumerate(init_years):
            if init - reference_length + 1 - lag < years.min() or init + b - lag > years.max():
                out[ii, iw] = True
                continue
            in_ref = (years >= init - reference_length + 1 - lag) & (years <= init - lag)
            in_win = (years >= init + a - lag) & (years <= init + b - lag)
            out[ii, iw] = in_ref | in_win
    return xr.DataArray(out, dims=("init_year", "window", "year"), coords={"init_year": init_years, "window": [window_label(w) for w in windows], "year": years})


#---------------fitting--------------#
def fit_coefficients(series: xr.DataArray, design: xr.DataArray, excluded: xr.DataArray, min_fit_years: int = 15) -> xr.DataArray:
    """
    least-squares coefficients for all cases in `excluded` (dims: case dims + year). `series` has
    dims (year, ...); grid points with missing values in the used years get NaN.
    Returns dims: case dims + (term, ...).
    """
    if design.ndim > 2:
        return _fit_pointwise(series, design, excluded, min_fit_years)
    series = series.transpose("year", ...)
    space_dims = list(series.dims[1:])
    design = design.sel(year=series.year)
    excluded = excluded.sel(year=series.year)
    case_dims = [d for d in excluded.dims if d != "year"]
    excl = excluded.transpose(*case_dims, "year").values if case_dims else excluded.values[None, :]

    Y = series.values.reshape(series.sizes["year"], -1)
    X = design.values
    year_valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).any(axis=1)

    n_cases = int(np.prod(excl.shape[:-1]))
    coef = np.full((n_cases, X.shape[1], Y.shape[1]), np.nan, dtype="float64")
    for ic, row in enumerate(excl.reshape(n_cases, -1)):
        keep = year_valid & ~row
        if keep.sum() < min_fit_years:
            continue
        Y_keep = Y[keep]
        finite = np.isfinite(Y_keep)
        col_ok = finite.all(axis=0)
        coef[ic][:, col_ok] = np.linalg.pinv(X[keep]) @ Y_keep[:, col_ok]
        # series covering only part of the years (e.g. runs ending earlier) are fitted on their own years
        for col in np.flatnonzero(finite.any(axis=0) & ~col_ok):
            rows = finite[:, col]
            if rows.sum() >= min_fit_years:
                coef[ic][:, col] = np.linalg.pinv(X[keep][rows]) @ Y_keep[rows, col]

    case_shape = [excluded.sizes[d] for d in case_dims]
    space_shape = [series.sizes[d] for d in space_dims]
    coords = {d: excluded[d] for d in case_dims}
    coords["term"] = design["term"]
    coords.update({d: series[d] for d in space_dims})
    return xr.DataArray(coef.reshape(*case_shape, X.shape[1], *space_shape), dims=(*case_dims, "term", *space_dims), coords=coords, name="coef")


def _fit_pointwise(series: xr.DataArray, design: xr.DataArray, excluded: xr.DataArray, min_fit_years: int) -> xr.DataArray:
    """`fit_coefficients` for a design with space dims: least squares per point on the point's own years"""
    series = series.transpose("year", ...)
    space_dims = list(series.dims[1:])
    design = design.sel(year=series.year).transpose("year", "term", *space_dims)
    excluded = excluded.sel(year=series.year)
    case_dims = [d for d in excluded.dims if d != "year"]
    excl = excluded.transpose(*case_dims, "year").values if case_dims else excluded.values[None, :]

    n_years, n_terms = series.sizes["year"], design.sizes["term"]
    Y = series.values.reshape(n_years, -1).astype("float64")
    X = design.values.reshape(n_years, n_terms, -1).astype("float64")
    valid = np.isfinite(Y) & np.isfinite(X).all(axis=1)          # (year, point)
    Y0 = np.where(valid, Y, 0.0)
    X0 = np.where(valid[:, None], X, 0.0)

    n_cases = int(np.prod(excl.shape[:-1]))
    coef = np.full((n_cases, n_terms, Y.shape[1]), np.nan, dtype="float64")
    for ic, row in enumerate(excl.reshape(n_cases, -1)):
        use = valid & ~row[:, None]
        ok = use.sum(axis=0) >= min_fit_years
        if not ok.any():
            continue
        Xu = X0[:, :, ok] * use[:, None, ok]
        gram = np.einsum("ytp,ysp->pts", Xu, X0[:, :, ok])
        rhs = np.einsum("ytp,yp->pt", Xu, Y0[:, ok])
        coef[ic][:, ok] = np.einsum("pts,ps->tp", np.linalg.pinv(gram), rhs)

    case_shape = [excluded.sizes[d] for d in case_dims]
    space_shape = [series.sizes[d] for d in space_dims]
    coords = {d: excluded[d] for d in case_dims}
    coords["term"] = design["term"]
    coords.update({d: series[d] for d in space_dims})
    return xr.DataArray(coef.reshape(*case_shape, n_terms, *space_shape), dims=(*case_dims, "term", *space_dims), coords=coords, name="coef")


def forced_part(design: xr.DataArray, coef: xr.DataArray) -> xr.DataArray:
    """the forced signal (incl. the intercept) implied by the coefficients"""
    return (design * coef).sum("term", skipna=False)


#---------------CMIP6--------------#
def split_ensemble(index: xr.DataArray, basis: xr.DataArray) -> xr.Dataset:
    """
    per model: fit the basis to the single-model mean; forced(model, year) = fit,
    internal(member, year) = member - forced of its model
    """
    smem = index.groupby("model_id").mean("member")
    design = design_matrix(basis, index.year.values)
    coef = fit_coefficients(smem.transpose("year", "model_id"), design, full_period_exclusions(index.year.values))
    forced = forced_part(design, coef).transpose("model_id", "year")
    internal = index - forced.sel(model_id=index["model_id"]).drop_vars("model_id")
    return xr.Dataset({"internal": internal.transpose("member", "year"), "forced": forced, "coef": coef})
