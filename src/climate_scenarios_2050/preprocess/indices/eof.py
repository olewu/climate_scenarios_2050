"""
EOF-based indices (NAO = mode 1 of DJFM SLP over the North Atlantic, East Atlantic pattern = mode 2, ...).

Pattern
-------
Fitted on one dataset over the normal period: anomalies with respect to the normal-period
mean, optionally linearly detrended (as before), sqrt(cos(lat))-weighted EOFs (xeofs).
The sign of each mode is fixed so that the pattern is positive at `sign_reference_point`.
Stored: unit-norm components V (in weighted space), the fit-sample score mean/std
(standardization), explained variance ratio.

Index
-----
For any dataset (obs or CMIP6 member): anomalies with respect to the dataset's own
normal-period mean (for CMIP6: mean over the model's members that cover the normal period),
projected onto V with the same weights, then standardized with the pattern's fit-sample
statistics. Models projected onto an observed pattern therefore keep their amplitude relative
to the observations.

"Own" patterns: every dataset/model gets its own pattern (CMIP6: fitted on all members of the
model in the normal period; models without complete normal-period coverage use the years they
have, recorded in the output). Its modes are matched to the reference pattern by the highest
absolute (area-weighted) pattern correlation, and their sign aligned with it.
"""

import logging

import numpy as np
import xarray as xr
import xeofs as xe
from scipy.signal import detrend

log = logging.getLogger(__name__)


def coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    return np.sqrt(np.cos(np.deg2rad(lat))).clip(min=0)


def _detrend_year(da: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(detrend, da, input_core_dims=[["year"]], output_core_dims=[["year"]], vectorize=True)


def fit_pattern(field: xr.DataArray, normal_period, n_modes: int = 10, detrend_fit: bool = True, sign_point: dict = None, source: str = "") -> xr.Dataset:
    """
    fit the EOF pattern on `field` with dims (year, lat, lon) or (member, year, lat, lon)
    restricted to the normal period; samples (member-years) with missing values are dropped
    """
    sub = field.sel(year=slice(*normal_period))
    sample_dims = [d for d in ["member", "year"] if d in sub.dims]
    if "member" in sub.dims:
        # drop member-years without data (e.g. members starting after the normal period begins)
        stacked = sub.stack(sample=sample_dims)
        valid = stacked.notnull().all(["lat", "lon"])
        clim = stacked.isel(sample=valid.values).mean("sample")
        anom = sub - clim
        if detrend_fit:
            # detrend each member over its available years
            anom = xr.concat([_detrend_year(anom.sel(member=m).dropna("year", how="all")).reindex(year=anom.year) for m in anom.member.values], dim="member")
        anom = anom.stack(sample=sample_dims)
        anom = anom.isel(sample=anom.notnull().all(["lat", "lon"]).values)
        years_used = np.unique(anom["year"].values)
    else:
        clim = sub.mean("year")
        anom = sub - clim
        if detrend_fit:
            anom = _detrend_year(anom)
        anom = anom.rename(year="sample")
        years_used = sub.year.values

    model = xe.single.EOF(n_modes=n_modes, use_coslat=True)
    model.fit(anom, dim="sample")
    comps = model.components(normalized=True).transpose("mode", "lat", "lon")
    weights = coslat_weights(comps["lat"])

    scores = (anom * weights * comps).sum(["lat", "lon"])
    if sign_point is not None:
        sign = np.sign(comps.sel(lat=sign_point["lat"], lon=sign_point["lon"], method="nearest"))
        sign = sign.where(sign != 0, 1)
        comps = comps * sign
        scores = scores * sign

    pattern = xr.Dataset(
        {
            "components": comps.drop_vars([c for c in comps.coords if c not in ["mode", "lat", "lon"]]),
            "score_mean": scores.mean("sample"),
            "score_std": scores.std("sample"),
            "explained_variance_ratio": model.explained_variance_ratio(),
            "climatology": clim.drop_vars([c for c in clim.coords if c not in ["lat", "lon"]]),
        },
        attrs={
            "source": source,
            "normal_period": str(list(normal_period)),
            "fit_years": f"{int(years_used.min())}-{int(years_used.max())}",
            "detrended_fit": int(detrend_fit),
            "n_samples": int(anom.sizes["sample"]),
        },
    )
    # xeofs attaches non-serializable attributes
    for name in pattern.variables:
        pattern[name].attrs = {}
    return pattern


def match_modes(own: xr.Dataset, reference: xr.Dataset, modes) -> xr.Dataset:
    """
    reorder/flip the modes of `own` to match the requested reference modes by highest absolute
    area-weighted pattern correlation
    """
    w = coslat_weights(own["lat"]) ** 2
    def pcorr(a, b):
        a = a - (a * w).sum(["lat", "lon"]) / w.sum() / a.sizes["lon"]
        b = b - (b * w).sum(["lat", "lon"]) / w.sum() / b.sizes["lon"]
        return (a * b * w).sum(["lat", "lon"]) / np.sqrt((a**2 * w).sum(["lat", "lon"]) * (b**2 * w).sum(["lat", "lon"]))

    selected = []
    for mode in modes:
        ref = reference["components"].sel(mode=mode)
        corr = pcorr(own["components"], ref)
        best = int(abs(corr).argmax("mode"))
        sign = float(np.sign(corr.isel(mode=best)))
        sel = own[["components", "score_mean", "score_std", "explained_variance_ratio"]].isel(mode=best)
        sel["components"] = sel["components"] * sign
        sel["score_mean"] = sel["score_mean"] * sign
        sel = sel.assign_coords(mode=mode).assign(own_mode=int(own["mode"].values[best]), pattern_corr=abs(float(corr.isel(mode=best))))
        selected.append(sel)
    out = xr.concat(selected, dim="mode")
    out["climatology"] = own["climatology"]
    out.attrs = own.attrs
    return out


def normal_period_climatology(field: xr.DataArray, normal_period) -> xr.DataArray:
    """
    mean over the normal period; for an ensemble (member dim) the mean over the members that
    fully cover the normal period, or over all available member-years if none does
    """
    sub = field.sel(year=slice(*normal_period))
    if "member" not in sub.dims:
        return sub.mean("year")
    full = sub.notnull().all(["year", "lat", "lon"])
    if bool(full.any()):
        return sub.sel(member=full.values).mean(["member", "year"])
    log.warning("no member covers the full normal period, using all available member-years")
    return sub.mean(["member", "year"])


def project(field: xr.DataArray, pattern: xr.Dataset, mode: int, climatology: xr.DataArray) -> xr.DataArray:
    """standardized index: projection of the anomalies onto the pattern's mode"""
    comp = pattern["components"].sel(mode=mode)
    weights = coslat_weights(comp["lat"])
    anom = field - climatology
    score = (anom * weights * comp).sum(["lat", "lon"], skipna=False)
    return ((score - pattern["score_mean"].sel(mode=mode)) / pattern["score_std"].sel(mode=mode)).rename("index")
