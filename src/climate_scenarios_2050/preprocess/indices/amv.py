"""
Area-mean indices, e.g. the Atlantic Multidecadal Variability (AMV): the area-weighted (cos lat) mean of
the field's anomalies over `predictor.domain` (with `predictor.mask: ocean` over the ocean only), anomalies
with respect to each dataset's own normal-period mean, standardized with the reference dataset's
normal-period statistics (as the box and EOF indices).

No detrending and no low-pass filter here: the forced part (the CMIP6 MMEM, scaled per model and per
observed dataset) is removed in the split like for every other index, and the analogue matching applies
its own filter. With `predictor.remove_global_mean` the global mean is subtracted first (Trenberth and
Shea 2006 instead of the MMEM-based removal alone).

Sparse observations (e.g. DCENT before 1950): the mean is taken over the valid points; a year needs valid
data at `predictor.amv.min_coverage` of the area (cos-lat weighted) of the points the dataset has a
normal-period mean at, otherwise it is NaN.
"""

import numpy as np
import xarray as xr

DEFAULTS = {"min_coverage": 0.5}


def options(pred: dict) -> dict:
    return {**DEFAULTS, **(pred.get("amv") or {})}


def area_mean(anom: xr.DataArray, climatology: xr.DataArray, min_coverage: float) -> xr.DataArray:
    """cos-lat weighted mean over the valid points; NaN where they cover less than `min_coverage`"""
    weights = np.cos(np.deg2rad(anom["lat"])).clip(min=0) * climatology.notnull()
    valid = anom.notnull()
    covered = (valid * weights).sum(["lat", "lon"])
    mean = (anom.fillna(0) * weights).sum(["lat", "lon"]) / covered
    return mean.where(covered >= min_coverage * weights.sum(["lat", "lon"]))


def raw_index(field: xr.DataArray, climatology: xr.DataArray, min_coverage: float) -> xr.DataArray:
    return area_mean(field - climatology, climatology, min_coverage)


def reference_stats(field: xr.DataArray, normal_period, climatology, min_coverage: float) -> xr.Dataset:
    raw = raw_index(field.sel(year=slice(*normal_period)), climatology, min_coverage)
    return xr.Dataset({"score_mean": raw.mean(), "score_std": raw.std()})


def project(field, stats, climatology, min_coverage: float) -> xr.DataArray:
    return ((raw_index(field, climatology, min_coverage) - stats["score_mean"]) / stats["score_std"]).rename("index")
