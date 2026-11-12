"""
Box-difference indices, e.g. the station-like NAO: mean over a southern box minus mean over a
northern box (Azores - Iceland, Smith et al. 2019 boxes in the old `config.NAO_box`).
Anomalies with respect to each dataset's own normal-period mean, standardized with the
reference dataset's normal-period statistics (as for the EOF indices).
"""

import xarray as xr

from climate_scenarios_2050.preprocess.indices.eof import coslat_weights


def box_mean(field: xr.DataArray, box: dict) -> xr.DataArray:
    sub = field.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
    w = coslat_weights(sub["lat"]) ** 2
    return (sub * w).sum(["lat", "lon"], skipna=False) / (w.sum() * sub.sizes["lon"])


def raw_index(field: xr.DataArray, boxes: dict, climatology: xr.DataArray) -> xr.DataArray:
    anom = field - climatology
    return box_mean(anom, boxes["south"]) - box_mean(anom, boxes["north"])


def reference_stats(field: xr.DataArray, boxes: dict, normal_period, climatology) -> xr.Dataset:
    raw = raw_index(field.sel(year=slice(*normal_period)), boxes, climatology)
    dims = [d for d in raw.dims]
    return xr.Dataset({"score_mean": raw.mean(dims), "score_std": raw.std(dims)})


def project(field, stats, boxes, climatology) -> xr.DataArray:
    return ((raw_index(field, boxes, climatology) - stats["score_mean"]) / stats["score_std"]).rename("index")
