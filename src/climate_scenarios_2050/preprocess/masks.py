"""
Land-sea mask from ERA5 SST (missing over land), mapped to any grid by nearest neighbour.

Used for
- `preprocess.fields.<field>.global_mean: ocean`: global mean over the ocean points of each model's
  native grid (the ERA5 mask does not follow each model's own coastline exactly; the effect on a global
  mean is small)
- `predictor.mask: ocean`: index field restricted to the ocean on the common grid (e.g. a PDO-like
  index from near-surface air temperature)
"""

import functools

import numpy as np
import xarray as xr


@functools.lru_cache(maxsize=4)
def _era5_ocean(path: str) -> xr.DataArray:
    from climate_scenarios_2050.preprocess.cmip6 import standardize_coords
    from climate_scenarios_2050.preprocess.obs import _prep_era5

    with xr.open_dataset(path) as ds:
        ds = standardize_coords(_prep_era5(ds))
        name = list(ds.data_vars)[0]
        sst = ds[name].isel(time=0, drop=True) if "time" in ds[name].dims else ds[name]
        return sst.notnull().load().rename("ocean")


def ocean_mask(cfg) -> xr.DataArray:
    """True over the ocean on the ERA5 grid (lat ascending, lon -180 ... 180), from the first ERA5 SST file"""
    from climate_scenarios_2050.preprocess.obs import era5_files

    files = era5_files(cfg, "tos")
    if not files:
        raise FileNotFoundError("no ERA5 SST files for the ocean mask (obs_sources.ERA5.variables.tos)")
    return _era5_ocean(str(files[0]))


def on_grid(mask: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """
    mask at the points of another grid (nearest neighbour); `lat`/`lon` 1-D coordinates of a regular grid
    or 2-D coordinates of a curvilinear one, any longitude convention
    """
    lon_std = (lon + 180) % 360 - 180
    if lat.ndim == 1 and lon.ndim == 1 and lat.dims != lon.dims:
        out = mask.sel(lat=xr.DataArray(lat.values, dims=lat.dims), lon=xr.DataArray(lon_std.values, dims=lon.dims), method="nearest")
        return out.drop_vars(["lat", "lon"]).assign_coords({lat.dims[0]: lat.values, lon.dims[0]: lon.values})
    out = mask.sel(lat=xr.DataArray(lat.values, dims=lat.dims), lon=xr.DataArray(lon_std.values, dims=lon.dims), method="nearest")
    return out.drop_vars(["lat", "lon"]).assign_coords(lat=(lat.dims, lat.values), lon=(lon.dims, lon.values))


def apply(cfg, field: xr.DataArray, kind) -> xr.DataArray:
    """field restricted to `kind` (None: unchanged, "ocean")"""
    if kind is None:
        return field
    if kind != "ocean":
        raise ValueError(f"unknown mask {kind}")
    return field.where(on_grid(ocean_mask(cfg), field["lat"], field["lon"]))
