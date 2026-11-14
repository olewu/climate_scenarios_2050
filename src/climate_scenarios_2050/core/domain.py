"""
Domains {lon: [west, east], lat: [south, north]} on longitudes -180 ... 180.

A domain with west > east crosses the date line (e.g. the North Pacific, [110, -100]). Fields of such a
domain are kept on longitudes 0 ... 360 (here 110 ... 260), so that they are contiguous and ascending;
all other fields use -180 ... 180.
"""

import numpy as np
import xarray as xr


def crosses_dateline(domain: dict) -> bool:
    return domain["lon"][0] > domain["lon"][1]


def lon_bounds(domain: dict) -> tuple:
    """(west, east) in the longitude convention of the domain's fields"""
    west, east = domain["lon"]
    return (west % 360, east % 360 if east % 360 > west % 360 else east % 360 + 360) if crosses_dateline(domain) else (west, east)


def to_convention(lon, domain: dict):
    """longitudes (array or DataArray) in the convention of the domain"""
    return lon % 360 if crosses_dateline(domain) else (lon + 180) % 360 - 180


def select(da, domain: dict, margin: float = 0.0):
    """
    cut a field on a 1-D regular grid (lat/lon) to the domain (plus margin); the longitudes are converted
    to the domain's convention and sorted first
    """
    west, east = lon_bounds(domain)
    if da["lon"].ndim == 1:
        lon = to_convention(da["lon"].values, domain)
        if not np.array_equal(lon, da["lon"].values):
            da = da.assign_coords(lon=(da["lon"].dims, lon, da["lon"].attrs)).sortby("lon")
        lat = da["lat"].values
        lat_slice = slice(domain["lat"][0] - margin, domain["lat"][1] + margin) if lat[0] <= lat[-1] else slice(domain["lat"][1] + margin, domain["lat"][0] - margin)
        return da.sel(lat=lat_slice, lon=slice(west - margin, east + margin))
    da = da.assign_coords(lon=(da["lon"].dims, to_convention(da["lon"].values, domain), da["lon"].attrs))
    lon = da["lon"]
    mask = (da["lat"] >= domain["lat"][0] - margin) & (da["lat"] <= domain["lat"][1] + margin) & (lon >= west - margin) & (lon <= east + margin)
    return da.where(mask.compute(), drop=True)


def point_lon(lon: float, domain: dict) -> float:
    """a point's longitude in the domain's convention (e.g. a sign reference point)"""
    return float(to_convention(np.asarray(lon), domain))
