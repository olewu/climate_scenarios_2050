import xarray as xr
import xesmf as xe
import pandas as pd
import numpy as np
from climate_scenarios_2050.config import *


def interpolate_grid(filename_in,grid_spacing,variable,preprocess=None,engine='zarr',method='bilinear',concat_dim='member'): # start_sel,end_sel

    if isinstance(filename_in,(Path,str,list)):
        # load file:
        ds_in = xr.open_mfdataset(filename_in,concat_dim=concat_dim,combine='nested',preprocess=preprocess,engine=engine,parallel=False)
        ds_in = ds_in.chunk({'member': 20,'time': 100,'lat': -1,'lon':-1})
    else:
        print('{:} is not a valid input format for `filename_in`'.format(type(filename_in)))

    # xESMF expects the coordinate names to be exactly 'lat' and 'lon'.
    # So rename them if needed:
    if 'lat' not in ds_in.coords:
        ds_in = ds_in.cf.rename({"Y": "lat"})
    if 'lon' not in ds_in.coords:
        ds_in = ds_in.cf.rename({"X": "lon"})

    # create the grid from given grid spacing:
    lat_out_half = np.arange(grid_spacing['lat']*1/2, 90.1,grid_spacing['lat'])
    lat_out = np.concatenate([np.flip(-lat_out_half),lat_out_half])
    lon_out = np.arange(grid_spacing['lon']*1/2, 360, grid_spacing['lon'])
    to_grid = xr.Dataset({"lat": (["lat"], lat_out),"lon": (["lon"], lon_out),})

    # create the regridder 
    # "bilinear" is the standard choice, but you can also use
    # conservative, patch, nearest_s2d, nearest_d2s, etc.
    # If your data are global and cross the 0/360 boundary, set periodic=True.   
    regridder = xe.Regridder(ds_in, to_grid, method=method, ignore_degenerate=True, periodic=True)

    # apply the regridder to the variable
    return regridder(ds_in[variable])