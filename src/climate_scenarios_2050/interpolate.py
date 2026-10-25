import xarray as xr
import xesmf as xe
import pandas as pd
import numpy as np
from climate_scenarios_2050.config import *


def interpolate_grid(filename_in,grid_spacing,variable,preprocess=None,engine='zarr',method='bilinear',concat_dim='member',to_stdgrid=False): # start_sel,end_sel

    if isinstance(filename_in,(Path,str,list)):
        # load file:
        ds_in = xr.open_mfdataset(filename_in,concat_dim=concat_dim,combine='nested',preprocess=preprocess,engine=engine,parallel=False)
        if "latitude" in list(ds_in.coords):
            ds_in = ds_in.rename({'latitude':'lat',"longitude":"lon"})
        chunking = {'time': 100,'lat': -1,'lon':-1}
        if "member" in ds_in.dims:
            chunking["member"] =  20
        ds_in = ds_in.chunk(chunking)
    else:
        print('{:} is not a valid input format for `filename_in`'.format(type(filename_in)))

    # xESMF expects the coordinate names to be exactly 'lat' and 'lon'.
    # So rename them if needed:
    if 'lat' not in ds_in.coords:
        ds_in = ds_in.cf.rename({"Y": "lat"})
    if 'lon' not in ds_in.coords:
        ds_in = ds_in.cf.rename({"X": "lon"})


    # create the grid from given grid spacing:
    if method == 'conservative':
        try:
            lon_b = np.concatenate([
                ds_in.lon_bnds[0, :, 0].values[:1],  # first lower bound
                ds_in.lon_bnds[0, :, 1].values       # all upper bounds
            ])

            lat_b = np.concatenate([
                ds_in.lat_bnds[0, :, 0].values[:1],
                ds_in.lat_bnds[0, :, 1].values
            ])            

            ds_in = ds_in.assign_coords({
                'lon_b': (('lon_b',), lon_b),
                'lat_b': (('lat_b',), lat_b)
            })
        except:
            pass

    if to_stdgrid:
        lat_out_half = np.arange(grid_spacing['lat']*1/2, 90.1,grid_spacing['lat'])
        lat_out = np.concatenate([np.flip(-lat_out_half),lat_out_half])
        lon_out = np.arange(-180+grid_spacing['lon']*1/2, 180, grid_spacing['lon'])
    else:
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