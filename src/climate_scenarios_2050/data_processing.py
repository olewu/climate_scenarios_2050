from cdo import *
from climate_scenarios_2050 import config
import xarray as xr
from pathlib import Path
import re
import numpy as np

def make_wind_speed_single_time_series_ERA5(lon,lat,pt_name='pt',outpath=None):

    if outpath is None:
        outpath = dpath
    elif isinstance(outpath,str):
        outpath = Path(outpath)
    
    outpath.mkdir(exist_ok=True,parents=True)

    dpath = config.proj_base/'data/ERA5/res_hrly_p25/100m_wind/'

    # for each monthly file, cut out the point and compute speed from components:
    u_files = sorted(list(dpath.glob('100m_uwind*.nc')))

    date_start = re.search(r'_(\d{4}_\d{2})$',u_files[0].stem).group(1)
    date_end = re.search(r'_(\d{4}_\d{2})$',u_files[-1].stem).group(1)

    for uf in u_files:
        
        date = re.search(r'_(\d{4}_\d{2})$',uf.stem).group(1)

        vf = uf.parent/uf.name.replace('uwind','vwind')
        if not vf.exists():
            print(f'corresponding vwind file does not exist for {date}')
            continue

        # open ONE file and search for the index of the closest grid point:
        ds = xr.open_dataset(uf,decode_times=False)
        ds_near = ds.sel(latitude=lat,longitude=lon,method='nearest')
        
        ds2 = xr.open_dataset(vf,decode_times=False)
        ds2_near = ds2.sel(latitude=lat,longitude=lon,method='nearest')

        new_ds = np.sqrt(ds_near.u100 ** 2 + ds2_near.v100 ** 2)
        new_ds.name = 'ws100'

        outf = str(uf).replace('.nc',f'_{pt_name}.nc').replace('uwind','wsp')

        new_ds.to_netcdf(outf)

    all_pt_files = sorted(list(dpath.glob(f'*{pt_name}.nc')))

    cdo = Cdo()

    inp = dpath/f"*{pt_name}.nc"
    outfile = outpath/f"100m_wsp_{date_start}-{date_end}_{pt_name}.nc"

    cdo.mergetime(input=str(inp),output=str(outfile))

    # delete the single month point files (except for the newly created):
    for ptf in all_pt_files:
        if ptf != outfile:
            ptf.unlink()
            
            
def timeseries_to_csv(directory,pattern='*.nc'):

    ts_files = list(directory.glob(pattern))

    for ts_file in ts_files:
        ds = xr.open_dataset(ts_file)
        ds.to_dataframe().to_csv(str(ts_file).replace('.nc','.csv'))

if __name__ == '__main__':

    # go through the set of stations defined in the config:
    for stat,prop in config.stations.items():
        make_wind_speed_single_time_series_ERA5(prop['lon'],prop['lat'],stat,outpath='/projects/NS9873K/www/scenarios_2050/ERA5/100m_wsp/')