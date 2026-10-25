# Make an actual NAO forecast, "initialized" in 2025 (i.e., last NAO observation should be from 2025!)

import xarray as xr
import pandas as pd
from climate_scenarios_2050.evaluate import prep_subs_hc
from climate_scenarios_2050.config import *
from climate_scenarios_2050.utils import make_season_group

import numpy as np
from scipy.stats import linregress
import proplot as pplt
import matplotlib.pyplot as plt
from pathlib import Path

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

def by_year_struct_to_by_leadtime(by_year_ds,fixed_order_dim="init_year",replc_dim="year",new_dim="lead_time",new_dim_cutoff_val=35):

    by_leadtime_ds = []
    for iyear in by_year_ds[fixed_order_dim].values:
        single_init = by_year_ds.sel({fixed_order_dim:iyear})
        single_init_shifted = single_init.sel({replc_dim:single_init[new_dim] > 0}).swap_dims({replc_dim:new_dim})
        single_init_incl = single_init_shifted.sel({new_dim:slice(new_dim_cutoff_val)})
        by_leadtime_ds.append(single_init_incl)

    return xr.concat(by_leadtime_ds,dim=fixed_order_dim)

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

# hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
hc_param_set = {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'varsplit_20y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
# {'subsampling' : "varsplit_7y", 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
# {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},

# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 9

# lead_times: what lead times to verify
lead_times = np.arange(0,36,1)
lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

linestyle = {5:'solid',15:'dashed',25:'dotted',35:'solid'}

index_name = "NAO"
filter_ext = "f-None"

#----------------------------------------------------------------------------------#
# Raw CMIP6 ensemble
#----------------------------------------------------------------------------------#
#-----Create a "baseline" hindcast that just takes the entire CMIP6 ensemble without subsampling
# load all "non-initialized" hindcasts:
index_int_var_model = xr.open_dataarray(CMIP6_base/"internal/internal-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))
index_forced_var_model = xr.open_dataarray(CMIP6_base/"forced/forced-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))
# # full variability:
# index_int_var_model = index_int_var_model.groupby("m_id") + index_forced_var_model

# index_int_var_model.to_netcdf("/projects/NS9873K/www/scenarios_2050/data/nao_{0}_1850-2099.nc".format(hc_param_set["scen"]))

# process (average/filter):
if proc_type == 'running_mean':
    index_int_var_model_proc = index_int_var_model.rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()

index_int_var_model_proc_std = index_int_var_model_proc

# expand the array to look like a hindcast:
index_int_var_model_proc = index_int_var_model_proc.expand_dims(dim={'init_year':len(index_int_var_model_proc.year)}).assign_coords({'init_year':('init_year',index_int_var_model_proc.year.values)})
index_int_var_model_proc = index_int_var_model_proc.assign_coords(lead_time=index_int_var_model_proc.year - index_int_var_model_proc.init_year)

# Observations:
if hc_param_set["ref_ds"] == "ERA5":
    obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
else:
    obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
index_int_var_obs_era = xr.open_dataarray(Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))
index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))
index_forced_var_obs = xr.open_dataarray(obs_ind_dir/"forced-{1}-variability_{0}.nc".format(filter_ext,index_name))
# index_int_var_obs += index_forced_var_obs

# process:
if proc_type == 'running_mean':
    index_int_var_obs_proc = index_int_var_obs.rolling(year=Nyr,center=True).mean('year').dropna('year')


ref_period = [2003,2025]

#----------------------------------------------------------------------------------#
# CMIP6 "initialized"/"hindcasts"
#----------------------------------------------------------------------------------#
# create the path the hindcast are saved to:
hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)

# load all hindcasts:
hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
ds_hc = xr.open_mfdataset(hindcast_files)

ds_hc.sel(init_year=2004).member_id.to_netcdf("/projects/NS9873K/www/scenarios_2050/data/NAO_analog_members_100_init_2004.nc")

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()


# re-arrange the hindcast array to have lead time as dimension instead of valid year
ds_hc_proc_bylt = by_year_struct_to_by_leadtime(ds_hc_proc)

ds_hc_proc_bylt.to_netcdf("/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/NAO_analog_hindcast_anomalies.nc")


ds_hc_proc_bylt_stdz = (ds_hc_proc_bylt - ds_hc_proc_bylt.mean(["init_year","member"]))/ds_hc_proc_bylt.std(["init_year","member"])

# use NAO to predict wind speed:
# linear model that uses NAO anomaly to predict wind speed anomaly

st_lat = 63.404939
st_lon = 7.865301


# need: ERA5 wind speed (100m), NAO index from ERA5 --> detrend both
# wsp_monthly = xr.open_dataset("/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/wsp100_GWAscaled_1940-2024.nc")
wsp_hrly = xr.open_zarr("/datalake/NS9873K/DATA/SFE/ERA5_processed/wsp_100m_gwa_scaled/100m_wsp_hrly_0.25_1940-2024.zarr")
wsp_monthly = wsp_hrly.sel(latitude=st_lat,longitude=st_lon,method="nearest").resample(time='MS').mean().compute()

season = make_season_group(wsp_monthly,months_in_season=[1,2,3,4,5,6,7,8,9,10,11,12])
wsp_seasonal = wsp_monthly.groupby(season).mean().sel(year=slice("1940-01-01","2023-06-01"))
# detrend & filter
wsp_seasonal_dt = detrend(wsp_seasonal.wsp100,dims=["year"]) + wsp_seasonal.wsp100.mean("year")
wsp_seasonal_dt_roll = wsp_seasonal_dt.rolling(year=Nyr,center=True,min_periods=Nyr).mean().dropna("year")


nao_full = index_int_var_obs + index_forced_var_obs
nao = nao_full.sel(year=slice(wsp_monthly.time.dt.year.min(),wsp_monthly.time.dt.year.max())).squeeze()
# detrend & filter
nao_dt = detrend(nao,dims=["year"]).rolling(year=Nyr,center=True).mean().dropna("year")

# normalize to reference period, standardize
# nao_z = (nao_dt - nao_dt.sel(year=slice(*ref_period)).mean())/nao_dt.sel(year=slice(*ref_period)).std()
nao_z = nao_dt
# express wind anomalies relative to reference period mean
wsp_z = (wsp_seasonal_dt_roll - wsp_seasonal_dt_roll.sel(year=slice(*ref_period)).mean("year"))/wsp_seasonal_dt_roll.sel(year=slice(*ref_period)).mean("year")

x_mean = nao_dt.mean("year")
y_mean = wsp_seasonal_dt_roll.mean("year")
x_mean_std = nao_z.mean("year")
y_mean_std = wsp_z.mean("year")

# the linear model predicts a detrended standardized wind anomaly relative to the reference period
slope = ((nao_dt - x_mean) * (wsp_seasonal_dt_roll - y_mean)).sum("year") / ((nao_dt - x_mean)**2).sum("year")
intercept = y_mean - slope * x_mean

slope_rel = ((nao_z - x_mean_std) * (wsp_z - y_mean_std)).sum("year") / ((nao_z - x_mean_std)**2).sum("year")
intercept_rel = y_mean_std - slope * x_mean_std

wsp_est = slope_rel * nao_z + intercept_rel

# Get historical reference wind farm data
wnd_frm_path = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/wind_farm/vortex.serie.612117.20y 100m UTC+01.0 ERA5.txt")
wnd_frm_data = pd.read_csv(
    wnd_frm_path,
    sep=r'\s+',
    skiprows=2,
    engine='python'
)

wnd_frm_data['date_str'] = wnd_frm_data['YYYYMMDD'].astype(str)
wnd_frm_data['time_str'] = wnd_frm_data['HHMM'].astype(str).str.zfill(4)  # Pad with zero if HHMM is e.g. "900" -> "0900"

# Concatenate to form 'YYYYMMDDHHMM'
combined_str = wnd_frm_data['date_str'] + wnd_frm_data['time_str']

# Parse the combined string
wnd_frm_data['datetime'] = pd.to_datetime(combined_str, format='%Y%m%d%H%M')
wnd_frm_data = wnd_frm_data.drop(columns=["YYYYMMDD","HHMM","date_str","time_str"])

wnd_frm_data_ref_mean = wnd_frm_data[wnd_frm_data.datetime.isin(pd.date_range(str(ref_period[0]),str(ref_period[-1]+1)))]["M(m/s)"].mean()

wnd_frm_data.plot(x="datetime",y="M(m/s)")

plt.hist(wnd_frm_data["M(m/s)"],bins=50,density=True)



index_int_var_obs_per = index_int_var_obs.sel(year=slice(index_int_var_obs_era.year.min(),index_int_var_obs.year.max()))
index_int_var_obs.plot();

index_int_var_obs_era_per = index_int_var_obs_era.sel(year=slice(index_int_var_obs_era.year.min(),index_int_var_obs.year.max()))
index_int_var_obs_era_scaled = (index_int_var_obs_era - index_int_var_obs_era_per.mean()) * (index_int_var_obs_per.std()/index_int_var_obs_era_per.std()) + index_int_var_obs_per.mean()
index_int_var_obs_era_scaled.plot()
