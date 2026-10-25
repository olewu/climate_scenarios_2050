"""
Verify the entire modeling chain, i.e. hindcast NAO as input to an a priori derived linear model that predicts
wind internal variability from NAO internal variability
Model parameters are estimated for each initialization year (sort of) independently
"""

import xarray as xr
import numpy as np
from pathlib import Path
from climate_scenarios_2050.config import proj_base
from climate_scenarios_2050.evaluate import prep_subs_hc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def by_year_struct_to_by_leadtime(by_year_ds,fixed_order_dim="init_year",replc_dim="year",new_dim="lead_time",new_dim_cutoff_val=35):

    by_leadtime_ds = []
    for iyear in by_year_ds[fixed_order_dim].values:
        single_init = by_year_ds.sel({fixed_order_dim:iyear})
        single_init_shifted = single_init.sel({replc_dim:single_init[new_dim] > 0}).swap_dims({replc_dim:new_dim})
        single_init_incl = single_init_shifted.sel({new_dim:slice(new_dim_cutoff_val)})
        by_leadtime_ds.append(single_init_incl)

    return xr.concat(by_leadtime_ds,dim=fixed_order_dim)


database = Path("/datalake/NS9873K/DATA/CMIP6/wind_db/")

scenario = "ssp370" # one of ["ssp245", "ssp370"] but does not make a difference since most of the verification period is part of "hist"
variable = "wsp100" # must be in ["wsp100", "energy"]
Nyr = 8 # running average window [years]
# pt_sel = {"latitude":61.75,"longitude":6.75}
# choose parameter set for the analog model:
hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : scenario, 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}

# load verification data:
if variable == "wsp100":
    vf_ds = xr.open_dataarray(f"/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/{variable}_GWAscaled_1940-2024.nc")
elif variable == "energy":
    vf_ds = xr.open_dataset(f"/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/N175-120m_dry-dens_1940-2024.nc").energy

# full verification field, remove forced component for verification:
vf_ds_ann = vf_ds.groupby(vf_ds.time.dt.year).mean()

NAO_MMEM = xr.open_dataarray(f"/datalake/NS9873K/DATA/CMIP6/wind_db/NAO_MMEM_{scenario}.nc")
NAO_MMEM_centered_full = NAO_MMEM - NAO_MMEM.sel(year=vf_ds_ann.year).mean(dim="year")

# remove forced signal from wind speed for the final verification
# compute forcing and internal component here, otherwise the forced component part will be "perfect"
vf_ds_annual_centered_full = vf_ds_ann - vf_ds_ann.mean(dim="year") # center
# note that this is strictly speaking leaking future information, since beta is estimated from the full NAO times series (just like a trend!), but this is hard to avoid
beta_vf_full = xr.cov(vf_ds_annual_centered_full,NAO_MMEM_centered_full.sel(year=vf_ds_ann.year),dim="year")/NAO_MMEM_centered_full.sel(year=vf_ds_ann.year).var(dim="year")
vf_ds_forced_full = beta_vf_full * NAO_MMEM_centered_full #.sel(year=vf_ds_ann.year)
# remove the forced response from the full wind field to get internal variability:

# THIS IS THE FIELD THAT THE HINDCASTS + LINEAR MODEL ARE EVALUATED AGAINST:
vf_ds_annual_internal_full = (vf_ds_annual_centered_full - vf_ds_forced_full).rolling(year=Nyr,center=True).mean("year")

# LOAD STATISTICAL MODEL PARAMETERS (FITTED TO 9-YEAR MEANS OF INTERNAL VARIABILITY):
# load linear model parameters (derived in `linear_model_wsp_NAO.py`):
model_params = xr.open_dataset(database/f"wsp100_NAO_linear_model_params_internal_var_{scenario}.nc")
# model_params_local = model_params.sel(pt_sel)

# LOAD NAO HINDCASTS:
hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
ds_hc = xr.open_mfdataset(hindcast_files,preprocess=prep_subs_hc)

# process into rolling means
ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()

# re-arrange the hindcast array to have lead time as dimension instead of valid year
# HINDCAST INPUT TO STATISTICAL MODEL:
ds_hc_proc_bylt = by_year_struct_to_by_leadtime(ds_hc_proc)


# APPLY STATISTICAL MODEL TO HINDCASTS:
lead_times_select = ds_hc_proc_bylt.lead_time.values[4::5] # selected lead times, take any subset of (0, 35)
# prediction of 9-year average internal variability component in the predictand (wsp or wp):
var_pred = model_params["slope"].rename(year="init_year") * ds_hc_proc_bylt.mean("member").sel(lead_time=lead_times_select) + model_params["intercept"].rename(year="init_year")
# # full variability (internal + forced) prediction
forced_rm = vf_ds_forced_full.rolling(year=Nyr,center=True).mean("year")
clim = vf_ds_ann.mean(dim="year")
# var_pred_full = var_pred


# VERIFY THE PREDICTION:
# internal variability component
corr = []; rmse = []; nrmse = []; SS_res = []; SS_tot = []; Rsq = []; sgn_agree_perc = []
# full variability
corr_full = []; rmse_full = []; nrmse_full = []; SS_res_full = []; SS_tot_full = []; Rsq_full = []; sgn_agree_perc_full = []
for lt in lead_times_select:
    print(lt)
    vf_range = [] # leave empty to verify all available time steps!
    
    # INTERNAL VARIABILITY COMPONENTS:
    pred_int = var_pred.sel(lead_time=lt)
    pred_int = pred_int.swap_dims({"init_year":"year"})
    if vf_range:
        pred_int = pred_int.sel(year=slice(*vf_range))
    vf_int, pred_int = xr.align(vf_ds_annual_internal_full, pred_int, join="inner")

    # FULL VARIABILITY:
    pred_full = pred_int + forced_rm + clim
    vf_full = vf_ds_ann.rolling(year=Nyr,center=True).mean("year")
    vf_full, pred_full = xr.align(vf_full, pred_full, join="inner")

    N_samples = len(vf_full.year)

    print(N_samples)

    corr.append(xr.corr(vf_int,pred_int,dim="year"))
    RMSE = ((vf_int - pred_int)**2).mean(dim="year")**.5
    rmse.append(RMSE)# root mean squared error
    nrmse.append(1 - RMSE/vf_int.std(dim="year")) # normalized RMSE
    ss_res = ((vf_int - pred_int)**2).sum(dim="year")
    ss_tot = ((vf_int - vf_int.mean(dim="year"))**2).sum(dim="year")
    SS_res.append(ss_res)
    SS_tot.append(ss_tot)
    Rsq.append(1 - ss_res/ss_tot)

    # verify directional change:
    sgn_agree_perc_raw = 1 - ((np.sign(vf_int.diff(dim="year")) + np.sign(pred_int.diff(dim="year"))) == 0).mean(dim="year")
    # mask nans again:
    sgn_agree_perc.append(xr.where(~pred_int.isel(year=0).isnull(),sgn_agree_perc_raw,np.nan))


    corr_full.append(xr.corr(vf_full,pred_full,dim="year"))
    RMSE_full = ((vf_full - pred_full)**2).mean(dim="year")**.5
    rmse_full.append(RMSE_full)# root mean squared error
    nrmse_full.append(1 - RMSE_full/vf_full.std(dim="year")) # normalized RMSE
    ss_res_full = ((vf_full - pred_full)**2).sum(dim="year")
    ss_tot_full = ((vf_full - vf_full.mean(dim="year"))**2).sum(dim="year")
    SS_res_full.append(ss_res_full)
    SS_tot_full.append(ss_tot_full)
    Rsq_full.append(1 - ss_res_full/ss_tot_full)

    # verify directional change:
    sgn_agree_perc_full_raw = 1 - ((np.sign(vf_full.diff(dim="year")) + np.sign(pred_full.diff(dim="year"))) == 0).mean(dim="year")
    # mask nans again:
    sgn_agree_perc_full.append(xr.where(~pred_full.isel(year=0).isnull(),sgn_agree_perc_full_raw,np.nan))

corr = xr.concat(corr,dim="lead_time"); corr_full = xr.concat(corr_full,dim="lead_time")
rmse = xr.concat(rmse,dim="lead_time"); rmse_full = xr.concat(rmse_full,dim="lead_time")
nrmse = xr.concat(nrmse,dim="lead_time"); nrmse_full = xr.concat(nrmse_full,dim="lead_time")
SS_res = xr.concat(SS_res,dim="lead_time"); SS_res_full = xr.concat(SS_res_full,dim="lead_time")
SS_tot = xr.concat(SS_tot,dim="lead_time"); SS_tot_full = xr.concat(SS_tot_full,dim="lead_time")
Rsq = xr.concat(Rsq,dim="lead_time"); Rsq_full = xr.concat(Rsq_full,dim="lead_time")
sgn_agree_perc = xr.concat(sgn_agree_perc,dim="lead_time"); sgn_agree_perc_full = xr.concat(sgn_agree_perc_full,dim="lead_time")

# full time series (forced + internal) OLD; WRONG!!!
# corr_full = xr.corr(vf_ds_ann.rolling(year=Nyr,center=True).mean("year"),var_pred_full,dim="year")
# rmse_full = ((vf_ds_ann.rolling(year=Nyr,center=True).mean("year") - var_pred_full)**2).mean(dim="year")**.5 # root mean squared error
# nrmse_full = 1 - rmse_full/vf_ds_ann.rolling(year=Nyr,center=True).mean("year").std(dim="year") # normalized RMSE
# SS_res_full = ((vf_ds_ann.rolling(year=Nyr,center=True).mean("year") - var_pred_full)**2).sum(dim="year")
# SS_tot_full = ((vf_ds_ann.rolling(year=Nyr,center=True).mean("year") - vf_ds_ann.rolling(year=Nyr,center=True).mean("year").mean(dim="year"))**2).sum(dim="year")
# Rsq_full = 1 - SS_res_full/SS_tot_full
# sgn_agree_perc_full = 1 - ((np.sign(vf_ds_ann.rolling(year=Nyr,center=True).mean("year").diff(dim="year")) + np.sign(var_pred_full.diff(dim="year"))) == 0).mean(dim="year")
# sgn_agree_perc_full = xr.where(~var_pred.isel(year=0).isnull(),sgn_agree_perc_full,np.nan)


# PLOT SKILL FOR SINGLE LEAD TIME
LT = 5

f,ax = plt.subplots(3,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,8))
sgn_agree_perc.sel(lead_time=LT).plot(ax=ax[0],vmin=0.2,vmax=.8,cmap="RdBu_r")
ax[0].coastlines()
sgn_agree_perc_full.sel(lead_time=LT).plot(ax=ax[1],vmin=0.2,vmax=.8,cmap="RdBu_r")
ax[1].coastlines()
(sgn_agree_perc - sgn_agree_perc_full).sel(lead_time=LT).plot(ax=ax[2],vmin=-.2,vmax=.2,cmap="RdBu_r")
ax[2].coastlines()
f.suptitle("Directional Change agreement")


f,ax = plt.subplots(3,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,8))
corr.sel(lead_time=LT).plot(ax=ax[0],vmin=-1,vmax=1,cmap="RdBu_r")
ax[0].coastlines()
corr_full.sel(lead_time=LT).plot(ax=ax[1],vmin=-1,vmax=1,cmap="RdBu_r")
ax[1].coastlines()
(corr - corr_full).sel(lead_time=LT).plot(ax=ax[2],vmin=-1,vmax=1,cmap="RdBu_r")
ax[2].coastlines()
f.suptitle("correlation")

f,ax = plt.subplots(3,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,8))
nrmse.sel(lead_time=LT).plot(ax=ax[0],vmin=-.3,vmax=.3,cmap="RdBu_r")
ax[0].coastlines()
nrmse_full.sel(lead_time=LT).plot(ax=ax[1],vmin=-.3,vmax=.3,cmap="RdBu_r")
ax[1].coastlines()
(nrmse - nrmse_full).sel(lead_time=LT).plot(ax=ax[2],vmin=-.3,vmax=.3,cmap="RdBu_r")
ax[2].coastlines()
f.suptitle("NRMSE")

f,ax = plt.subplots(3,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,8))
Rsq.sel(lead_time=LT).plot(ax=ax[0],vmin=-.5,vmax=.5,cmap="RdBu_r")
ax[0].coastlines()
Rsq_full.sel(lead_time=LT).plot(ax=ax[1],vmin=-.5,vmax=.5,cmap="RdBu_r")
ax[1].coastlines()
(Rsq - Rsq_full).sel(lead_time=LT).plot(ax=ax[2],vmin=-.5,vmax=.5,cmap="RdBu_r")
ax[2].coastlines()
f.suptitle("R$^{2}$")

# Variance ratio of forced vs full variability in the verification:
var_ratio = vf_ds_forced_full.rolling(year=Nyr,center=True).mean("year").var("year")/vf_ds_ann.rolling(year=Nyr,center=True).mean("year").var("year")
f,ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,3))
var_ratio.plot(ax=ax,vmin=0,vmax=.2)
ax.coastlines()

# probabilistic verification?
# generate samples around each ensemble member from the uncertainty (sigma) in the linear model prediction:
