"""
Produce the entire modeling chain, i.e. hindcast NAO as input to an a priori derived linear model that predicts
wind internal variability from NAO internal variability
Model parameters are estimated for each initialization year (sort of) independently
"""

import xarray as xr
from pathlib import Path

def prep_subs_hc(ds):
    ds_cp = ds.copy()
    if "member_id" in ds.coords:
        ds_cp = ds_cp.drop("member_id")
    if "m_id" in ds.coords:
        ds_cp = ds_cp.drop("m_id")
    return ds_cp

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
# choose parameter set for the analog model:
hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : scenario, 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}

# load verification data:
if variable == "wsp100":
    vf_ds = xr.open_dataarray(f"/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/{variable}_GWAscaled_1940-2024_.nc")
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
vf_ds_forced_full = beta_vf_full * NAO_MMEM_centered_full.sel(year=vf_ds_ann.year)
# remove the forced response from the full wind field to get internal variability:

# THIS IS THE FIELD THAT THE HINDCASTS + LINEAR MODEL ARE EVALUATED AGAINST:
vf_ds_annual_internal_full = (vf_ds_annual_centered_full - vf_ds_forced_full).rolling(year=Nyr,center=True).mean("year")

# LOAD STATISTICAL MODEL PARAMETERS (FITTED TO 9-YEAR MEANS OF INTERNAL VARIABILITY):
# load linear model parameters (derived in `linear_model_wsp_NAO.py`):
model_params = xr.open_dataset(database/f"wsp100_NAO_linear_model_params_internal_var_{scenario}.nc")

# LOAD NAO HINDCASTS:
hc_path = '/projects/NS9873K/owul/projects/climate_scenarios_2050data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
ds_hc = xr.open_mfdataset(hindcast_files,preprocess=prep_subs_hc)

# process into rolling means
ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()

# re-arrange the hindcast array to have lead time as dimension instead of valid year
# HINDCAST INPUT TO STATISTICAL MODEL:
ds_hc_proc_bylt = by_year_struct_to_by_leadtime(ds_hc_proc).rename({"year":"vf_year"}).rename({"init_year":"year"})


# APPLY STATISTICAL MODEL TO HINDCASTS:
lead_times_select = ds_hc_proc_bylt.lead_time.values[4::5] # selected lead times, take any subset of (0, 35)
# prediction of `Nyr`-year average internal variability component in the predictand (wsp or wp):
var_pred = model_params["slope"] * ds_hc_proc_bylt.mean("member").sel(lead_time=lead_times_select) + model_params["intercept"]
# full variability (internal + forced) prediction
var_pred_full = var_pred + vf_ds_forced_full.rolling(year=Nyr,center=True).mean("year") + vf_ds_ann.mean(dim="year")
