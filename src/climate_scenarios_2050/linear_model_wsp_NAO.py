"""
Make a linear prediction model of the form wsp100/wep = alpha*NAO + beta + eps that makes a 
prediction of wind speed or wind energy production from the state of the NAO
Make a cross-validated version of the model so that it can be applied in hindcast mode,
i.e. for a prediction of wsp100 for the year 2000, the mode is only fitted excluding "present" 
and "near-future" data (but including some distant future data to not exclude too much
for fitting).
Save model parameters w/ dimensions (lat/lon/(init)time) for both wsp100 and wep
"""

import xarray as xr
from pathlib import Path
from climate_scenarios_2050.utils import make_season_group,load_model_ensemble_NAO
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from climate_scenarios_2050.config import *

# def detrend_dim(da, dim, deg=1):
#     # detrend along a single dimension
#     p = da.polyfit(dim=dim, deg=deg)
#     fit = xr.polyval(da[dim], p.polyfit_coefficients)
#     return fit, da - fit

# def detrend(da, dims, deg=1):
#     # detrend along multiple dimensions
#     # only valid for linear detrending (deg=1)
#     da_detrended = da
#     for dim in dims:
#         trend, da_detrended = detrend_dim(da_detrended, dim, deg=deg)
#     return trend, da_detrended

predictand_name = "energy"
forcing = "ssp245"
Nyr = 8

# load (hourly) predictand dataset:
if predictand_name == "wsp100":
    prdctnd_monthly = xr.open_dataset("/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/wsp100_GWAscaled_1940-2024_.nc")
elif predictand_name == "energy": # note that this is just taking one specific turbine and hub height!
    prdctnd_monthly = xr.open_dataset("/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/N175-120m_dry-dens_1940-2024.nc")

prdctnd_monthly.attrs["history"] = ""
prdctnd_monthly = prdctnd_monthly.sel(time=slice("1940","2024"))

# season = make_season_group(prdctnd_monthly,months_in_season=[1,2,3,4,5,6,7,8,9,10,11,12])
# wsp_annual = prdctnd_monthly.groupby(season).sum()
wsp_annual = prdctnd_monthly.groupby(prdctnd_monthly.time.dt.year).mean()
wsp_annual = wsp_annual.assign_coords(year=wsp_annual.year-1) # subtract one year because otherwise the following winter NAO is used to predict that year's wind energy!

# detrend & filter wind/prod
# wsp_trend, wsp_seasonal_dt = detrend(wsp_seasonal[predictand_name],dims=["year"])
# wsp_seasonal_dt = wsp_seasonal[predictand_name]

# Observations (DJFM(!) NAO), one value per year:
obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-NAO-variability_f-None.nc")
index_forced_var_obs = xr.open_dataarray(obs_ind_dir/"forced-NAO-variability_f-None.nc")

# In NAO, year: December (year) + Jan, Feb, Mar (year+1)
nao_full = index_int_var_obs + index_forced_var_obs # reconstruct full NAO signal
# subselect to match the predictand period
# nao = nao_full.squeeze()
# detrend NAO
# nao_trnd, nao_dt_unfilt = detrend(nao,dims=["year"])
# nao_dt_unfilt = nao

# to remove a forced response regress the wsp onto the  NAO forced signal:
# note that this only removes the forced repsonse that is NAO-driven
NAO_model_ensemble = load_model_ensemble_NAO(None,forcing,ref_ds="ERA5")["scores"] # choose None for subsampling to retain all ensemble members/models
NAO_MMEM = NAO_model_ensemble.mean("member")
NAO_MMEM.to_netcdf(f"/datalake/NS9873K/DATA/CMIP6/wind_db/NAO_MMEM_{forcing}.nc")
# center both NAO and wsp:
NAO_MMEM_centered_full = NAO_MMEM - NAO_MMEM.sel(year=wsp_annual.year).mean(dim="year")

# remove forced signal from wind speed for the final verification
# compute forcing and internal component here, otherwise the forced component part will be "perfect"
wsp_annual_centered_full = wsp_annual - wsp_annual.mean(dim="year")
# note that this is strictly speaking leaking future information, since beta is estimated from the full NAO times series (just like a trend!), but this is hard to avoid
beta_wsp_full = xr.cov(wsp_annual_centered_full[predictand_name],NAO_MMEM_centered_full.sel(year=wsp_annual.year),dim="year")/NAO_MMEM_centered_full.sel(year=wsp_annual.year).var(dim="year")
wsp_forced_full = beta_wsp_full * NAO_MMEM_centered_full
# remove the forced response from the full wind field to get internal variability:
wsp_annual_internal_full = wsp_annual_centered_full[predictand_name] - wsp_forced_full


# iterate over all years to make a model for each initialization
alpha, beta, sigma = [],[],[]
wsp_pred = []
for init_year in wsp_annual.year.values:
# init_year = 2000
    # print(init_year)
    # exclude "future" data:
    year_excl= [init_year - Nyr//2, init_year+Nyr]
    incl_years = [yr for yr in wsp_annual.year.values if yr not in range(*year_excl)]

    wsp_annual_subs = wsp_annual.sel(year=incl_years)
    nao_subs = nao_full.sel(year=incl_years)

    NAO_MMEM_centered = NAO_MMEM - NAO_MMEM.sel(year=incl_years).mean(dim="year")

    # compute forcing and internal component here, otherwise the forced component part will be "perfect"
    wsp_annual_centered = wsp_annual_subs - wsp_annual_subs.mean(dim="year")
    # at this point, the beta estimation is not leaking future information (different from a trend!)
    beta_wsp = xr.cov(wsp_annual_centered[predictand_name],NAO_MMEM_centered.sel(year=incl_years),dim="year")/NAO_MMEM_centered.sel(year=incl_years).var(dim="year")
    wsp_forced = beta_wsp * NAO_MMEM_centered.sel(year=incl_years)
    # remove the forced response from the full wind field to get internal variability:
    wsp_annual_internal = wsp_annual_centered[predictand_name] - wsp_forced

    # compute forcing and internal component here, otherwise the forced component part will be "perfect"
    nao_centered = nao_subs - nao_subs.mean(dim="year")
    # at this point, the beta estimation is not leaking future information (different from a trend!)
    beta_nao = xr.cov(nao_centered,NAO_MMEM_centered.sel(year=incl_years),dim="year")/NAO_MMEM_centered.sel(year=incl_years).var(dim="year")
    nao_forced = beta_nao*NAO_MMEM_centered.sel(year=incl_years)
    # remove the forced response from the full wind field to get internal variability:
    nao_internal = nao_centered - nao_forced

    # filter
    nao_roll = nao_internal.rolling(year=Nyr,center=True).mean("year").dropna("year")
    # final predictor

    # filter
    wsp_seasonal_int_roll = wsp_annual_internal.rolling(year=Nyr,center=True).mean().sel(year=nao_roll.year)
    # this is the final predictand field

    prdctnd = wsp_seasonal_int_roll #.sel(year=year_subs)
    prdctr = nao_roll #.sel(year=year_subs)

    x_mean = prdctr.mean("year")
    y_mean = prdctnd.mean("year")
    
    # the linear model predicts absolute wind from a given NAO anomaly
    slope = ((prdctr - x_mean) * (prdctnd - y_mean)).sum("year") / ((prdctr - x_mean)**2).sum("year")
    intercept = y_mean - slope * x_mean

    wsp_est = slope * prdctr + intercept
    residual = prdctnd - wsp_est
    res_std = residual.std("year")

    alpha.append(slope)
    beta.append(intercept)
    sigma.append(res_std)

    # make actual prediction:
    wsp_pred.append(slope * index_int_var_obs.rolling(year=Nyr,center=True).mean("year").sel(year=init_year) + intercept)

alpha = xr.concat(alpha,dim=wsp_annual.year)
beta = xr.concat(beta,dim=wsp_annual.year)
sigma = xr.concat(sigma,dim=wsp_annual.year)

# prediction in m/s deviation from the trend!
wsp_pred = xr.concat(wsp_pred,dim=wsp_annual.year)

pt_sel = {"latitude":65.75,"longitude":10}

wsp_target = wsp_annual_internal_full.rolling(year=Nyr,center=True).mean("year")

# add back trend (even though this is cheating a bit):
wsp_pred_full = wsp_pred + wsp_forced_full + wsp_annual.mean(dim="year")[predictand_name]
wsp_target_full = wsp_annual[predictand_name]

# correlation skill of the linear model at predicting long-term wind variability
f = plt.figure()
ax1 = f.add_subplot(211,projection=ccrs.PlateCarree())
xr.corr(wsp_target,wsp_pred,dim="year").plot(ax=ax1, transform=ccrs.PlateCarree())
ax2 = f.add_subplot(212,projection=ccrs.PlateCarree())
xr.corr(wsp_target_full,wsp_pred_full,dim="year").plot(ax=ax2, transform=ccrs.PlateCarree())

f,ax = plt.subplots()
ax.fill_between(wsp_pred.year,wsp_pred.sel(pt_sel)-sigma.sel(pt_sel),wsp_pred.sel(pt_sel)+sigma.sel(pt_sel),alpha=.5)
ax.plot(wsp_pred.year,wsp_pred.sel(pt_sel))
ax.plot(wsp_target.year,wsp_target.sel(pt_sel))
# ax.hlines(wsp_target_full.sel(pt_sel).sel(year=slice(*ref_period)).mean("year"),ref_period[0],ref_period[-1])
# ax.plot(wsp_est.year,wsp_est.sel(pt_sel)+wsp_trend.sel(pt_sel))

# normalize to reference period, standardize
ref_period = [1960,1990]
wsp_target_ref = wsp_target_full.sel(year=slice(*ref_period)).mean("year")

wsp_pred_norm = (wsp_pred_full.sel(year=ref_period[-1]) - wsp_target_ref)/wsp_target_ref
wsp_target_norm = (wsp_target_full.sel(year=ref_period[-1]) - wsp_target_ref)/wsp_target_ref

# fr,axr = plt.subplots()
# # axr.fill_between(wsp_est_rel.year,wsp_est_rel.sel(pt_sel)-res_std_rel.sel(pt_sel),wsp_est_rel.sel(pt_sel)+res_std_rel.sel(pt_sel),alpha=.5)
# # axr.plot(wsp_est_rel.year,wsp_est_rel.sel(pt_sel))
# abs_rescaled = (wsp_est - wsp_target_ref)/wsp_target_ref
# axr.fill_between(abs_rescaled.year,abs_rescaled.sel(pt_sel)-res_std_scaled.sel(pt_sel),abs_rescaled.sel(pt_sel)+res_std_scaled.sel(pt_sel),alpha=.5)
# axr.plot(abs_rescaled.year,abs_rescaled.sel(pt_sel))
# axr.plot(wsp_target_norm.year,wsp_target_norm.sel(pt_sel))
# axr.hlines(wsp_target_norm.sel(pt_sel).sel(year=slice(*ref_period)).mean("year"),ref_period[0],ref_period[-1])

# save model parameters:
model_params = xr.Dataset(
    data_vars={
        "slope":alpha,
        "intercept":beta,
        "sigma":sigma
    }
)
# this model predicts absolute but detrended (!) wind speed from detrended NAO
# it can be directly applied to a NAO hindcast (which is a prediction of the internal
# variability componente of the NAO) to get at an estimate of the internal var component of wind
# and then the forced wind signal needs to be added
model_params.to_netcdf(f"/datalake/NS9873K/DATA/CMIP6/wind_db/{predictand_name}_NAO_linear_model_params_internal_var_{forcing}.nc")