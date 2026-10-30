"""
Verify a user-defined set of wind (speed/energy) hindcasts against observations, with
deterministic (correlation, RMSE skill, R^2, directional change agreement) and probabilistic
(CRPS, Brier score for exceeding observed quantiles) metrics.

Hindcast format: all hindcasts are organized by lead time and VALID year, i.e. dims (lead_time, year, <space>),
where year follows the NAO convention (year of December, wind of calendar year Y+1 labeled Y). Either
    - an xr.Dataset with "mu" and "sigma" (Gaussian predictive distribution), or
    - an xr.DataArray with a "member" dimension (ensemble).
Deterministic metrics are computed from the forecast mean ("mu" or member mean).

The main block evaluates the NAO analogue hindcasts + linear model (`linear_model_wsp_NAO.py`),
with and without a cross-validated calibration of the ensemble mean NAO hindcast, against
    - a climatology ("no change in internal variability") reference,
    - persistence of the last few observed years (the "measure and assume no change" practice),
    - the raw (uninitialized) CMIP6 wind ensemble, and
    - the CMIP6 wind of the members selected by the NAO analogue ranking.
The CMIP6 baselines are only defined for the full variability (internal + forced).
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.special import ndtr
from pathlib import Path
from climate_scenarios_2050.config import proj_base
from climate_scenarios_2050.evaluate import prep_subs_hc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#----------------------------------------------------------------------------------#
# Hindcast re-organization
#----------------------------------------------------------------------------------#
def by_year_struct_to_by_leadtime(by_year_ds,fixed_order_dim="init_year",replc_dim="year",new_dim="lead_time",new_dim_cutoff_val=35):

    by_leadtime_ds = []
    for iyear in by_year_ds[fixed_order_dim].values:
        single_init = by_year_ds.sel({fixed_order_dim:iyear})
        single_init_shifted = single_init.sel({replc_dim:single_init[new_dim] > 0}).swap_dims({replc_dim:new_dim})
        single_init_incl = single_init_shifted.sel({new_dim:slice(new_dim_cutoff_val)})
        by_leadtime_ds.append(single_init_incl)

    return xr.concat(by_leadtime_ds,dim=fixed_order_dim)

def by_leadtime_to_by_valid_year(fc_bylt):
    """
    re-arrange a hindcast with dims (init_year, lead_time) and a valid year coordinate `year`
    to dims (lead_time, year)
    """
    fc_byvy = []
    for lt in fc_bylt.lead_time.values:
        fc_lt = fc_bylt.sel(lead_time=lt,drop=True)
        # drop initializations that do not reach this lead time:
        fc_lt = fc_lt.isel(init_year=fc_lt.year.notnull().values)
        fc_lt = fc_lt.assign_coords(year=fc_lt.year.astype(int)).swap_dims({"init_year":"year"}).drop_vars("init_year")
        fc_byvy.append(fc_lt)
    return xr.concat(fc_byvy,dim=pd.Index(fc_bylt.lead_time.values,name="lead_time"))

#----------------------------------------------------------------------------------#
# Verification targets
#----------------------------------------------------------------------------------#
def wind_targets(vf_ds,NAO_MMEM,Nyr):
    """
    split the observed wind into a forced (regressed onto the CMIP6 multi-model ensemble mean NAO)
    and an internal component and compute Nyr-running means of internal and full wind
    """
    vf_ds_ann = vf_ds.groupby(vf_ds.time.dt.year).mean()
    # label calendar year Y+1 with Y to match the DJFM NAO convention (year of December), as in `linear_model_wsp_NAO.py`:
    vf_ds_ann = vf_ds_ann.assign_coords(year=vf_ds_ann.year - 1)

    NAO_MMEM_centered = NAO_MMEM - NAO_MMEM.sel(year=vf_ds_ann.year).mean(dim="year")
    vf_ds_ann_centered = vf_ds_ann - vf_ds_ann.mean(dim="year")
    # note that this is strictly speaking leaking future information, since beta is estimated from the full NAO times series (just like a trend!)
    beta = xr.cov(vf_ds_ann_centered,NAO_MMEM_centered.sel(year=vf_ds_ann.year),dim="year")/NAO_MMEM_centered.sel(year=vf_ds_ann.year).var(dim="year")
    forced = beta * NAO_MMEM_centered

    return {
        "internal": (vf_ds_ann_centered - forced).rolling(year=Nyr,center=True).mean("year"),
        "full": vf_ds_ann.rolling(year=Nyr,center=True).mean("year"),
        "forced": forced.rolling(year=Nyr,center=True).mean("year"),
        "clim": vf_ds_ann.mean(dim="year"),
        "annual": vf_ds_ann,
    }

#----------------------------------------------------------------------------------#
# Hindcast construction
#----------------------------------------------------------------------------------#
def calibrate_nao_hindcast(nao_hc_mean,nao_obs,Nyr):
    """
    cross-validated linear calibration of the (ensemble mean) NAO hindcast against the observed
    NAO, separately for each lead time:
        NAO_obs(year) = c(lead_time) * NAO_hc(lead_time, year) + d(lead_time) + eps
    for the prediction valid in `year`, all samples whose Nyr-window overlaps with the target window
    (|year' - year| < Nyr) are excluded from the fit
    nao_hc_mean: dims (lead_time, year), Nyr-running mean
    nao_obs: dims (year), Nyr-running mean
    returns a Gaussian hindcast (Dataset with "mu" and "sigma" = residual std of the calibration)
    """
    nao_hc_mean = nao_hc_mean.transpose("lead_time","year")
    obs = nao_obs.to_series()
    years = nao_hc_mean.year.values

    mu = np.full(nao_hc_mean.shape,np.nan)
    sigma = np.full(nao_hc_mean.shape,np.nan)
    for ii,lt in enumerate(nao_hc_mean.lead_time.values):
        x = nao_hc_mean.sel(lead_time=lt).values
        y = obs.reindex(years).values
        valid = np.isfinite(x) & np.isfinite(y)
        for jj,year in enumerate(years):
            if not np.isfinite(x[jj]):
                continue
            train = valid & (np.abs(years - year) >= Nyr)
            if train.sum() < 3:
                continue
            c, d = np.polyfit(x[train],y[train],1)
            residual = y[train] - (c*x[train] + d)
            mu[ii,jj] = c*x[jj] + d
            sigma[ii,jj] = residual.std(ddof=2)

    dims = nao_hc_mean.dims
    coords = {"lead_time":nao_hc_mean.lead_time,"year":nao_hc_mean.year}
    return xr.Dataset({"mu":(dims,mu),"sigma":(dims,sigma)},coords=coords)

def linear_model_hindcast(nao_fc,model_params):
    """
    apply the linear model (parameters indexed by valid year, see `linear_model_wsp_NAO.py`) to a
    Gaussian NAO hindcast; the predictive variance combines the regression residual and the NAO uncertainty
    """
    mu = model_params["slope"] * nao_fc["mu"] + model_params["intercept"]
    sigma = np.sqrt(model_params["sigma"]**2 + model_params["slope"]**2 * nao_fc["sigma"]**2)
    return xr.Dataset({"mu":mu,"sigma":sigma})

def climatology_hindcast(target,lead_times):
    """
    "no change" reference: zero internal variability anomaly with the climatological spread of the target
    (in the full variability space this corresponds to climatology + forced signal)
    """
    clim = xr.Dataset({
        "mu":xr.zeros_like(target),
        "sigma":xr.ones_like(target)*target.std(dim="year"),
    })
    return clim.expand_dims(lead_time=lead_times)

def to_full_variability(fc,targets):
    """
    add forced signal and climatology to a hindcast of the internal variability component
    """
    if isinstance(fc,xr.Dataset):
        fc_full = fc.copy()
        fc_full["mu"] = fc["mu"] + targets["forced"] + targets["clim"]
        return fc_full
    return fc + targets["forced"] + targets["clim"]

def persistence_hindcast(annual,n_years,lead_times,target):
    """
    "measure and assume no change" reference: the mean over the `n_years` last fully observed years
    before initialization is persisted to all lead times (full variability)
    the NAO hindcast initialized in init_year uses the DJFM NAO of init_year (i.e. initialization in spring
    of init_year + 1), so the last fully observed calendar year is init_year, which is labeled init_year - 1
    spread: std of the persistence errors over all years (for each lead time)
    """
    # label refers to the last year in the window:
    past_mean = annual.rolling(year=n_years).mean("year")
    # valid year = init_year + lead_time, last observed label = init_year - 1:
    mu = xr.concat(
        [past_mean.assign_coords(year=past_mean.year + lt + 1) for lt in lead_times],
        dim=pd.Index(lead_times,name="lead_time")
    )
    sigma = (mu - target).std("year")
    return xr.Dataset({"mu":mu,"sigma":xr.ones_like(mu)*sigma})

def to_target_grid(da,target):
    """
    bilinear interpolation of a field on a (lat, lon) grid to the grid of `target`
    """
    lat_name = [dim for dim in target.dims if "lat" in dim][0]
    lon_name = [dim for dim in target.dims if "lon" in dim][0]
    da_interp = da.interp(lat=target[lat_name].values,lon=target[lon_name].values)
    if lat_name != "lat":
        da_interp = da_interp.rename({"lat":lat_name})
    if lon_name != "lon":
        da_interp = da_interp.rename({"lon":lon_name})
    return da_interp.assign_coords({lat_name:target[lat_name].values,lon_name:target[lon_name].values})

def load_cmip6_wind_ratio(zarr_path,target,ref_years,Nyr,years=None,margin=3):
    """
    annual Nyr-running mean CMIP6 wind of each member relative to the member's mean over `ref_years`
    (NAO year labels), restricted to the domain of `target` (+ margin in degrees)
    the relative anomaly makes the 10 m model wind comparable to the 100 m (GWA-scaled) observations,
    assuming a constant (power law) shear factor and a multiplicative model bias
    `years`: [first, last] year label to load (reduces the amount of data read)
    """
    wind = xr.open_zarr(zarr_path)["sfcWind"]
    if wind.lon.max() > 180:
        wind = wind.assign_coords(lon=(wind.lon + 180) % 360 - 180).sortby("lon")

    lat_name = [dim for dim in target.dims if "lat" in dim][0]
    lon_name = [dim for dim in target.dims if "lon" in dim][0]
    lat_sel = ((wind.lat >= target[lat_name].min() - margin) & (wind.lat <= target[lat_name].max() + margin)).values
    lon_sel = ((wind.lon >= target[lon_name].min() - margin) & (wind.lon <= target[lon_name].max() + margin)).values
    wind = wind.isel(lat=lat_sel,lon=lon_sel)

    if years is not None:
        # calendar year Y+1 is labeled Y:
        wind = wind.sel(time=slice(str(years[0] + 1),str(years[-1] + 1)))
    wind_ann = wind.groupby("time.year").mean("time")
    # label calendar year Y+1 with Y to match the DJFM NAO convention (year of December):
    wind_ann = wind_ann.assign_coords(year=wind_ann.year - 1)

    wind_clim = wind_ann.sel(year=ref_years).mean("year")
    return (wind_ann.rolling(year=Nyr,center=True).mean("year")/wind_clim).astype("float32").compute()

def load_member_selection(hindcast_files):
    """
    members selected by the NAO analogue ranking for each initialization (dims init_year, member)
    """
    selection, init_years = [], []
    for hc_file in hindcast_files:
        with xr.open_dataset(hc_file) as ds:
            selection.append(ds.member_id.transpose("init_year","member").values[0].astype(str))
            init_years.append(int(ds.init_year.values[0]))
    return xr.DataArray(np.stack(selection),dims=("init_year","member"),coords={"init_year":init_years})

def cmip6_ensemble_hindcast(ratio,clim_obs,lead_times):
    """
    uninitialized CMIP6 reference (all members, identical for all lead times), Gaussian approximation
    of the ensemble; the relative wind (see `load_cmip6_wind_ratio`) is rescaled with the observed climatology
    """
    stats = to_target_grid(xr.Dataset({"mu":ratio.mean("member"),"sigma":ratio.std("member")}),clim_obs)
    fc = xr.Dataset({"mu":clim_obs*stats["mu"],"sigma":clim_obs*stats["sigma"]})
    return fc.expand_dims(lead_time=lead_times)

def analogue_ensemble_hindcast(ratio,member_selection,clim_obs,lead_times):
    """
    CMIP6 wind of the members selected by the NAO analogue ranking for each initialization,
    Gaussian approximation of the ensemble, rescaled with the observed climatology
    """
    available = np.isin(member_selection.values,ratio.member.values)
    print("{:.0f}% of the selected members have wind data".format(100*available.mean()))

    mu_lt, sigma_lt = [], []
    for lt in lead_times:
        mu_y, sigma_y, valid_years = [], [], []
        for init_year in member_selection.init_year.values:
            valid_year = init_year + lt
            if valid_year not in ratio.year.values:
                continue
            mems = member_selection.sel(init_year=init_year).values
            ratio_sel = ratio.sel(year=valid_year,drop=True).sel(member=mems[np.isin(mems,ratio.member.values)])
            mu_y.append(ratio_sel.mean("member"))
            sigma_y.append(ratio_sel.std("member"))
            valid_years.append(valid_year)
        mu_lt.append(xr.concat(mu_y,dim=pd.Index(valid_years,name="year")))
        sigma_lt.append(xr.concat(sigma_y,dim=pd.Index(valid_years,name="year")))

    stats = xr.Dataset({
        "mu":xr.concat(mu_lt,dim=pd.Index(lead_times,name="lead_time")),
        "sigma":xr.concat(sigma_lt,dim=pd.Index(lead_times,name="lead_time")),
    })
    stats = to_target_grid(stats,clim_obs)
    return xr.Dataset({"mu":clim_obs*stats["mu"],"sigma":clim_obs*stats["sigma"]})

#----------------------------------------------------------------------------------#
# Metrics
#----------------------------------------------------------------------------------#
def forecast_mean(fc):
    if isinstance(fc,xr.Dataset):
        return fc["mu"]
    return fc.mean("member")

def deterministic_scores(fc_mean,obs,dim="year"):
    """
    correlation, RMSE, RMSE skill (1 - RMSE/std(obs)), R^2 and directional change agreement,
    all computed over the common valid samples of forecast and observation
    """
    fc_mean, obs = xr.align(fc_mean,obs,join="inner")
    valid = fc_mean.notnull() & obs.notnull()
    fc_mean = fc_mean.where(valid); obs = obs.where(valid)

    err = fc_mean - obs
    rmse = (err**2).mean(dim)**.5
    ss_res = (err**2).sum(dim)
    ss_tot = ((obs - obs.mean(dim))**2).sum(dim)

    # verify directional change (only where both differences are defined):
    d_fc = fc_mean.diff(dim); d_obs = obs.diff(dim)
    sgn_agree = (np.sign(d_fc) == np.sign(d_obs)).where(d_fc.notnull() & d_obs.notnull()).mean(dim)

    n_samples = valid.sum(dim)
    return xr.Dataset({
        "corr":xr.corr(fc_mean,obs,dim=dim),
        "rmse":rmse.where(n_samples > 0),
        "nrmse":1 - rmse/obs.std(dim),
        "rsq":(1 - ss_res/ss_tot).where(n_samples > 0),
        "sgn_agree":sgn_agree,
        "n_samples":n_samples,
    })

def crps_gaussian(mu,sigma,obs):
    z = (obs - mu)/sigma
    return sigma * (z*(2*ndtr(z) - 1) + 2*np.exp(-z**2/2)/np.sqrt(2*np.pi) - 1/np.sqrt(np.pi))

def _crps_ensemble(fc,obs):
    # CRPS = E|X - y| - 1/2 E|X - X'|, the second term from the sorted ensemble
    M = fc.shape[-1]
    term1 = np.mean(np.abs(fc - obs[...,None]),axis=-1)
    weights = 2*np.arange(1,M+1) - M - 1
    term2 = np.sum(weights*np.sort(fc,axis=-1),axis=-1)/M**2
    return term1 - term2

def crps_ensemble(fc,obs):
    return xr.apply_ufunc(_crps_ensemble,fc,obs,input_core_dims=[["member"],[]],dask="allowed")

def probabilistic_scores(fc,obs,thresholds,dim="year"):
    """
    CRPS and Brier (skill) score for exceeding `thresholds` (with a "quantile" dimension);
    the Brier skill score is relative to the climatological exceedance probability 1 - quantile
    """
    if isinstance(fc,xr.Dataset):
        mu, sigma, obs = xr.align(fc["mu"],fc["sigma"],obs,join="inner")
        valid = mu.notnull() & (sigma > 0) & obs.notnull()
        crps = crps_gaussian(mu,sigma,obs)
        p_exceed = 1 - ndtr((thresholds - mu)/sigma)
    elif "member" in fc.dims:
        fc, obs = xr.align(fc,obs,join="inner")
        valid = fc.notnull().all("member") & obs.notnull()
        crps = crps_ensemble(fc,obs)
        p_exceed = (fc > thresholds).mean("member")
    else:
        raise ValueError("probabilistic verification needs a Gaussian (mu/sigma) or ensemble (member) hindcast")

    obs_exceed = (obs > thresholds)
    bs = ((p_exceed - obs_exceed)**2).where(valid).mean(dim)
    bs_clim = (((1 - thresholds["quantile"]) - obs_exceed)**2).where(valid).mean(dim)

    return xr.Dataset({
        "crps":crps.where(valid).mean(dim),
        "bs":bs,
        "bss":1 - bs/bs_clim,
    })

def evaluate(hindcasts,obs,lead_times,quantiles=[.5,.9],reference=None):
    """
    apply all metrics to a dict of hindcasts {name: hindcast} for the selected lead times
    `reference`: name of the hindcast used as reference for the CRPS skill score
    event thresholds are the observed quantiles over all available years
    """
    thresholds = obs.quantile(quantiles,dim="year")

    scores = []
    for name,fc in hindcasts.items():
        scores_lt = []
        for lt in lead_times:
            fc_lt = fc.sel(lead_time=lt,drop=True)
            scores_lt.append(xr.merge([
                deterministic_scores(forecast_mean(fc_lt),obs),
                probabilistic_scores(fc_lt,obs,thresholds),
            ]))
        scores.append(xr.concat(scores_lt,dim=pd.Index(lead_times,name="lead_time")))
    scores = xr.concat(scores,dim=pd.Index(list(hindcasts),name="hindcast"))

    if reference is not None:
        scores["crpss"] = 1 - scores["crps"]/scores["crps"].sel(hindcast=reference)

    return scores

def plot_score_maps(scores,metric,lead_time,vmin,vmax,title=None,sel={}):
    names = scores.hindcast.values
    f,ax = plt.subplots(len(names),1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(5,2.7*len(names)),squeeze=False)
    for ii,name in enumerate(names):
        scores[metric].sel(hindcast=name,lead_time=lead_time,**sel).plot(ax=ax[ii,0],vmin=vmin,vmax=vmax,cmap="RdBu_r")
        ax[ii,0].coastlines()
        ax[ii,0].set_title(name)
    f.suptitle(title if title is not None else f"{metric} (lead time {lead_time})")
    return f

if __name__ == "__main__":

    database = Path("/datalake/NS9873K/DATA/CMIP6/wind_db/")

    scenario = "ssp245" # one of ["ssp245", "ssp370"]
    variable = "wsp100" # must be in ["wsp100", "energy"]
    Nyr = 8 # running average window [years], must match `linear_model_wsp_NAO.py`
    # choose parameter set for the analog model:
    hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : scenario, 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
    lead_times_select = [5,10,15,20,25,30,35]
    persistence_years = [2,8] # length of the observational record used for the "no change" reference(s)

    # VERIFICATION DATA:
    if variable == "wsp100":
        vf_ds = xr.open_dataarray(f"/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/{variable}_GWAscaled_1940-2024.nc")
    elif variable == "energy":
        vf_ds = xr.open_dataset(f"/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/monthly_aggregation/N175-120m_dry-dens_1940-2024.nc").energy

    NAO_MMEM = xr.open_dataarray(database/f"NAO_MMEM_{scenario}.nc")
    targets = wind_targets(vf_ds,NAO_MMEM,Nyr)

    # LINEAR MODEL PARAMETERS (indexed by valid year):
    model_params = xr.open_dataset(database/f"{variable}_NAO_linear_model_params_internal_var_{scenario}_by-valid-year.nc")

    # NAO HINDCASTS:
    hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
    hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
    ds_hc = xr.open_mfdataset(hindcast_files,preprocess=prep_subs_hc)
    ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
    nao_hc = by_leadtime_to_by_valid_year(by_year_struct_to_by_leadtime(ds_hc_proc))
    # Gaussian NAO hindcast from the analogue ensemble (mean and spread):
    nao_hc_ens = xr.Dataset({"mu":nao_hc.mean("member"),"sigma":nao_hc.std("member")})

    # OBSERVED NAO (internal variability, the linear model predictor) for the calibration:
    nao_obs = xr.open_dataarray(Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")/"internal-NAO-variability_f-None.nc").squeeze(drop=True)
    nao_obs_roll = nao_obs.rolling(year=Nyr,center=True).mean("year")
    nao_hc_cal = calibrate_nao_hindcast(nao_hc_ens["mu"],nao_obs_roll,Nyr)

    # SET OF HINDCASTS (internal variability component):
    hindcasts_int = {
        "analogue": linear_model_hindcast(nao_hc_ens,model_params),
        "analogue_calibrated": linear_model_hindcast(nao_hc_cal,model_params),
        "climatology": climatology_hindcast(targets["internal"],lead_times_select),
    }
    hindcasts_full = {name:to_full_variability(fc,targets) for name,fc in hindcasts_int.items()}

    # BASELINES (full variability only):
    for n_years in persistence_years:
        hindcasts_full[f"persistence_{n_years}y"] = persistence_hindcast(targets["annual"],n_years,lead_times_select,targets["full"])
    if variable == "wsp100": # the relative wind rescaling is not valid for energy (non-linear power curve)
        years_load = [int(targets["annual"].year.min()) - Nyr, int(targets["annual"].year.max()) + Nyr]
        wind_ratio = load_cmip6_wind_ratio(database/f"sfcWind_Amon_{scenario}_multimodel.zarr",targets["clim"],targets["annual"].year.values,Nyr,years=years_load)
        hindcasts_full["cmip6_all"] = cmip6_ensemble_hindcast(wind_ratio,targets["clim"],lead_times_select)
        hindcasts_full["cmip6_analogue"] = analogue_ensemble_hindcast(wind_ratio,load_member_selection(hindcast_files),targets["clim"],lead_times_select)

    reference_full = f"persistence_{persistence_years[0]}y"
    scores_int = evaluate(hindcasts_int,targets["internal"],lead_times_select,reference="climatology").compute()
    scores_full = evaluate(hindcasts_full,targets["full"],lead_times_select,reference=reference_full).compute()

    # PLOT SKILL FOR SINGLE LEAD TIME
    LT = 15
    plot_score_maps(scores_int,"corr",LT,-1,1,title=f"correlation, internal (lead time {LT})")
    plot_score_maps(scores_int,"nrmse",LT,-.3,.3,title=f"RMSE skill, internal (lead time {LT})")
    plot_score_maps(scores_int,"crpss",LT,-.3,.3,title=f"CRPSS vs climatology, internal (lead time {LT})")
    plot_score_maps(scores_int,"bss",LT,-.3,.3,title=f"BSS (median), internal (lead time {LT})",sel={"quantile":.5})
    plot_score_maps(scores_int,"bss",LT,-.3,.3,title=f"BSS (90th percentile), internal (lead time {LT})",sel={"quantile":.9})
    plot_score_maps(scores_full,"crpss",LT,-.3,.3,title=f"CRPSS vs {reference_full}, full (lead time {LT})")
    plot_score_maps(scores_full,"corr",LT,-1,1,title=f"correlation, full (lead time {LT})")
