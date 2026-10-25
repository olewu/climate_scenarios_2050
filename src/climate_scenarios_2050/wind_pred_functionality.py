"""
Computational routines for the dashboard.
Needs access to NAO files in `database`:
 - internal-NAO-variability-HadSLP-ERA5-merged-1980.nc
 - internal-NAO-variability_hist-ssp245.nc
 - internal-NAO-variability_hist-ssp370.nc
 - wsp100_NAO_linear_model_params.nc
(all those files should be on Teams under /Documents/General/OsloWorkshop_April-2026/Tool/db_data/)
and to the CMIP6 data via the cmip6-api.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import re
# import os
# from scipy.stats import gaussian_kde, norm

# set the path to where you saved the (NAO) data:
database = Path("/datalake/NS9873K/DATA/CMIP6/wind_db/")

#---------------INPUT--------------#
site_name = "smoela"
# could link lat/lon coords to specific sites if we could get a simple look-up table that selects coords based on site name
site_dict = {
    "smoela" : {"lat":63.404939, "lon":7.865301},
    "montes_del_cierzo": {"lat":42.087595, "lon":-1.709988},
    "berry_burn": {"lat":57.472550, "lon":-3.540099},
}
# for now, select from the above three stations or put in lat/lon manually:
site_latitude = site_dict[site_name]["lat"]
site_longitude = site_dict[site_name]["lon"]

# choose a scenario:
scenario = "ssp370" # choose from ["ssp245", "ssp370"]

# choose variable
variable = "sfcWind" # choose from ["sfcWind", "tas"]

# define the reference period by start and end year:
reference_period = [1980,2000] # the end year will be the start/initialization year for the analog method

# define future periods to look at (add as many as you like):
future_periods = {
    "p1":[2000,2005],
    "p2":[2005,2010],
    "p3":[2000,2010],
    "p4":[2000,2015],
    "p5":[2010,2020],
    "p6":[2020,2040],
}

# put in wind in reference period manually or leave None
V_mean = None # measured average wind speed over reference period

# get reference wind time series at the point:
request_wind_clim_url = f"https://wind-api.dashboards.sigma2.no/timeseries?latitude={site_latitude}&longitude={site_longitude}&method=nearest&return_distance=false"

df_clim = pd.read_csv(request_wind_clim_url,sep=",")
df_clim["time"] = pd.to_datetime(df_clim["time"])
df_clim = df_clim.groupby(df_clim.time.dt.year).mean().drop(columns="time").reset_index()

# number of regression ensemble members:
M_mem = 100

# if no measurements available, get reference wind from API
if V_mean is None:
    V_mean = df_clim[(df_clim.time >= reference_period[0]) & (df_clim.time < reference_period[-1])]["wsp100"].mean()


#---------------ANALOG MODEL PARAMETERS--------------#
# these can be changed but always need to keep in mind 
# that only specific parameter values have been evaluated:
analog_params = {
    "matching_roll_years":41,
    "matching_metric":"CORR",
    "matching_window_length":70,
    "N_pred":8,
    "analog_members":100,
}

#---------------METHOD 1--------------#
# assume stationarity --> climate in future periods will be the same as 
# in reference period, i.e. 0 relative and absolute change
# nothing to implement

#---------------METHOD 2--------------#
# M2) compute change in variable between historical and future period for a given scenario:
# build the request URL from input parameters:
request_url = f"https://cmip6-api.dashboards.sigma2.no/timeseries?latitude={site_latitude}&longitude={site_longitude}&variable={variable}&scenario={scenario}&method=nearest&return_distance=false"

df = pd.read_csv(request_url,sep=";")
df["time"] = pd.to_datetime(df["time"])
da = df.set_index(["member","time"]).to_xarray()
clim_ds_var_of_interest_at_point = da.assign(member=da.member.astype("U13"))[variable]
# need to generate the model_id coordinate!
clim_ds_var_of_interest_at_point = clim_ds_var_of_interest_at_point.assign_coords(model_id=("member",[re.match(r"(m\d{1,2})r",mem).group(1) for mem in clim_ds_var_of_interest_at_point.member.values]))


def compute_change_future_reference(
        future_clim_ts:xr.DataArray,
        reference_period_mean:xr.DataArray,
        future_periods:dict,
        group_dim = None,
        time_coord = 'time',
    ) -> xr.DataArray:
    """
    compute the change between reference and future periods based on the historical runs and projections/predictions
    for the variable of interest
    needed for M2 and M3, in the latter case, just pass the subset of models selected via NAO analogs
    """
       
    future_mean_by_model = []
    for future_period_tag, future_period in future_periods.items():
        if group_dim is not None:
            future_mean_by_model.append(
                future_clim_ts.sel(
                    {time_coord:slice(str(future_period[0]),str(future_period[-1]))}
                ).mean(time_coord,skipna=True).groupby(group_dim).mean(skipna=True).expand_dims(dim={"period_id":[future_period_tag]})
            )
        else:
            future_mean_by_model.append(
                future_clim_ts.sel(
                    {time_coord:slice(str(future_period[0]),str(future_period[-1]))}
                ).mean(time_coord,skipna=True).expand_dims(dim={"period_id":[future_period_tag]})
            )

    future_mean_by_model = xr.concat(future_mean_by_model,dim="period_id")

    if isinstance(reference_period_mean,xr.DataArray):
        if ("model_id" in reference_period_mean.dims):
            delta_change_abs = future_mean_by_model.groupby("model_id") - reference_period_mean # change in units of variable
            delta_change_rel = delta_change_abs.groupby("model_id")/reference_period_mean # change relative to reference period mean
    else:
        delta_change_abs = future_mean_by_model - reference_period_mean # change in units of variable
        delta_change_rel = delta_change_abs/reference_period_mean # change relative to reference period mean

    return delta_change_rel, delta_change_abs

# compute each model's own reference climate
reference_mean_by_model = clim_ds_var_of_interest_at_point.sel(time=slice(str(reference_period[0]),str(reference_period[-1]-1))).mean("time",skipna=True).groupby("model_id").mean(skipna=True)

# compute relative (fraction) and absolute (in units of variable) 
# change between reference and selected future periods
delta_rel_m2, delta_abs_m2 = compute_change_future_reference(clim_ds_var_of_interest_at_point,reference_mean_by_model,future_periods,group_dim=None)

# scale the relative change from the model with the "measured" climatology
# to get the change in absolute terms
if (V_mean is not None) and (variable == "sfcWind"):
    delta_abs_m2 = V_mean * delta_rel_m2

#---------------METHOD 3--------------#
#------------------------------------------------------------------------------------------------#
# M3a) take NAO analog members but use 'variable' directly:
# functions for the analog routine:
def get_ranking(obs,model,init_year,analog_params):
    """
    Derives the ensemble member ranking (best to worst) based on a measure of
    goodness-of-fit (currently correlation or RMSE) between an observed
    time series `obs` and a number of model simulations `model`.
    The measure, as well as other parameters of the analog model
    need to be passed in the dict `analog_params`.
    The `init_year` variable defines after which year observational
    data in `obs` is cut off (pretending to make a forecast with 
    only the information that would have been available in `init_year`).
    """
    # Cut the observations off at init_year and compute rolling mean:
    obs_roll = obs.sel(year=slice(init_year)).rolling(year=analog_params["matching_roll_years"],center=True,min_periods=analog_params["matching_roll_years"]//4).mean(skipna=True)
    
    # compute rolling mean of model simulations:
    model_roll = model.rolling(year=analog_params["matching_roll_years"],center=True,min_periods=analog_params["matching_roll_years"]//4).mean(skipna=True)

    # define the matching window:
    match_period = [str(init_year-analog_params["matching_window_length"]-1),str(init_year)]

    # subselect data to matching window:
    obs_match = obs_roll.sel(year=slice(*match_period))
    model_match = model_roll.sel(year=slice(*match_period))

    # center data over matching window:
    obs_match_center = obs_match - obs_match.mean()
    # center each model separately:
    model_match_centr = model_match.groupby("model_id") - model_match.groupby('model_id').mean("member")

    # compute ranking metric/score over matching window:
    if analog_params["matching_metric"] == "RMSE":
        score = ((obs_match_center - model_match_centr)**2).mean(dim="year",skipna=True)**.5
    elif analog_params["matching_metric"] == "CORR":
        score = xr.corr(obs_match_center, model_match_centr,dim="year")
    
    # sort members by how well they match, exclude NaN values:
    if analog_params["matching_metric"] in ["RMSE"]:
        ensemble_sorted = model.sel(member=~score.isnull()).sortby(score.sel(member=~score.isnull()))
    elif analog_params["matching_metric"] in ["CORR"]:
        ensemble_sorted = model.sel(member=~score.isnull()).sortby(score.sel(member=~score.isnull()),ascending=False)
    # !sorted from best to worst!

    # return both the unsorted (!) scores and the ranking (ensemnble member ids from best to worst)
    return score.sel(member=~score.isnull()),ensemble_sorted.member

def subselect_ensemble(full_ensemble,ens_ranking,M_analog):
    """
    function to subselect the `full_ensemble` to `M_analog` first members
    based non the `ens_ranking`. 
    """
    ens_ranking_subs = list(set(full_ensemble.member.values) & set(ens_ranking.isel(member=slice(M_analog)).values))
    return full_ensemble.sel(member=ens_ranking_subs)

# load observed and simulated NAO internal variability:
obs_to_match = xr.open_dataarray(database/"internal-NAO-variability-HadSLP-ERA5-merged-1980.nc")
model_to_match = xr.open_dataarray(database/f"internal-NAO-variability_hist-{scenario}.nc").rename({"m_id":"model_id"})

# derive the ranking
_,ens_ranking = get_ranking(obs_to_match,model_to_match,reference_period[-1],analog_params)

# subselect the ensemble of `variable` to the M best (NAO) members:
analog_forecast_var_of_interest_at_point = subselect_ensemble(clim_ds_var_of_interest_at_point,ens_ranking,analog_params["analog_members"])

# change relative to SINGLE MODEL FULL ENSEMBLE MEAN
delta_rel_m3a, delta_abs_m3a = compute_change_future_reference(analog_forecast_var_of_interest_at_point,reference_mean_by_model,future_periods,group_dim=None)

# scale change to reference:
if (V_mean is not None) and (variable == "sfcWind"):
    delta_abs_m3a = delta_rel_m3a * V_mean

#-------------------------------------------------------------------------------------#
# M3b) take NAO analog members and compute wind from linear model wsp100 = alpha * NAO + beta + eps(0,sigma):
# note that the staistical model has only been build for wind, so this 
# does not work for any other variables at the moment!
def get_forced_signal(var_obs,scenario):
    """
    compute the forced signal for each model by scaling the
    multi-model ensemble mean (mmem) to the variability of the 
    single-model ensemble means (smem) assuming that the 
    mmem is the best available representation of the forced signal so that
    each single realization (sm) can be split into:
    sm = b * mmem + n
    where b is the scaling to be estimated and n is remaining unforced/internal variability in the smem
    b can be computed as b = cov(sm,mmem)/var(mmem)
    """
    # Load NAO MMEM:
    nao_mmem = xr.open_dataarray(database/f"NAO_MMEM_{scenario}.nc")
    # center:
    nao_mmem_centered = nao_mmem - nao_mmem.sel(year=var_obs["year"]).mean(dim="year")

    # center wind speed:
    var_obs_centered = var_obs - var_obs.mean()
    # linearly regress forced NAO signal onto wind speed:
    beta_wsp = xr.cov(var_obs_centered["wsp100"],nao_mmem_centered.sel(year=var_obs["year"]),dim="year")/nao_mmem_centered.sel(year=var_obs["year"]).var(dim="year")
    # forced signal is the regression parameter x the mmem (+ average)
    forced_signal = beta_wsp * nao_mmem_centered + var_obs["wsp100"].mean()

    return forced_signal

if variable == "sfcWind":
    # subselect the ensemble of NAO to the M best (NAO) members:
    analog_model_NAO = subselect_ensemble(model_to_match,ens_ranking,analog_params["analog_members"])

    # need to take rolling NAO mean (this is what the linear model expects as NAO input):
    analog_model_NAO_rm = analog_model_NAO.rolling(year=analog_params["N_pred"],center=True).mean()

    # load model parameters alpha, beta, sigma for the linear model at point of interest:
    model_params = xr.open_mfdataset(database/f"wsp100_NAO_linear_model_params_internal_var_{scenario}.nc").sel(year=reference_period[-1]).interp(latitude=site_latitude,longitude=site_longitude,method="nearest")

    # apply linear model to get wind delta in the requested future periods (change due to *internal* NAO variability alone)
    # this is expressed as absolute 8-year mean deviations (in m/s) from the overall trend
    wind_pred_int = model_params["slope"] * analog_model_NAO_rm + model_params["intercept"]
    wind_pred_int = wind_pred_int.load()

    # generate more ensemble members by sampling from a normal distribution with sigma around each model's estimate:
    rng = np.random.default_rng(seed=42)  # preferred modern API
    samples = rng.normal(loc=0.0, scale=model_params["sigma"].values, size=(len(wind_pred_int.member),len(wind_pred_int.year),M_mem)) 
    rand_regression_pt = xr.DataArray(
        samples,
        dims = {"member":wind_pred_int.member,"year":wind_pred_int.year,"sample":np.arange(M_mem)},
    )

    # compute forced signal:
    wind_forced_signal = get_forced_signal(df_clim.set_index("time").to_xarray().rename({"time":"year"}),scenario)
    wind_forced_signal = wind_forced_signal.load()

    # forced signal is absolute wind, just add the absolute deviation from the linear model
    # also add the random perturbations based on the residual variance of the linear model:
    wind_pred_full = wind_pred_int + wind_forced_signal + rand_regression_pt

    delta_rel_m3b, delta_abs_m3b = compute_change_future_reference(wind_pred_full,V_mean,future_periods,group_dim=None,time_coord="year")

    # scale change to reference:
    if (V_mean is not None) and (variable == "sfcWind"):
        delta_abs_m3b = delta_rel_m3b * V_mean

#---------------METHOD 4--------------#
# M4) make wind (instead of NAO) analogs
# first, get the reference observations fromt the wind api
request_all_wind_url = f"https://wind-api.dashboards.sigma2.no/timeseries?latitude={site_latitude}&longitude={site_longitude}&method=nearest&return_distance=false"

wind_obs = pd.read_csv(request_all_wind_url,sep=",")
wind_obs["time"] = pd.to_datetime(wind_obs["time"])
wind_obs_year = wind_obs.groupby(wind_obs.time.dt.year).mean().drop(columns=["time"])
wind_obs_to_match = wind_obs_year.to_xarray().rename({"time":"year"})["wsp100"]

wind_sim_to_match = clim_ds_var_of_interest_at_point.groupby(clim_ds_var_of_interest_at_point.time.dt.year).mean()

# get the ranking. Note that this is comparing sfcWind (models) with 100 m wind speed (ERA5)!!
_,ens_ranking_wind = get_ranking(wind_obs_to_match,wind_sim_to_match,reference_period[-1],analog_params)

# subselect the ensemble of `variable` to the M best (NAO) members:
wind_analog_forecast_var_of_interest_at_point = subselect_ensemble(clim_ds_var_of_interest_at_point,ens_ranking_wind,analog_params["analog_members"])

# change relative to SINGLE MODEL FULL ENSEMBLE MEAN
delta_rel_m4, delta_abs_m4 = compute_change_future_reference(wind_analog_forecast_var_of_interest_at_point,reference_mean_by_model,future_periods,group_dim=None)

# scale change to reference:
if (V_mean is not None) and (variable == "sfcWind"):
    delta_abs_m4 = delta_rel_m4 * V_mean

#---------------PLOT RESULTS--------------#
def make_vio_plot(a,b,c,d,verification,xticklabels,ylabl="",label_a="",label_b="",label_c="",label_d="",title=""):
    """
    makes a violin plot comparing three different disitribtions for
    an arbitrary number of different periods 
    """
    # Remove NaNs row-wise
    a_clean = [row[~np.isnan(row)] for row in a]
    a_num = (~np.isnan(a[0])).sum()
    b_clean = [row[~np.isnan(row)] for row in b]
    b_num = (~np.isnan(b[0])).sum()
    c_clean = [row[~np.isnan(row)] for row in c]
    c_num = (~np.isnan(c[0])).sum()
    d_clean = [row[~np.isnan(row)] for row in d]
    d_num = (~np.isnan(d[0])).sum()

    # X positions
    x = np.arange(len(a))
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Slight offsets so violins are side-by-side
    offset = 0.225

    vp1 = ax.violinplot(
        a_clean,
        positions=x - 1.5*offset,
        widths=0.175,
        showmeans=False,
        showmedians=True
    )

    vp2 = ax.violinplot(
        b_clean,
        positions=x - .5*offset,
        widths=0.175,
        showmeans=False,
        showmedians=True
    )

    vp3 = ax.violinplot(
        c_clean,
        positions=x + .5*offset,
        widths=0.175,
        showmeans=False,
        showmedians=True
    )

    vp4 = ax.violinplot(
        d_clean,
        positions=x + 1.5*offset,
        widths=0.175,
        showmeans=False,
        showmedians=True
    )

    # Coloring
    for body in vp1['bodies']:
        body.set_facecolor('C0')
        body.set_alpha(0.6)

    for body in vp2['bodies']:
        body.set_facecolor('C1')
        body.set_alpha(0.6)

    for body in vp3['bodies']:
        body.set_facecolor('C2')
        body.set_alpha(0.6)

    for body in vp4['bodies']:
        body.set_facecolor('C3')
        body.set_alpha(0.6)


    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel('Period')
    ax.set_ylabel(ylabl)

    if verification is not None:
        for ii,(_,vf_val) in enumerate(verification.items()):
            ax.hlines(vf_val, x[ii] - 2*offset, x[ii] + 2*offset,color='k',ls="dashed")

    # Manual legend
    ax.plot([], [], color='C0', label=label_a+f" ({a_num})")
    ax.plot([], [], color='C1', label=label_b+f" ({b_num})")
    ax.plot([], [], color='C2', label=label_c+f" ({c_num})")
    ax.plot([], [], color='C3', label=label_d+f" ({d_num})")
    ax.legend(ncol=4)
    ax.grid()
    ax.set_title(" ".join([substr.capitalize() for substr in title.split("_")]))

    plt.tight_layout()

    return fig

# As long as the "future" periods lie in the actual past,
# we can compute what change was actually observed in those periods
# relative to the reference period (use wind-api data):
# actual future change by period:
# TODO: avoid computing averages over prtially available periods!!
V_future = {nm:df_clim[(df_clim.time >= p[0]) & (df_clim.time < p[-1])]["wsp100"].mean() if p[-1] <= pd.Timestamp("today").year else np.nan for nm,p in future_periods.items()}
V_future_abs = {nm:(v-V_mean) for nm,v in V_future.items()}
V_future_rel_prcnt = {nm:(v-V_mean)/V_mean*100 for nm,v in V_future.items()}

f = make_vio_plot(
    delta_rel_m2.values*100,delta_rel_m3a.values*100,delta_rel_m3b.values*100,delta_rel_m4.values*100,
    V_future_rel_prcnt,
    ["{:} - {:}".format(*intrv) for _,intrv in future_periods.items()],
    ylabl=(f"{variable} change [%]"),
    label_a="M2",label_b="M3a",label_c="M3b",label_d="M4",
    title=site_name
)

f.savefig("violin_plot.png",dpi=300,bbox_inches="tight")
