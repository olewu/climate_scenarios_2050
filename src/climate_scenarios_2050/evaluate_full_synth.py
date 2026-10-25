# Do the full evaluation of NAO analogue hindcasts
# this includes:
# 1) compute analogue hindcast skill wrt to observed NAO
# 2) AR(1) hindcast skill wrt to observed NAO --> does the analogue outperform a simple AR(1) baseline?
# 3) analogue hindcast skill wrt to synthetic AR(1) NAO time series --> is the analogue just picking up persistence?
# all of these are generated prior 

import xarray as xr

from climate_scenarios_2050.evaluate import moving_block_bootstrap_pairs, null_dist_phase_randomization, bootstrap_ci_from_bootrs, bootstrap_pvalue_null, prep_subs_hc
from climate_scenarios_2050.config import *
from climate_scenarios_2050.filters import butter_lowpass_filter

import pandas as pd
import numpy as np
from scipy import stats
import proplot as pplt
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

import properscoring as ps


def by_year_struct_to_by_leadtime(by_year_ds,fixed_order_dim="init_year",replc_dim="year",new_dim="lead_time",new_dim_cutoff_val=35):

    by_leadtime_ds = []
    for iyear in by_year_ds[fixed_order_dim].values:
        single_init = by_year_ds.sel({fixed_order_dim:iyear})
        single_init_shifted = single_init.sel({replc_dim:single_init[new_dim] > 0}).swap_dims({replc_dim:new_dim})
        single_init_incl = single_init_shifted.sel({new_dim:slice(new_dim_cutoff_val)})
        by_leadtime_ds.append(single_init_incl)

    return xr.concat(by_leadtime_ds,dim=fixed_order_dim)

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# hc_param_set = {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'varsplit_20y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}
# {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
# {'subsampling' : "varsplit_7y", 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
# {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},

# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 8

# lead_times: what lead times to verify
lead_times = np.arange(0,34,1)
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
# index_forced_var_model = xr.open_dataarray(CMIP6_base/"forced/forced-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))
# # full variability:
# index_int_var_model = index_int_var_model.groupby("m_id") + index_forced_var_model

# process (average/filter):
if proc_type == 'running_mean':
    index_int_var_model_proc = index_int_var_model.rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    index_int_var_model_proc = xr.apply_ufunc(
        butter_lowpass_filter,index_int_var_model,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

index_int_var_model_proc_std = index_int_var_model_proc

# expand the array to look like a hindcast:
index_int_var_model_proc = index_int_var_model_proc.expand_dims(dim={'init_year':len(index_int_var_model_proc.year)}).assign_coords({'init_year':('init_year',index_int_var_model_proc.year.values)})
index_int_var_model_proc = index_int_var_model_proc.assign_coords(lead_time=index_int_var_model_proc.year - index_int_var_model_proc.init_year)

# Observations:
if hc_param_set["ref_ds"] == "ERA5":
    obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
else:
    obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))
# index_forced_var_obs = xr.open_dataarray(obs_ind_dir/"forced-{1}-variability_{0}.nc".format(filter_ext,index_name))
# index_int_var_obs += index_forced_var_obs

# process:
if proc_type == 'running_mean':
    index_int_var_obs_proc = index_int_var_obs.rolling(year=Nyr).mean('year').dropna('year')
elif proc_type == 'lowpass':
    index_int_var_obs_proc = xr.apply_ufunc(
        butter_lowpass_filter,index_int_var_obs,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )

# print(xr.corr(index_int_var_model_proc.isel(init_year=0).mean('member'),index_int_var_obs_proc,dim='year').values)


#----------------------------------------------------------------------------------#
# CMIP6 "initialized"/"hindcasts"
#----------------------------------------------------------------------------------#
# create the path the hindcast are saved to:
hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)

# load all hindcasts:
hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
ds_hc = xr.open_mfdataset(hindcast_files,preprocess=prep_subs_hc)

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc['scores'],1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# re-arrange the hindcast array to have lead time as dimension instead of valid year
ds_hc_proc_bylt = by_year_struct_to_by_leadtime(ds_hc_proc)

# save NAO hindcast dataset for sharing:
# ds_hc_proc_bylt.to_netcdf("/projects/NS9873K/www/scenarios_2050/wind_power_production_dataset/NAO_HadSLP_analog_hindcast_anomalies.nc")

# design a climatological prediction as reference:
ds_modelclim_bylt = by_year_struct_to_by_leadtime(index_int_var_model_proc)


#----------------------------------------------------------------------------------#
# get the AR(1) hindcasts
#----------------------------------------------------------------------------------#
ds_ar1_hc = xr.open_dataset(proj_base/"data/AR1_hindcast/AR1_HadSLP_100member_NAO_hindcast.nc")
# process (average/filter):
if proc_type == 'running_mean':
    ds_ar1_hc_proc = ds_ar1_hc[['forecast','forecast_full','verification']].rolling(lead_time=Nyr,center=True).mean('lead_time').dropna('lead_time',how='all')
elif proc_type == 'lowpass':
    ds_ar1_hc_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_ar1_hc[['forecast','forecast_full','verification']],1/7,1,
        input_core_dims=[['lead_time'],[],[]],output_core_dims=[['lead_time']],
        dask='allowed',vectorize=True,
    )

hc_coll = xr.merge([ds_ar1_hc_proc.rename({"valid_year":"year"}),ds_hc_proc_bylt.transpose("init_year","lead_time","member").to_dataset(name="analogue_forecast")])

# add verification to the climatological forecast
ds_modelclim_bylt = ds_modelclim_bylt.transpose("init_year","lead_time","member").to_dataset(name="climatological_forecast").assign({"verification":hc_coll.verification})
# restrict "predictions" to where there other forecasts exist, too
ds_modelclim_bylt = ds_modelclim_bylt.sel(init_year=hc_coll.init_year)

# plot some hindcasts to get a sense of what the prediction looks like:
fig_lt = pplt.figure(width=5,height=2,suptitle='NAO prediction')
ax = fig_lt.add_subplot()
for lt in [5,15,25]:
    if (hc_param_set["ref_ds"] == "ERA5") and (lt > 15):
        continue

    # group hindcasts by lead time:
    lt_group = hc_coll.sel(lead_time=lt).mean("member")["analogue_forecast"]
    lt_group_scaled = lt_group * hc_coll.sel(lead_time=lt).verification.std()/lt_group.sel(init_year=slice(1912,2019-lt)).std()
    # Plot some of the lead times:
    ax.plot(lt_group_scaled.year,lt_group,ls=linestyle[abs(lt)],color='k',label='',lw=.5)
    ax.plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],color='k',label='{:} yr prediction'.format(abs(lt)))
    print(f"LT:{lt}Y")
    print(xr.corr(index_int_var_obs_proc, lt_group.swap_dims({"init_year":"year"})).values)
    print(xr.corr(index_int_var_obs_proc, lt_group_scaled.swap_dims({"init_year":"year"})).values)
    
ax.plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label='HadSLP')
ax.format(xlim=[hc_coll.year.min(),hc_coll.year.max().values],ylabel='NAO index')
ax.legend(loc='t',prop={'size':8},ncol=4)

# fig_lt.savefig(fpath/"hc_cmip6-{scen}_fixedlead_{subsampling}_sync_{ref_ds}_subs41y_{vf_win_len}yvf.png".format(**hc_param_set),dpi=300,bbox_inches="tight")

# hindcast_em_plot()
    # # cool plot "visual verification" of ensemble mean:
    # array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
    #     [1, ],
    #     [1, ],
    #     [1, ],
    #     [2, ],
    # ]

    # one_init_year = 1950

    # fig,axs = pplt.subplots(array,figwidth=5,figheight=4.8,suptitle='NAO ensemble mean hindcast')
    # axs.format(abc=True)
    # # cf = axs[0].contourf(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
    # cf = axs[0].pcolormesh(ds_hc_ensmean.where(ds_hc_proc.lead_time>0),vmin=-20,vmax=20,extend='both')
    # axs[0].pcolormesh(ds_hc_ensmean.where(ds_hc_proc.lead_time<0),colors="white")
    # axs[0].set_facecolor("lightgrey")
    # # single_hc_masked = ds_hc_ensmean.where(ds_hc_proc.lead_time!=0).sel(init_year=one_init_year)
    # # single_hc = ds_hc_ensmean.sel(init_year=one_init_year)
    # # cf = axs[0].pcolormesh(single_hc_masked.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[1949.5,1950.5])}),vmin=-20,vmax=20,extend='both')

    # # cf = ax.contourf(ds_hc_ensmean.scores)
    # axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(index_int_var_obs_proc.year)).lead_time,levels=[ 5, 15,25,35],colors='k',linestyles='dotted',labels=True)
    # # axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(vf_proc.year)).lead_time,levels=[5],colors='k',linestyles='solid',labels=True)
    # axs[0].colorbar(cf, loc='r', label='')
    # # axs[0].vlines(one_init_year+np.arange(5,31,10),ds_hc_proc.init_year.min().values,1950,color='k',linestyle='dashed',linewidth=.5)
    # # axs[0].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[1949.5,1949.5],[1950.5,1950.5],alpha=.2,color='k',zorder=100)

    # axs[1].plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label="HadSLP")
    # # axs[1].plot(single_hc.year,single_hc,color='k')
    # # lt5_hc = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(5)
    # # axs[1].plot(lt5_hc.year,lt5_hc,color='k')
    # # axs[1].vlines(one_init_year+np.arange(5,26,10),-80,80,color='k',linestyle='dashed',linewidth=.5)
    # # axs[1].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[-80,-80],[80,80],alpha=.2,color='k',zorder=100)
    # # axs[1].text(one_init_year-hc_param_set['vf_win_len']//2,-75,s='synchronization window',va='bottom',ha='center',fontsize=8)
    # # axs[1].text(one_init_year+5,80,s='LT5Y',va='bottom',ha='center',fontsize=6)
    # # axs[1].text(one_init_year+15,80,s='LT15Y',va='bottom',ha='center',fontsize=6)
    # # axs[1].text(one_init_year+25,80,s='LT25Y',va='bottom',ha='center',fontsize=6)
    # # axs[1].text(one_init_year+1.5,55,s='"forecast"',va='top',ha='left',fontsize=8,bbox={'boxstyle':'rarrow','alpha':.7,'facecolor':'white'})
    # # axs[1].arrow(one_init_year,30,10,0,head_width=10,head_length=2)
    # axs[1].pcolormesh(index_int_var_obs_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}),)

    # for lt in [5,15,25]:
    #     # group hindcasts by lead time:
    #     lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
    #     lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
    #     lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
        
    #     axs[1].plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],lw=.75,color='k',label='LT{:}Y'.format(abs(lt)))


    # axs[0].format(
    #     xlabel='Year', ylabel='Hindcast Initialization Year',
    #     xlim=[ds_hc_proc.init_year.min().values,index_int_var_obs_proc.year.max().values]
    # )

    # axs[1].format(ylim=[-80,80],ylabel='NAO index')
    # axs[1].legend(loc='b',fontsize=8,ncols=4, frame=False)

    # bbox = axs[0].get_position()
    # width = bbox.width
    # height = bbox.height
    # angle = np.rad2deg(np.arctan2(height / width, 1))

    # lock_year = 1985
    # if hc_param_set["vf_win_len"] != "all":
    #     axs[0].annotate(text='', xy=(lock_year,lock_year), xytext=(lock_year-hc_param_set["vf_win_len"],lock_year), arrowprops=dict(arrowstyle='<->'))
    #     axs[0].text(lock_year-hc_param_set["vf_win_len"]//2,lock_year,'matching window ({:} years)'.format(hc_param_set["vf_win_len"]),ha='center',va='center',rotation=37,bbox={"fc":"lightgrey","ec":"none","pad":0})
    # # fig.savefig(fpath/"illustration/hc3.png",dpi=300,bbox_inches='tight')

    # # fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


# compute bootstrap intervals for each lead time for real analogue and AR(1) hindcasts:
def compute_bootstrap_score_lt(input_ds,lt,fc_type_name,n_boot,measure="corr", scale=False):
    
    lt_sel = input_ds.sel(lead_time=lt)
    lt_sel = lt_sel.where(~np.isnan(lt_sel.verification), drop=True)
    vf = lt_sel.verification.values
    fc = lt_sel[fc_type_name].mean("member").values
    if scale:
        fc = fc * np.nanstd(vf)/np.nanstd(fc)
    
    if measure == "corr":
        score,_ = stats.pearsonr(vf,fc)
    elif measure == "rmse":
        score = 1 - (np.nanmean((vf-fc)**2)**.5)/np.nanstd(vf)
    
    # compute p-value from phase randomization:
    null_phase_rs = null_dist_phase_randomization(vf, fc, n_boot=n_boot, measure=measure)
    pval_phase = bootstrap_pvalue_null(score, null_phase_rs)

    boot_rs = moving_block_bootstrap_pairs(vf, fc, L=max(int(np.round(len(vf)**(1/3))),Nyr), n_boot=n_boot, measure=measure)
    ci_lo, ci_hi = bootstrap_ci_from_bootrs(boot_rs, alpha=0.05)
    
    return score, pval_phase, ci_lo, ci_hi

n_boot = 1000
fc_types = [fct for fct in list(hc_coll.data_vars) if "forecast" in fct]
fc_types.append("climatological_forecast")
corr_hc = xr.Dataset(
    coords = {'lead_time':('lead_time',lead_times)}
)
for fc_type in fc_types:
    
    if fc_type == "climatological_forecast":
        input_ds = ds_modelclim_bylt
    else:
        input_ds = hc_coll
    bs_stats = Parallel(n_jobs=-1)(
        delayed(compute_bootstrap_score_lt)(input_ds,lt,fc_type,n_boot,"corr") for lt in lead_times
    )

    corr = []; p_val = []; ci_lower = []; ci_upper = []
    for bs_stat in bs_stats:
        corr.append(bs_stat[0]); p_val.append(bs_stat[1]); ci_lower.append(bs_stat[2]); ci_upper.append(bs_stat[3])

    corr_hc = corr_hc.assign({f"corr_{fc_type}":("lead_time",corr)})
    corr_hc = corr_hc.assign({f"pval_{fc_type}":("lead_time",p_val)})
    corr_hc = corr_hc.assign({f"ci_lower_{fc_type}":("lead_time",ci_lower)})
    corr_hc = corr_hc.assign({f"ci_upper_{fc_type}":("lead_time",ci_upper)})


# Take the hindcasts synchronized with the synthetic time series and look at their skill:
hc_synt_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_synthetic-{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)

# load all hindcasts:
hindcast_synt_files = sorted(list(hc_synt_path.glob('hc_*.nc')))
ds_hc_synt = xr.open_mfdataset(hindcast_synt_files,preprocess=prep_subs_hc)

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_synt_proc = ds_hc_synt['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_synt_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc_synt['scores'],1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# re-arrange the hindcast array to have lead time as dimension instead of valid year
ds_hc_synt_proc_bylt = by_year_struct_to_by_leadtime(ds_hc_synt_proc).rename("synthetic_analogue_forecast")

# include the synthetic index time series:
synt_obs = xr.open_dataset(proj_base/"data/synthetic_ar1_{:}_nao.nc".format(hc_param_set["ref_ds"]))
# process (average/filter):
synt_obs_proc = synt_obs['synthetic_NAO'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
# reindex the synthetic observations to match the hindcasts:
synt_obs_proc = synt_obs_proc.expand_dims(dim={'init_year':len(synt_obs_proc.year)}).assign_coords({'init_year':('init_year',synt_obs_proc.year.values)})
synt_obs_proc = synt_obs_proc.assign_coords(lead_time=synt_obs_proc.year - synt_obs_proc.init_year)
synt_obs_proc_bylt = by_year_struct_to_by_leadtime(synt_obs_proc)

synt_hc_ds = xr.merge([ds_hc_synt_proc_bylt,synt_obs_proc_bylt],join="inner")

# Evaluate the synthetic analogue hindcasts on the synthetic observations
syn_corr = xr.corr(synt_hc_ds["synthetic_analogue_forecast"].mean("member"),synt_hc_ds["synthetic_NAO"],dim="init_year")
sync_corr_95 = syn_corr.quantile([.05,.5,.95],dim="syobs_member")

#---------Plot the results------------#
fig_corr = pplt.figure(suptitle='NAO Correlation Skill',figsize=[5,3])
axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='Correlation')
axc.hlines(0,lead_times[0],lead_times[-1],color='grey',linestyle='dashed')
axc.plot(corr_hc.lead_time,corr_hc.corr_forecast,color='C1',label='AR(1) fair')
axc.fill_between(corr_hc.lead_time,corr_hc.ci_lower_forecast,corr_hc.ci_upper_forecast,alpha=.2,color='C1',label="CI AR(1)")
axc.plot(corr_hc.lead_time,corr_hc.corr_forecast_full,color='C1',label='AR(1) full',ls='dashed',lw=.5)
axc.plot(corr_hc.lead_time,corr_hc.corr_analogue_forecast,color='C0',label='Analogue')
axc.fill_between(corr_hc.lead_time,corr_hc.ci_lower_analogue_forecast,corr_hc.ci_upper_analogue_forecast,alpha=.2,color='C0',label="CI analogue")
axc.plot(sync_corr_95.lead_time,sync_corr_95.sel(quantile=.95),color='k',label='random 95th',ls="dotted")

axc.plot([5.5,5.5],[0.57,0.79],color='k',marker='_')
axc.annotate(text="  DCPP (LY2-9)",xy=(5.5,0.8),ha="left",va="center")
axc.legend(loc=3,ncol=3)
axc.format(ylim=[-.4,.9],xlim=[lead_times[0],lead_times[-1]])
# fig_corr.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_correlation_{subsampling}_sync_{ref_ds}_subs41y_{vf_win_len}yvf.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


# Do the same evaluation but using RMSE as deterministic score:
# SCALE??
var_rat = hc_coll.verification.std(dim=["init_year"])/hc_coll.mean("member").std(dim=["init_year"])
# hc_coll *= var_rat

rmse_hc = xr.Dataset(
    coords = {'lead_time':('lead_time',lead_times)}
)
for fc_type in fc_types:
    
    if fc_type == "climatological_forecast":
        input_ds = ds_modelclim_bylt
    else:
        input_ds = hc_coll
    bs_stats = Parallel(n_jobs=-1)(
        delayed(compute_bootstrap_score_lt)(input_ds,lt,fc_type,n_boot,"rmse",scale=False) for lt in lead_times
    )

    rmse = []; p_val = []; ci_lower = []; ci_upper = []
    for bs_stat in bs_stats:
        rmse.append(bs_stat[0]); p_val.append(bs_stat[1]); ci_lower.append(bs_stat[2]); ci_upper.append(bs_stat[3])

    rmse_hc = rmse_hc.assign({f"rmse_{fc_type}":("lead_time",rmse)})
    rmse_hc = rmse_hc.assign({f"pval_{fc_type}":("lead_time",p_val)})
    rmse_hc = rmse_hc.assign({f"ci_lower_{fc_type}":("lead_time",ci_lower)})
    rmse_hc = rmse_hc.assign({f"ci_upper_{fc_type}":("lead_time",ci_upper)})

# Evaluate the synthetic analogue hindcasts on the synthetic observations
syn_rmse = ((synt_hc_ds["synthetic_analogue_forecast"].mean("member") - synt_hc_ds["synthetic_NAO"])**2).mean(dim="init_year")**.5
syn_rmse_ref = synt_hc_ds["synthetic_NAO"].std(["init_year"])
syn_rmsess = 1 - (syn_rmse/syn_rmse_ref)
sync_rmsess_95 = syn_rmsess.quantile([.05,.5,.95],dim="syobs_member")


#---------Plot the results------------#
fig_corr = pplt.figure(suptitle='NAO RMSE Skill',figsize=[5,3])
axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='RMSESS')
axc.hlines(0,lead_times[0],lead_times[-1],color='grey',linestyle='dashed')
axc.plot(rmse_hc.lead_time,rmse_hc.rmse_forecast,color='C1',label='AR(1) fair')
axc.fill_between(rmse_hc.lead_time,rmse_hc.ci_lower_forecast,rmse_hc.ci_upper_forecast,alpha=.2,color='C1',label="CI AR(1)")
axc.plot(rmse_hc.lead_time,rmse_hc.rmse_forecast_full,color='C1',label='AR(1) full',ls='dashed',lw=.5)
axc.plot(rmse_hc.lead_time,rmse_hc.rmse_analogue_forecast,color='C0',label='Analogue')
axc.fill_between(rmse_hc.lead_time,rmse_hc.ci_lower_analogue_forecast,rmse_hc.ci_upper_analogue_forecast,alpha=.2,color='C0',label="CI analogue")
axc.plot(sync_rmsess_95.lead_time,sync_rmsess_95.sel(quantile=.95),color='k',label='random 95th',ls="dotted")

axc.legend(loc=1,ncol=3)
axc.format(ylim=[-.5,.5],xlim=[lead_times[0],lead_times[-1]])
# fig_corr.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_rmsess_{subsampling}_sync_{ref_ds}_subs41y_{vf_win_len}yvf.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


# Transform forecasts and observations into categories/binary time series
# first, need a proper reference period
# for each init_year, the reference period should cover X years prior to the init_year
# note that we want to always pick the same valid years as far as there exists data 
# for each lead_time

# for each initialization, get a separate climatology based on the previous N years
N = 30
# N_clim_thres = N//3
quantiles = [0.333,.5,.667]

hc_clim = []; obs_quants = []; full_clim = []; obs_mean = []; obs_std = []
nan_fill = xr.Dataset(
    data_vars = {dv:(["quantile"],np.nan*np.zeros_like(quantiles)) for dv in hc_coll.data_vars},
    coords={"quantile":(["quantile"],quantiles)}
)
nan_fill_full = xr.Dataset(
    data_vars = {dv:(["quantile"],np.nan*np.zeros_like(quantiles)) for dv in ds_modelclim_bylt.data_vars},
    coords={"quantile":(["quantile"],quantiles)}
)

for iy in hc_coll.init_year.values:
    clim_range = np.arange(iy-N,iy)

    clim_incl = (hc_coll.year.values >= clim_range[0]) & (hc_coll.year.values <= clim_range[-1])
    clim_excl = (hc_coll.year.values >= clim_range[-1]) & (hc_coll.year.values <= (clim_range[-1] + N))

    obs_clim = index_int_var_obs_proc.sel(year=clim_range)
    # compute quantiles from this distribution:
    obs_quants.append(obs_clim.quantile(quantiles))

    obs_mean.append(obs_clim.mean())
    obs_std.append(obs_clim.std())

    hc_clim_lt = []; full_clim_lt = []
    for ll,lt in enumerate(hc_coll.lead_time.values):
        if clim_incl[:,ll].sum() == N:
            hc_clim_lt.append(hc_coll.sel(lead_time=lt).isel(init_year = clim_incl[:,ll]).quantile(quantiles,skipna=True))
            full_clim_lt.append(ds_modelclim_bylt.sel(lead_time=lt).isel(init_year = clim_incl[:,ll]).quantile(quantiles,skipna=True))
        else: # if not all years are available relax the requirement for exact years and take some from the future to get a larger sample
            hc_clim_lt.append(hc_coll.sel(lead_time=lt).isel(init_year = ~clim_excl[:,ll]).isel(init_year=slice(N)).quantile(quantiles,skipna=True))
            full_clim_lt.append(ds_modelclim_bylt.sel(lead_time=lt).isel(init_year = ~clim_excl[:,ll]).isel(init_year=slice(N)).quantile(quantiles,skipna=True))
        
    hc_clim.append(xr.concat(hc_clim_lt,dim=hc_coll.lead_time))
    full_clim.append(xr.concat(full_clim_lt,dim=ds_modelclim_bylt.lead_time))

clim_quants = xr.concat(hc_clim,dim=hc_coll.init_year).assign_coords(year=hc_coll.year)
full_clim_quants = xr.concat(full_clim,dim=ds_modelclim_bylt.init_year).assign_coords(year=ds_modelclim_bylt.year)
obs_quants = xr.concat(obs_quants,dim=hc_coll.init_year)
obs_mean = xr.concat(obs_mean,dim=hc_coll.init_year)
obs_std = xr.concat(obs_std,dim=hc_coll.init_year)

# Do the same for synthetic forecasts
synth_clim = []
nan_fill_synth = xr.Dataset(
    data_vars = {dv:(["syobs_member","quantile"],np.nan*np.zeros([len(synt_hc_ds.syobs_member),len(quantiles)])) for dv in synt_hc_ds.data_vars},
    coords={
        "quantile":(["quantile"],quantiles),
        "syobs_member":synt_hc_ds.syobs_member
    }
)
for iy in synt_hc_ds.init_year.values:
    clim_range = np.arange(iy-N,iy)

    clim_incl = (synt_hc_ds.year.values >= clim_range[0]) & (synt_hc_ds.year.values <= clim_range[-1])
    clim_excl = (synt_hc_ds.year.values >= clim_range[-1]) & (synt_hc_ds.year.values <= (clim_range[-1] + N))

    synth_clim_lt = []
    for ll,lt in enumerate(synt_hc_ds.lead_time.values):
        if clim_incl[:,ll].sum() == N:
            synth_clim_lt.append(synt_hc_ds.sel(lead_time=lt).isel(init_year = clim_incl[:,ll]).quantile(quantiles,skipna=True,dim=["init_year","member"]))
        else:
            synth_clim_lt.append(synt_hc_ds.sel(lead_time=lt).isel(init_year = ~clim_excl[:,ll]).isel(init_year=slice(N)).quantile(quantiles,skipna=True,dim=["init_year","member"]))
            # synth_clim_lt.append(nan_fill_synth)

    synth_clim.append(xr.concat(synth_clim_lt,dim=synt_hc_ds.lead_time))

clim_quants_synth = xr.concat(synth_clim,dim=synt_hc_ds.init_year).assign_coords(year=synt_hc_ds.year)


thresh = 0.5
# Prediction: is the NAO above or below normal (median):
# for observations, estimate the median from the available observational record
# obs_median_thresh = index_int_var_obs_proc.sel(year=slice(1911,2020)).quantile(0.5)
vf_pos = (hc_coll.verification < obs_quants.sel(quantile=thresh)).astype(int)
# vf_pos = (hc_coll.verification > 0).astype(int)

# for the model, compute a lead-time dependent median from the full hindcast set
hc_probs = (hc_coll < clim_quants.sel(quantile=thresh)).astype(int).mean("member")
# hc_probs = (hc_coll > 0).astype(int).mean("member")
# set nan where no climatology exists:
hc_probs = hc_probs.where(~np.isnan(clim_quants.sel(quantile=thresh)))

# # anfc_median_thresh = hc_coll.analogue_forecast.quantile(0.5,dim=["init_year","member"])
# anfc_pos = (hc_coll.analogue_forecast > anfc_median_thresh).astype(int).mean("member")

# # for the model, compute a lead-time dependent median from the full hindcast set
# arfc_median_thresh = hc_coll.forecast.quantile(0.5,dim=["init_year","member"])
# arfc_pos = (hc_coll.forecast > arfc_median_thresh).astype(int).mean("member")

# # for the model, compute a lead-time dependent median from the full hindcast set
# fullarfc_median_thresh = hc_coll.forecast_full.quantile(0.5,dim=["init_year","member"])
# fullarfc_pos = (hc_coll.forecast_full > fullarfc_median_thresh).astype(int).mean("member")

# for the model, compute a lead-time dependent median from the full hindcast set
# clim_median_thresh = ds_modelclim_bylt.climatological_forecast.quantile(0.5,dim=["init_year","member"])
clim_pos = (ds_modelclim_bylt < full_clim_quants.sel(quantile=thresh)).astype(int).mean("member")
# clim_pos = (ds_modelclim_bylt > 0).astype(int).mean("member")
clim_pos = clim_pos.where(~np.isnan(full_clim_quants.sel(quantile=thresh)))


# synthetic forecasts:
synt_probs = (synt_hc_ds < clim_quants_synth.sel(quantile=thresh)).astype(int).mean("member")
# synt_probs = (synt_hc_ds > 0).astype(int).mean("member")
# synt_pos = (synt_hc_ds.synthetic_analogue_forecast > synt_hc_ds.synthetic_analogue_forecast.quantile(0.5,dim=["init_year","member"])).astype(int).mean("member")
# synt_vf_pos = (synt_hc_ds.synthetic_NAO > synt_hc_ds.synthetic_NAO.sel(lead_time=1).quantile(0.5,dim=["init_year"])).astype(int)

bs_anfc = np.nanmean(ps.brier_score(vf_pos,hc_probs.analogue_forecast),axis=0)
bs_arfc = np.nanmean(ps.brier_score(vf_pos,hc_probs.forecast),axis=0)
bs_fullarfc = np.nanmean(ps.brier_score(vf_pos,hc_probs.forecast_full),axis=0)
bs_clim = np.nanmean(ps.brier_score(vf_pos,clim_pos.climatological_forecast),axis=0)
bs_synt = np.nanmean(ps.brier_score(synt_probs.synthetic_NAO,synt_probs.synthetic_analogue_forecast),axis=0)

signif = 90
# reference forecast is always predicting .5
bs_ref = np.nanmean(ps.brier_score(vf_pos,.5),axis=0)
bs_ref_synt = np.nanmean(ps.brier_score(synt_probs.synthetic_NAO,.5),axis=0)
bss_synt_sign = np.percentile(1 - bs_synt/bs_ref_synt,signif,axis=0)

plt.plot(1 - bs_anfc/bs_ref,label="analog")
plt.plot(1 - bs_arfc/bs_ref,label="AR(1)")
plt.plot(1 - bs_fullarfc/bs_ref,label="AR(1) full")
plt.plot(1 - bs_clim/bs_ref,'k',label="clim")
plt.plot(bss_synt_sign,'k--',label=f"p{signif} synth")
plt.legend()

# Look at calibration seperately??
# can we get some indication of whether its reasonable to look at single ensemble members?
# need to fix bins, digitize
bwidth = .2
prob_bin_edges = np.concatenate([np.arange(0,1,bwidth),[1 + bwidth/100]])
prob_vals = prob_bin_edges[:-1] + np.diff(prob_bin_edges)/2
for LT in [5,10,15,20,25]:
    pos_mod_lt = np.digitize(hc_probs.analogue_forecast.sel(lead_time=LT),bins=prob_bin_edges)
    pos_mod_lt[np.isnan(hc_probs.analogue_forecast.sel(lead_time=LT))] = 0
    #! TODO: is this correct???
    pos_vf_lt = hc_probs.verification.sel(lead_time=LT)
    
    # how often are forecasts in each bin (sum)? --> size of the dot
    fc_freq = np.bincount(pos_mod_lt,minlength=len(prob_bin_edges))[1:] # cut off the first bin which represents nan values

    # what's the average occurrence of NAO > 0 in each of these bins? --> y-axis value
    obs_freq = []
    for pp in range(1,len(prob_bin_edges)):
        cat_masked = pos_vf_lt.where(pos_mod_lt==pp)
        if not np.isfinite(cat_masked).any():
            obs_freq.append(0)
        else:
            obs_freq.append(cat_masked.sum().values/np.isfinite(cat_masked).sum().values*100)

    f,ax = plt.subplots()
    ax.scatter(prob_vals*100,obs_freq,s=fc_freq*10)
    ax.plot([0,100],[0,100],'k')
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    # ax.set_xlim([0,100])
    # ax.set_ylim([0,100])
    ax.hlines((1-thresh)*100,0,100,ls="dashed",color='grey',lw=.75)
    ax.vlines((1-thresh)*100,0,100,ls="dashed",color='grey',lw=.75)
    ax.set_title(f"Reliability {LT}Y prediction")

    axi = ax.inset_axes([5, 75, 35, 20], transform=ax.transData)
    axi.bar(prob_vals*100,fc_freq,width=bwidth*90);
    axi.set_xlim([0,100])
    # axi.set_ylabel("# samples")
    axi.set_ylim([0,(~np.isnan(hc_probs.analogue_forecast.sel(lead_time=LT))).sum()+2])
    # axi.set_title(f"# Samples")


# Probabilistic Evaluation:
# test whether there is useful information in the ensemble spread as well
# CRPS but also BS for realistic thresholds
fc_crps = np.nanmean(ps.crps_ensemble(hc_coll.verification,hc_coll.forecast),axis=0)
fc_ana_crps = np.nanmean(ps.crps_ensemble(hc_coll.verification,hc_coll.analogue_forecast),axis=0)
fc_full_crps = np.nanmean(ps.crps_ensemble(hc_coll.verification,hc_coll.forecast_full),axis=0)
fc_clim_crps = np.nanmean(ps.crps_ensemble(ds_modelclim_bylt.verification,ds_modelclim_bylt.climatological_forecast),axis=0)
fc_ref_crps = np.nanmean(ps.crps_gaussian(ds_modelclim_bylt.verification,obs_mean.expand_dims({"lead_time":ds_modelclim_bylt.lead_time}).T,obs_std.expand_dims({"lead_time":ds_modelclim_bylt.lead_time}).T),axis=0)
fc_perfref_crps = np.nanmean(ps.crps_gaussian(ds_modelclim_bylt.verification,ds_modelclim_bylt.verification.mean(),ds_modelclim_bylt.verification.std()),axis=0)
fc_synth_crps = np.nanmean(ps.crps_ensemble(synt_hc_ds.synthetic_NAO,synt_hc_ds.synthetic_analogue_forecast.transpose("init_year","syobs_member","lead_time","member")),axis=0)
fc_synth_crps_perfref = np.nanmean(ps.crps_gaussian(synt_hc_ds.synthetic_NAO,synt_hc_ds.synthetic_NAO.mean(),synt_hc_ds.synthetic_NAO.std()),axis=0)
fc_synth_crps_sign = np.quantile(1 - fc_synth_crps/fc_synth_crps_perfref,signif/100,axis=0)
# These are basically uncalibrated!
# Calibration is extremely difficult due to the relative shortness of the record! It can make things worse
# Rather use ensemble members at face value instead of trying to calibrate?
# --> each ens mem: deviation from the forced signal

plt.plot(1 - fc_ana_crps/fc_perfref_crps,label="analog")
plt.plot(1 - fc_crps/fc_perfref_crps,label="AR(1)")
plt.plot(1 - fc_full_crps/fc_perfref_crps,label="AR(1) full")
plt.plot(1 - fc_clim_crps/fc_perfref_crps,'k',label="clim")
# plt.plot(fc_perfref_crps,'k',label="perfect ref",ls="dotted")
# plt.plot(fc_ref_crps,'k',label="realistic ref",ls="dashed")
plt.plot(fc_synth_crps_sign,'k--',label=f"p{signif} synth")
plt.legend()

# # Consider the spread of the hindcasts:
# quantiles = ds_hc_proc.quantile([.1,.25,.5,.75,.9],dim="member")
# spread = quantiles.sel(quantile=.75) - quantiles.sel(quantile=.25)
# spread_90 = quantiles.sel(quantile=.9) - quantiles.sel(quantile=.1)

# quantiles_f = index_int_var_model_proc.quantile([.1,.25,.5,.75,.9],dim="member")
# spread_f = quantiles_f.sel(quantile=.75) - quantiles_f.sel(quantile=.25)
# spread_90_f = quantiles_f.sel(quantile=.9) - quantiles_f.sel(quantile=.1)

# f,ax = pplt.subplots()
# spread.groupby("lead_time").mean().plot(ax=ax)
# spread_f.groupby("lead_time").mean().plot(ax=ax)

# # re-arrange the hindcasts to have lead time as a dimension instead of valid year:
# ds_hc_mtrx = []; vf_hc_like = []; ds_fullens_mtrx = []
# for iyear in ds_hc_proc.init_year.values:
#     single_init = ds_hc_proc.sel(init_year=iyear)
#     single_init_shifted = single_init.sel(year=single_init.lead_time > 0).swap_dims({"year":"lead_time"})
#     single_init_incl = single_init_shifted.sel(lead_time=slice(35))
#     ds_hc_mtrx.append(single_init_incl)
    
#     vf_matched = index_int_var_obs_proc.reindex(year=single_init_incl.year.values).assign_coords({"lead_time":("year",single_init_incl.lead_time.values)}).swap_dims({"year":"lead_time"})
#     vf_hc_like.append(vf_matched)

#     single_init_fullens = index_int_var_model_proc.sel(init_year=iyear)
#     single_init_fullens_shifted = single_init_fullens.sel(year=single_init_fullens.lead_time > 0).swap_dims({"year":"lead_time"})
#     single_init_fullens_incl = single_init_fullens_shifted.sel(lead_time=slice(35))
#     ds_fullens_mtrx.append(single_init_fullens_incl)
    
# ds_hc_mtrx = xr.concat(ds_hc_mtrx,dim="init_year")
# ds_fullens_mtrx = xr.concat(ds_fullens_mtrx,dim="init_year")
# # verification
# vf_hc_like = xr.concat(vf_hc_like,dim=ds_hc_mtrx.init_year).drop("mode")

# # bias correct (not properly leave-out!)
# hc_bias = ds_hc_mtrx.mean(("init_year","member")) - vf_hc_like.mean("init_year")
# fullens_bias = ds_hc_mtrx.mean(("init_year","member")) - vf_hc_like.mean("init_year")
# # ds_hc_mtrx -= hc_bias
# # ds_fullens_mtrx -= fullens_bias

# # RMSE:
# rmse_hc = (((ds_hc_mtrx.mean("member") - vf_hc_like)**2).mean("init_year")**.5)
# rmse_ref = (((ds_fullens_mtrx.mean("member") - vf_hc_like)**2).mean("init_year")**.5)
# rmsess = 1 - rmse_hc/rmse_ref
# rmsess.plot()

# # Evaluate the ensemble with a simple Brier Score (NAO above or below 0)
# thresh_vf = vf_hc_like.quantile(.5,dim="init_year") # 0
# thresh_hc = ds_hc_mtrx.quantile(.5,dim="init_year") # index_int_var_model_proc.quantile(.5)
# thresh_mod = ds_fullens_mtrx.quantile(.5,dim="init_year") # index_int_var_model_proc.quantile(.5)

# prob_vals = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
# pos_mod = (ds_hc_mtrx > thresh_hc).mean("member").round(1)
# pos_fullens = (ds_fullens_mtrx > thresh_mod).mean("member").round(1)
# pos_vf = vf_hc_like > thresh_vf
# BS = ((pos_mod - pos_vf)**2).mean("init_year")
# BSref = ((pos_fullens - pos_vf)**2).mean("init_year")
# BSS = 1 - BS/BSref
# BSS.plot()

# # Reliability Diagram (0 thresh, one lead year):
# for LT in [5,10,15,20]:
#     pos_mod_lt = pos_fullens.sel(lead_time=LT)
#     pos_vf_lt = pos_vf.sel(lead_time=LT)
#     obs_freq = []; fc_freq = []
#     for prob in prob_vals:
#         fc_freq.append((pos_mod_lt==prob).sum().values*10)
#         cat_masked = pos_vf_lt.where(pos_mod_lt==prob)
#         obs_freq.append(cat_masked.sum().values/np.isfinite(cat_masked).sum().values*100)

#     plt.figure()
#     plt.scatter(prob_vals*100,obs_freq,s=fc_freq)
#     plt.plot([0,100],[0,100],'k')
#     plt.title(f"Reliability {LT}Y prediction")

# Evaluate three category ranked probability score for terciles
# derive terciles from obs:



# TODO:
# evaluate spread --> RPSS?
# Random Walk test? DelSole & Tippett (2015) 

# Illustration of analogue method:
import matplotlib.transforms as transforms
from matplotlib.patches import FancyArrowPatch

obs_year_lims = [min(index_int_var_obs_proc.year),max(index_int_var_obs_proc.year)]
N_win = 70
N_ld = 35

f,ax = pplt.subplots(width=7,height=2)
ax.plot(index_int_var_obs_proc,'k')
ax.set_xlim([obs_year_lims[0],obs_year_lims[-1]+N_ld])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_ylim([min(index_int_var_obs_proc)-30,max(index_int_var_obs_proc)+100])
ax.grid(False)
ax.set_xticks([obs_year_lims[-1]-N_win,obs_year_lims[-1]])
ax.set_xticklabels(["t$_{0}$ - N$_{win}$","t$_{0}$\n(today)"])
ax.set_yticks([])
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.xaxis.set_minor_locator(plt.NullLocator())
yls = ax.get_ylim()
ax.vlines(obs_year_lims[-1],yls[0],yls[-1],color='k',ls="dashed")
ax.vlines(obs_year_lims[-1]-N_win,yls[0],yls[-1],color='k',ls="dashed")
ax.set_ylabel("")
ax.set_xlabel("")
ax.annotate("Observed NAO",(1875,20),color='k')

blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
rect_win = plt.Rectangle(
    (obs_year_lims[-1] - N_win, 0),                    # (x_start, y_start)
    N_win,                  # width in data coords
    0.9,                          # height in axes coords (0 to 0.8)
    transform=blend,
    facecolor='C1',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)
ax.add_patch(rect_win)

rect_fc = plt.Rectangle(
    (obs_year_lims[-1], 0),                    # (x_start, y_start)
    N_ld,                  # width in data coords
    1,                          # height in axes coords (0 to 0.8)
    transform=blend,
    facecolor='C0',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)
ax.add_patch(rect_fc)
ax.annotate("Forecast Period\nt$_0$ to t$_0$ + N$_{lead}$",(obs_year_lims[-1]+N_ld//2,80),color="C0",va="center",ha="center")
ax.annotate("Matching Window\nt$_0$-N$_{win}$ to t$_0$",(obs_year_lims[-1]-N_win//2,80),color="C1",va="center",ha="center")

arrow = FancyArrowPatch(
    (obs_year_lims[-1] - N_win, .9),       # Start: (data_x, axes_y)
    (obs_year_lims[-1], .9),       # End: (data_x, axes_y)
    arrowstyle='<->',          # Double headed
    mutation_scale=10,         # Size of arrow heads
    linewidth=1.5,
    color='black',
    transform=blend            # Apply the blended transform here
)
ax.add_patch(arrow)

ax.text(
    obs_year_lims[-1] - N_win//2,
    .9 + 0.05,  # Slightly above the arrow
    'N$_{win}$',
    ha='center',
    va='bottom',
    transform=blend,
    fontsize=10
)
f.savefig(fpath/"illustration/obs_match.png",dpi=300,bbox_inches="tight")

rng = np.random.default_rng(seed=325)
rand_mems = rng.integers(len(index_int_var_model_proc_std.member),size=15,)
rand_mem_sel = index_int_var_model_proc_std.isel(member=rand_mems)
offs = np.cumsum(np.array([150]*rand_mem_sel.shape[0]))

f,ax = pplt.subplots(width=7,height=2)
for i,mem in enumerate(rand_mem_sel):
    ax.plot(mem+offs[i],lw=.5,alpha=.5,color='k')
ax.set_xlim([obs_year_lims[0],obs_year_lims[-1]+N_ld])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.set_ylim([min(index_int_var_obs_proc)-30,max(index_int_var_obs_proc)+100])
ax.grid(False)
# ax.set_xticks([obs_year_lims[-1]-N_win,obs_year_lims[-1]])
# ax.set_xticklabels(["t$_{0}$ - N$_{win}$","t$_{0}$\n(today)"])
ax.set_xticks([])
ax.set_yticks([])
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.xaxis.set_minor_locator(plt.NullLocator())
yls = ax.get_ylim()
ax.vlines(obs_year_lims[-1],yls[0],yls[-1],color='k',ls="dashed")
ax.vlines(obs_year_lims[-1]-N_win,yls[0],yls[-1],color='k',ls="dashed")
ax.set_ylabel("")
ax.set_xlabel("")
# ax.annotate("Observed NAO",(1875,20),color='k')

blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
rect_win = plt.Rectangle(
    (obs_year_lims[-1] - N_win, 0),                    # (x_start, y_start)
    N_win,                  # width in data coords
    1,                          # height in axes coords (0 to 0.8)
    transform=blend,
    facecolor='C1',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)
ax.add_patch(rect_win)

rect_fc = plt.Rectangle(
    (obs_year_lims[-1], 0),                    # (x_start, y_start)
    N_ld,                  # width in data coords
    1,                          # height in axes coords (0 to 0.8)
    transform=blend,
    facecolor='C0',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)
ax.add_patch(rect_fc)

f.savefig(fpath/"illustration/sim_match.png",dpi=300,bbox_inches="tight")


# Simulations ranked by goodness of match:
rand_mem_corr = xr.corr(rand_mem_sel.sel(year=slice(obs_year_lims[-1]-N_win,obs_year_lims[-1])),index_int_var_obs_proc.sel(year=slice(obs_year_lims[-1]-N_win,obs_year_lims[-1])),dim="year")
ranked_sims = rand_mem_sel.isel(member=rand_mem_corr.argsort().values[::-1])

f,ax = pplt.subplots(width=4,height=2)
for i,mem in enumerate(ranked_sims[::-1]):
    ax.plot(mem+offs[i],alpha=.5,lw=1+rand_mem_corr.sel(member=mem.member.values).values,color="k")
ax.set_xlim([obs_year_lims[-1]-N_win,obs_year_lims[-1]])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.set_ylim([min(index_int_var_obs_proc)-30,max(index_int_var_obs_proc)+100])
ax.grid(False)
# ax.set_xticks([obs_year_lims[-1]-N_win,obs_year_lims[-1]])
# ax.set_xticklabels(["t$_{0}$ - N$_{win}$","t$_{0}$\n(today)"])
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_minor_locator(plt.NullLocator())
yls = ax.get_ylim()
ax.set_ylabel("")
ax.set_xlabel("")
# ax.annotate("Observed NAO",(1875,20),color='k')

blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
rect_win = plt.Rectangle(
    (obs_year_lims[-1] - N_win, 0),                    # (x_start, y_start)
    N_win,                  # width in data coords
    1,                          # height in axes coords (0 to 0.8)
    transform=blend,
    facecolor='C1',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)
ax.add_patch(rect_win)
f.savefig(fpath/"illustration/sim_ranked.png",dpi=300,bbox_inches="tight")


# Ensemble forecast:

illu_ana =ranked_sims[-5:][::-1]

f,ax = pplt.subplots(width=4,height=2)
ax.plot(illu_ana.T,alpha=.5,lw=.75,color='k')
ax.plot(illu_ana.mean("member"),lw=1.25,color='k')
ax.set_xlim([obs_year_lims[-1],obs_year_lims[-1]+N_ld])

rect_fc = plt.Rectangle(
    (0, 0),                    # (x_start, y_start)
    1,                  # width in data coords
    1,                          # height in axes coords (0 to 0.8)
    transform=ax.transAxes,
    facecolor='C0',           # highlight color
    alpha=0.2,                    # transparency
    clip_on=False                 # allow drawing outside axes if needed
)

ax.add_patch(rect_fc)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.set_ylim([min(index_int_var_obs_proc)-30,max(index_int_var_obs_proc)+100])
ax.grid(False)
# ax.set_xticks([obs_year_lims[-1]-N_win,obs_year_lims[-1]])
# ax.set_xticklabels(["t$_{0}$ - N$_{win}$","t$_{0}$\n(today)"])
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylabel("")
ax.set_xlabel("")

f.savefig(fpath/"illustration/sim_analog_fc.png",dpi=300,bbox_inches="tight")
