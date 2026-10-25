
import xarray as xr
import pandas as pd
import numpy as np
import re
import xesmf as xe
from scipy import stats
import proplot as pplt
from pathlib import Path
from climate_scenarios_2050.config import proj_base
from climate_scenarios_2050.generate_NAO_hindcast import load_model_ensemble_NAO
from climate_scenarios_2050.filters import butter_bandpass_filter, butter_lowpass_filter
from climpred import HindcastEnsemble
from dask.distributed import Client, get_client

client = Client()


def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def align_e5(ds):
    if "valid_time" in list(ds.coords):
        return ds.rename({"valid_time":"time"}).drop_vars(["number","expver"])
    elif "date" in list(ds.coords):
        return ds.assign_coords(time=pd.Timestamp(str(ds.date.values))).expand_dims("time").drop_vars(["number","date","expver"])
    else:
        return ds

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

hc_dir = Path("/datalake/NS9873K/DATA/CMIP6_hindcast/")
vf_path = Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1")

hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,}

hc_dir_params = hc_dir/"hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx".format(**hc_param_set)


# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 8

# lead_times: what lead times to verify
lead_times = np.arange(0,36,1)
lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

linestyle = {5:'solid',15:'dashed',25:'dotted',35:'solid'}

variable = "tas" # "sfcWind"
var_sh = "t2m" # "si10"
variable_long = "2m_temperature" # "10m_wind_speed"
int_meth = "bilinear"

#----------------------------------------------------------------------------------#
# Load CMIP6 "hindcast" fields
#----------------------------------------------------------------------------------#
hc_filename = list(hc_dir_params.glob(f"hc_{variable}_*.zarr"))[0]
hc_ens_field = xr.open_mfdataset(hc_filename,engine="zarr")
hc_ens_field = hc_ens_field.chunk({"lat":60,"lon":120,"lead_time":-1,"member":100,"init_year":1})

hc_ens_field_grouped = hc_ens_field.groupby("lead_time").mean().chunk({"lead_time":-1})
hc_ens_field_av = hc_ens_field_grouped.rolling(lead_time=Nyr).mean().chunk({"lat":60,"lon":120,"lead_time":-1,"member":100,"init_year":1})
# manually drop the first Nyr-1 years since they're NaN:
hc_ens_field_av = hc_ens_field_av.isel(lead_time=slice(Nyr-1,None))

# generate the "valid time" variable to align with observations:
# hc_ens_field_av = hc_ens_field_av.assign_coords(valid_time = hc_ens_field.init_year + hc_ens_field_av.lead_time)


# load a verification dataset/reference:
ref_files = sorted(list(vf_path.glob(f"{variable_long}/*.nc")))
ref_field = xr.open_mfdataset(ref_files,preprocess=align_e5,combine="nested",concat_dim="time").rename({"latitude":"lat","longitude":"lon"})

# average to annual data
ref_field_av = ref_field.groupby("time.year").mean().rolling(year=Nyr).mean().chunk({"lat":60,"lon":120,"year":-1}).dropna("year")

# re-grid to match hindcasts:
to_grid = xr.Dataset({"lat": hc_ens_field_av.lat,"lon": hc_ens_field_av.lon,})
regridder = xe.Regridder(ref_field_av, to_grid, method=int_meth, ignore_degenerate=True, periodic=True)

ref_field_av_rgr = regridder(ref_field_av)

# make anomalies:
hc_ens_field_av = hc_ens_field_av.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in hc_ens_field_av.member.values]))

# Compute model means
hc_ens_field_av_mmean = hc_ens_field_av.sel(init_year=slice(1980,2010)).groupby("m_id").mean(["member","init_year"])

# Subtract model means → anomalies per member
hc_ens_field_av_an = hc_ens_field_av.chunk({'member':10}).groupby("m_id") - hc_ens_field_av_mmean
hc_ens_field_av_an = hc_ens_field_av_an.chunk({'member':100})

ref_field_av_rgr_an = ref_field_av_rgr - ref_field_av_rgr.sel(year=slice(1980,2010)).mean("year")

# wrap into a climpred dataset:
hindcasts = HindcastEnsemble(hc_ens_field_av_an.where(hc_ens_field_av_an.init_year >= ref_field_av_rgr_an.year.min(), drop=True).rename({"init_year":"init","lead_time":"lead"}))
hindcasts = hindcasts.add_observations(ref_field_av_rgr_an.rename({"year":"time",var_sh:variable}))

# evaluate
acc = hindcasts.chunk({"init":-1,"lead":1}).verify(metric="acc", comparison="e2o", dim="init", alignment="same_verif").load()

# plot global skill:
fig,axs = pplt.subplots(figwidth=5,figheight=4,suptitle="ACC {:}".format(" ".join(variable_long.split('_'))))
acc.sel(lead=15)[variable].plot(ax=axs)
fig.savefig(f"acc_{variable_long}.png",dpi=300,bbox_inches="tight")

hc_ens_field_av_an = hc_ens_field_av_an.assign_coords(time = hc_ens_field_av_an.init_year + hc_ens_field_av_an.lead_time)
fig,axs = pplt.subplots(figwidth=5,figheight=4,suptitle="Prediction global {:}".format(" ".join(variable_long.split('_'))))
hc_ens_field_av_an[variable].mean(["lat","lon","member"]).sel(lead_time=[7,10,15,20,25,30,35]).plot(hue="lead_time",ax=axs,x="time")
ref_field_av_rgr_an[var_sh].mean(["lat","lon"]).plot(color="k",ax=axs)
fig.savefig(f"global_tas_{variable_long}.png",dpi=300,bbox_inches="tight")

hindcast = hc_ens_field_av.where(hc_ens_field_av.valid_time <= ref_field_av_rgr.year.max().values, drop=True)
hcst_anom = hindcast - hindcast.mean("init_year")
ref_anom  = ref_field_av_rgr - ref_field_av_rgr.mean("year")  # example



#----------------------------------------------------------------------------------#
# Raw CMIP6 ensemble
#----------------------------------------------------------------------------------#

#-----Create a "baseline" hindcast that just takes the CMIP6 ensemble without syncing
# reference: what observational dataset to use as a reference:

dummy_params = hc_param_set

# load all "non-initialized" hindcasts:
ds_hc_raw_full = load_model_ensemble_NAO(None,dummy_params['scen'],dummy_params['ref_ds'])

# interpolate 2014:
years_full = np.arange(ds_hc_raw_full.year.min(), ds_hc_raw_full.year.max() + 1)
ds_hc_raw_full = ds_hc_raw_full.reindex(year=years_full).interpolate_na(dim='year', method='linear')

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_raw_proc = ds_hc_raw_full['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_raw_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc_raw_full['scores'],1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# Observations:
nao_vf_full = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(dummy_params['ref_ds']))
nao_vf_full = nao_vf_full.sel(mode=1).squeeze().scores

# process:
if proc_type == 'running_mean':
    vf_proc_raw = nao_vf_full.rolling(year=Nyr).mean('year').dropna('year')
elif proc_type == 'lowpass':
    vf_proc_raw = xr.apply_ufunc(
        butter_lowpass_filter,nao_vf_full,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )

# remove forced signal:
forced_signal_model = ds_hc_raw_proc.mean('member')
forced_signal_obs = xr.cov(vf_proc_raw.sel(year=forced_signal_model.sel(year=vf_proc_raw.year).year),forced_signal_model.sel(year=vf_proc_raw.year))/forced_signal_model.sel(year=vf_proc_raw.year).var() * forced_signal_model
ds_hc_proc = ds_hc_raw_proc - forced_signal_model

# remove forced signal from model:
vf_proc = vf_proc_raw - forced_signal_obs

# expand the array to look like a hindcast:
ds_hc_proc = ds_hc_proc.expand_dims(dim={'init_year':len(ds_hc_proc.year)}).assign_coords({'init_year':('init_year',ds_hc_proc.year.values)})


print(xr.corr(ds_hc_proc.isel(init_year=0).mean('member'),vf_proc,dim='year').values)

# corr_raw = xr.corr(ds_hc_raw_proc.isel(init_year=0).mean('member'),vf_proc,dim='year')

# cool plot "visual verification" of ensemble mean:
array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [2,2,2,2],
]

fig,axs = pplt.subplots(array,figwidth=5,figheight=4,suptitle='NAO ensemble mean hindcast')
axs.format(abc=True)
cf = axs[0].pcolormesh(ds_hc_proc.mean('member'),vmin=-20,vmax=20,extend='both')
# cf = ax.contourf(ds_hc_raw_proc.mean('member').scores)
axs[0].colorbar(cf, loc='r', label='')

axs[1].plot(ds_hc_proc.year,ds_hc_proc.isel(init_year=0).mean('member'),color='k',label='CMIP6 EM')
axs[1].plot(vf_proc.year,vf_proc,label='HadSLP')
axs[1].plot(forced_signal_obs.year,forced_signal_obs,label='forced sig obs')
axs[1].plot(forced_signal_model.year,forced_signal_model,label='forced sig mod')
axs[1].pcolormesh(vf_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}))
axs[0].format(
    xlabel='Year', ylabel='Hindcast Initialization Year',
    # xlim=[ds_hc_proc.year.min().values,nao_vf.year.max().values]
    xlim=[1879,2019], ylim=[1925,2019]
)

axs[1].format(ylim=[-80,80],ylabel='NAO index')
axs[1].legend(loc=3,prop={'size':6})

# fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_None_sync_raw.png'.format(**dummy_params),dpi=300,bbox_inches='tight')
# fig.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_None_sync_raw.png'.format(**dummy_params),dpi=300,bbox_inches='tight')

#----------------------------------------------------------------------------------#
# CMIP6 "initialized"/"hindcasts"
#----------------------------------------------------------------------------------#


# reference: what observational dataset to use as a reference:
nao_vf_full = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(hc_param_set['ref_ds']))
nao_vf_full = nao_vf_full.sel(mode=1).squeeze().scores
# process:
if proc_type == 'running_mean':
    vf_proc_raw = nao_vf_full.rolling(year=Nyr).mean('year').dropna('year')
elif proc_type == 'lowpass':
    vf_proc_raw = xr.apply_ufunc(
        butter_lowpass_filter,nao_vf_full,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )
# remove forced signal:
vf_proc = vf_proc_raw - forced_signal_obs

# create the path the hindcast are saved to:
hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)

# load all hindcasts:
hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
ds_hc = xr.open_mfdataset(hindcast_files)

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc['scores'],1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# forced signal is already removed from the hindcasts
ds_hc_ensmean = ds_hc_proc.mean('member') # - forced_signal_model

# cool plot "visual verification" of ensemble mean:
array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
    [1, ],
    [1, ],
    [1, ],
    [2, ],
]


one_init_year = 1950

fig,axs = pplt.subplots(array,figwidth=5,figheight=4,suptitle='NAO ensemble mean hindcast')
axs.format(abc=True)
# cf = axs[0].contourf(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
cf = axs[0].pcolormesh(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
# single_hc_masked = ds_hc_ensmean.where(ds_hc_proc.lead_time!=0).sel(init_year=one_init_year)
# single_hc = ds_hc_ensmean.sel(init_year=one_init_year)
# cf = axs[0].pcolormesh(single_hc_masked.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[1949.5,1950.5])}),vmin=-20,vmax=20,extend='both')

# cf = ax.contourf(ds_hc_ensmean.scores)
axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(vf_proc.year)).lead_time,levels=[-25,-15,-5, 5,15,25],colors='k',linestyles='dotted',labels=True)
# axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(vf_proc.year)).lead_time,levels=[5],colors='k',linestyles='solid',labels=True)
axs[0].colorbar(cf, loc='r', label='')
# axs[0].vlines(one_init_year+np.arange(5,31,10),ds_hc_proc.init_year.min().values,1950,color='k',linestyle='dashed',linewidth=.5)
# axs[0].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[1949.5,1949.5],[1950.5,1950.5],alpha=.2,color='k',zorder=100)

axs[1].plot(vf_proc.year,vf_proc)
# axs[1].plot(single_hc.year,single_hc,color='k')
# lt5_hc = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(5)
# axs[1].plot(lt5_hc.year,lt5_hc,color='k')
# axs[1].vlines(one_init_year+np.arange(5,26,10),-80,80,color='k',linestyle='dashed',linewidth=.5)
# axs[1].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[-80,-80],[80,80],alpha=.2,color='k',zorder=100)
# axs[1].text(one_init_year-hc_param_set['vf_win_len']//2,-75,s='synchronization window',va='bottom',ha='center',fontsize=8)
# axs[1].text(one_init_year+5,80,s='LT5Y',va='bottom',ha='center',fontsize=6)
# axs[1].text(one_init_year+15,80,s='LT15Y',va='bottom',ha='center',fontsize=6)
# axs[1].text(one_init_year+25,80,s='LT25Y',va='bottom',ha='center',fontsize=6)
# axs[1].text(one_init_year+1.5,55,s='"forecast"',va='top',ha='left',fontsize=8,bbox={'boxstyle':'rarrow','alpha':.7,'facecolor':'white'})
# axs[1].arrow(one_init_year,30,10,0,head_width=10,head_length=2)
axs[1].pcolormesh(vf_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}),)

axs[0].format(
    xlabel='Year', ylabel='Hindcast Initialization Year',
    xlim=[ds_hc_proc.year.min().values,vf_proc.year.max().values]
)

axs[1].format(ylim=[-80,80],ylabel='NAO index')

# fig.savefig(fpath/"illustration/hc3.png",dpi=300,bbox_inches='tight')

fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


# Plot the hindcasts for a fixed lead time:
fig_lt = pplt.figure(width=5,height=2,suptitle='NAO prediction')
ax = fig_lt.add_subplot()
corr = []; corr_raw = []
p_val = []; ci_lower = []; ci_upper = []
p_val_raw = []; ci_lower_raw = []; ci_upper_raw = []
for ii,lt in enumerate(lead_times):
    # group hindcasts by lead time:
    lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
    lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
    lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
    
    # Correlation score of the hindcasts at lead time lt:
    year_intersect = np.intersect1d(vf_proc.year,lt_group.year)
    cor,p,ci_l,ci_u = pearsonr_ci(vf_proc.sel(year=year_intersect),lt_group.sel(year=year_intersect),alpha=.1)
    # corr.append(xr.corr(vf_proc,lt_group,dim='year'))
    corr.append(cor); p_val.append(p); ci_lower.append(ci_l); ci_upper.append(ci_u)
    
    # Compute correlation of the raw ensemble over the exact same years where hidncasts are available:
    # cor_raw,p_raw,ci_l_raw,ci_u_raw = pearsonr_ci(vf_proc.sel(year=year_intersect),ds_hc_raw_proc.isel(init_year=0).sel(year=year_intersect).mean('member'),alpha=.1)
    # corr_raw.append(cor_raw); p_val_raw.append(p_raw); ci_lower_raw.append(ci_l_raw); ci_upper_raw.append(ci_u_raw)
    # corr_raw.append(xr.corr(ds_hc_raw_proc.isel(init_year=0).sel(year=lt_group.year).mean('member'),vf_proc.sel(year=np.intersect1d(lt_group.year,vf_proc.year)),dim='year'))
    
    # Plot some of the lead times:
    if ii == 0:
        min_year = lt_group.year[0].values
    if lt in [-25,-15,-5,5,15,25]:
        ax.plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],color='k',label='{:} yr prediction'.format(abs(lt)))

ax.plot(vf_proc.year,vf_proc,label='HadSLP')
ax.format(xlim=[min_year,vf_proc.year.max().values],ylabel='NAO index')

ax.legend(loc='t',prop={'size':8},ncol=4)
fig_lt.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_fixedlead_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

corr_hc = xr.DataArray(
    data = corr,
    coords = {'lead_time':('lead_time',lead_times)}
)
# corr_raw = xr.DataArray(
#     data = corr_raw,
#     coords = {'lead_time':('lead_time',lead_times)}
# )
# corr_hc = xr.concat(corr,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})
# corr_raw = xr.concat(corr_raw,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})

fig_corr = pplt.figure(suptitle='NAO Correlation Skill',figsize=[4,3])
axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='Correlation')
axc.plot(corr_hc,color='C1',label='hindcast')
axc.fill_between(corr_hc.lead_time,ci_lower,ci_upper,alpha=.2,color='C1')
# axc.plot(corr_raw,color='C0',ls='dotted',label='raw')
# axc.fill_between(corr_raw.lead_time,ci_lower_raw,ci_upper_raw,alpha=.2,color='C0')
axc.hlines(0,lead_times[0],lead_times[-1],color='grey',linestyle='dashed')
# axc.legend(loc=4,ncol=1)
fig_corr.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_correlation_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')



# TODO:
# evaluate spread --> RPSS?
# RMSE? then, need to standardize somehow. difficult to say what stdzation would make sense...
# Random Walk test? DelSole & Tippett (2015)
# Retrieve fields such as precip/wind/temp and check the skill globally