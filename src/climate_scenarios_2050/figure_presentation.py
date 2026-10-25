
import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
import proplot as pplt
from pathlib import Path
from climate_scenarios_2050.config import proj_base
from climate_scenarios_2050.generate_NAO_hindcast import load_model_ensemble_NAO
from climate_scenarios_2050.filters import butter_bandpass_filter, butter_lowpass_filter

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures/illustration')

parameter_set = {
    'subsampling' : None,
    'ref_ds' : 'HadSLP', 
    'scen' : 'ssp245',
    'cutoff_period_short' : 20,
    'cutoff_period_long' : 50,
    'vf_win_len' : 50,
    'mxlags' : 25,
}

# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 8

# lead_times: what lead times to verify
lead_times = np.arange(0,36,1)
lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

linestyle = {5:'solid',15:'dashed',25:'dotted'}


#----------------------------------------------------------------------------------#
# Raw CMIP6 ensemble
#----------------------------------------------------------------------------------#

#-----Create a "baseline" hindcast that just takes the CMIP6 ensemble without syncing
# reference: what observational dataset to use as a reference:

# load all "non-initialized" hindcasts:
ds_hc_raw_full = load_model_ensemble_NAO(None,parameter_set['scen'],parameter_set['ref_ds']).scores

# interpolate 2014:
years_full = np.arange(ds_hc_raw_full.year.min(), ds_hc_raw_full.year.max() + 1)
ds_hc_raw_full = ds_hc_raw_full.reindex(year=years_full).interpolate_na(dim='year', method='linear')

# remove forced signal:
forced_signal = ds_hc_raw_full.mean('member')
ds_hc_raw = ds_hc_raw_full - forced_signal

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_raw_proc = ds_hc_raw.rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_raw_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc_raw,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# expand the array to look like a hindcast:
ds_hc_raw_proc = ds_hc_raw_proc.expand_dims(dim={'init_year':len(ds_hc_raw_proc.year)}).assign_coords({'init_year':('init_year',ds_hc_raw_proc.year.values)})


# Observations:
nao_vf_full = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(parameter_set['ref_ds']))
nao_vf_full = nao_vf_full.sel(mode=1).squeeze().scores
# remove forced signal from model:
nao_vf = nao_vf_full - forced_signal

# process:
if proc_type == 'running_mean':
    vf_proc = nao_vf.rolling(year=Nyr).mean('year').dropna('year')
elif proc_type == 'lowpass':
    vf_proc = xr.apply_ufunc(
        butter_lowpass_filter,nao_vf,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )


print(xr.corr(ds_hc_raw_proc.isel(init_year=0).mean('member'),vf_proc,dim='year').values)

#----------------------------------------------------------------------------------#
# CMIP6 "initialized"/"hindcasts"
#----------------------------------------------------------------------------------#

# create the path the hindcast are saved to:
hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**parameter_set)

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

ds_hc_ensmean = ds_hc_proc.mean('member')


# collect predictions by lead time
lt_pred = {}
for ii,lt in enumerate(lead_times):
    # group hindcasts by lead time:
    lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
    lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
    lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
    
    # Plot some of the lead times:
    if ii == 0:
        min_year = lt_group.year[0].values
    lt_pred[lt] = lt_group

# Make a time series plot to illustrate the NAO prediction

fig = pplt.figure(width=6,height=2.5,suptitle='NAO prediction')
ax = fig.add_subplot(
    xlabel='Year', ylabel='NAO (observed)',
    xlim=[min_year,nao_vf.year.max().values],
    ylim=[-75,75]
)
axt = ax.twinx(ylabel='NAO (model mean)')
axt.set_ylim([-30,30])
ax.plot(vf_proc.year,vf_proc,label='observations',color='C0',lw=2)
axt.plot(ds_hc_raw_proc.year,ds_hc_raw_proc.isel(init_year=0).mean('member'),color='C1',lw=1,label='CMIP6 ensemble')
for lt_key, ls in linestyle.items():
    axt.plot(lt_pred[lt_key].year,lt_pred[lt_key],color='k',lw=2,linestyle=ls,label=f'{lt_key} year prediction')

handlest, labelst = axt.get_legend_handles_labels()
handles, labels = ax.get_legend_handles_labels()
handles, labels = ax.get_legend_handles_labels()
handlest.append(handles[0]); labelst.append(labels[0])
order = [1,2,3,4,0]

#add legend to plot
ax.legend([handlest[idx] for idx in order],[labelst[idx] for idx in order],loc='b',ncol=3)
fig.savefig(fpath/'NAO_prediction_20-50yr_sync_CMIP6_HadSLP.eps',bbox_inches='tight')