import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
import proplot as pplt
from pathlib import Path
from climate_scenarios_2050.config import proj_base
from climate_scenarios_2050.generate_NAO_hindcast import load_model_ensemble_NAO
from climate_scenarios_2050.filters import butter_bandpass_filter, butter_lowpass_filter


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

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

parameter_sets = [
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        {'subsampling' : 'varsplit_7y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
    ]

# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 8

# lead_times: what lead times to verify
lead_times = np.arange(0,36,1)
lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

linestyle = {5:'solid',15:'dashed',25:'dotted',35:'solid',45:'dashed'}


#----------------------------------------------------------------------------------#
# Raw CMIP6 ensemble
#----------------------------------------------------------------------------------#

#-----Create a "baseline" hindcast that just takes the CMIP6 ensemble without syncing
# reference: what observational dataset to use as a reference:

dummy_params = parameter_sets[0]

nao_vf = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(dummy_params['ref_ds']))
nao_vf = nao_vf.sel(mode=1).squeeze().scores
# process:
if proc_type == 'running_mean':
    vf_proc = nao_vf.rolling(year=Nyr).mean('year').dropna('year')
elif proc_type == 'lowpass':
    vf_proc = xr.apply_ufunc(
        butter_lowpass_filter,nao_vf,1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )

# load all "non-initialized" hindcasts:
ds_hc_raw = load_model_ensemble_NAO(None,dummy_params['scen'],dummy_params['ref_ds'])

# process (average/filter):
if proc_type == 'running_mean':
    ds_hc_raw_proc = ds_hc_raw['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
elif proc_type == 'lowpass':
    ds_hc_raw_proc = xr.apply_ufunc(
        butter_lowpass_filter,ds_hc_raw['scores'],1/7,1,
        input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    ).compute()

# expand the array to look like a hindcast:
ds_hc_raw_proc = ds_hc_raw_proc.expand_dims(dim={'init_year':len(ds_hc_raw_proc.year)}).assign_coords({'init_year':('init_year',ds_hc_raw_proc.year.values)})
# interpolate 2014:
years_full = np.arange(ds_hc_raw_proc.year.min(), ds_hc_raw_proc.year.max() + 1)
ds_hc_raw_proc = ds_hc_raw_proc.reindex(year=years_full).interpolate_na(dim='year', method='linear')

print(xr.corr(ds_hc_raw_proc.isel(init_year=0).mean('member'),vf_proc,dim='year').values)

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
cf = axs[0].contourf(ds_hc_raw_proc.mean('member'),vmin=-20,vmax=20,extend='both')
# cf = ax.contourf(ds_hc_raw_proc.mean('member').scores)
axs[0].colorbar(cf, loc='r', label='')

axs[1].plot(ds_hc_raw_proc.year,ds_hc_raw_proc.isel(init_year=0).mean('member'),color='k',label='CMIP6 EM')
axs[1].plot(vf_proc.year,vf_proc,label='HadSLP')
axs[1].pcolormesh(vf_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}))
axs[1].legend(loc=1,prop={'size':6})
axs[0].format(
    xlabel='Year', ylabel='Hindcast Initialization Year',
    xlim=[ds_hc_raw_proc.year.min().values,nao_vf.year.max().values]
)

axs[1].format(ylabel='NAO index')

# fig.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_{subsampling}_sync_raw.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

#----------------------------------------------------------------------------------#
# CMIP6 "initialized"/"hindcasts"
#----------------------------------------------------------------------------------#

for hc_param_set in parameter_sets:

    # reference: what observational dataset to use as a reference:
    nao_vf = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(hc_param_set['ref_ds']))
    nao_vf = nao_vf.sel(mode=1).squeeze().scores
    # process:
    if proc_type == 'running_mean':
        vf_proc = nao_vf.rolling(year=Nyr).mean('year').dropna('year')
    elif proc_type == 'lowpass':
        vf_proc = xr.apply_ufunc(
            butter_lowpass_filter,nao_vf,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )
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

    ds_hc_ensmean = ds_hc_proc.mean('member')

    # cool plot "visual verification" of ensemble mean:
    array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
        [1, ],
        [1, ],
        [1, ],
        [2, ],
    ]


    fig,axs = pplt.subplots(array,figwidth=5,figheight=4,suptitle='NAO ensemble mean hindcast')
    axs.format(abc=True)
    cf = axs[0].contourf(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
    # cf = ax.contourf(ds_hc_ensmean.scores)
    axs[0].contour(ds_hc_proc.lead_time,levels=[-25,-15,-5, 5,15,25],colors='k',linestyles='dotted',labels=True)
    axs[0].colorbar(cf, loc='r', label='')

    axs[1].plot(vf_proc.year,vf_proc)
    axs[1].pcolormesh(vf_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}),)
    
    axs[0].format(
        xlabel='Year', ylabel='Hindcast Initialization Year',
        xlim=[ds_hc_proc.year.min().values,nao_vf.year.max().values]
    )
    
    axs[1].format(ylabel='NAO index')

    fig.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


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
        cor_raw,p_raw,ci_l_raw,ci_u_raw = pearsonr_ci(vf_proc.sel(year=year_intersect),ds_hc_raw_proc.isel(init_year=0).sel(year=year_intersect).mean('member'),alpha=.1)
        corr_raw.append(cor_raw); p_val_raw.append(p_raw); ci_lower_raw.append(ci_l_raw); ci_upper_raw.append(ci_u_raw)
        # corr_raw.append(xr.corr(ds_hc_raw_proc.isel(init_year=0).sel(year=lt_group.year).mean('member'),vf_proc.sel(year=np.intersect1d(lt_group.year,vf_proc.year)),dim='year'))
        
        # Plot some of the lead times:
        if ii == 0:
            min_year = lt_group.year[0].values
        if lt in [-25,-15,-5,5,15,25]:
            ax.plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],color='k',label='{:} yr prediction'.format(abs(lt)))
    
    ax.plot(vf_proc.year,vf_proc,label='HadSLP')
    ax.format(xlim=[min_year,nao_vf.year.max().values],ylabel='NAO index')

    ax.legend(loc='t',prop={'size':8},ncol=4)
    fig_lt.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_fixedlead_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

    corr_hc = xr.DataArray(
        data = corr,
        coords = {'lead_time':('lead_time',lead_times)}
    )
    corr_raw = xr.DataArray(
        data = corr_raw,
        coords = {'lead_time':('lead_time',lead_times)}
    )
    # corr_hc = xr.concat(corr,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})
    # corr_raw = xr.concat(corr_raw,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})

    fig_corr = pplt.figure(suptitle='NAO Correlation Skill')
    axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='Correlation')
    axc.plot(corr_hc,color='C1',label='hindcast')
    axc.fill_between(corr_hc.lead_time,ci_lower,ci_upper,alpha=.2,color='C1')
    axc.plot(corr_raw,color='C0',ls='dotted',label='raw')
    axc.fill_between(corr_raw.lead_time,ci_lower_raw,ci_upper_raw,alpha=.2,color='C0')
    axc.legend(loc=4,ncol=1)
    fig_corr.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_correlation_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


# TODO:
# evaluate spread --> RPSS?
# RMSE? then, need to standardize somehow. difficult to say what stdzation would make sense...
