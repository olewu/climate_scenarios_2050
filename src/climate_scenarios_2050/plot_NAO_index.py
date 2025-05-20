# Project the models' SLP anomalies (wrt to the historical period) onto the NAO pattern from the observations

import xarray as xr
import numpy as np
import proplot as pplt
import matplotlib.cm as cm
from climate_scenarios_2050.config import *
from climate_scenarios_2050.cmip6_nao import butter_bandpass_filter, butter_lowpass_filter
from climate_scenarios_2050.generate_NAO_hindcast import RMSE_shift
from climate_scenarios_2050.utils import preprocess_expand_member_dim


# Functions
def plot_time_series(percentiles, ensemble, model, model_index, obs, exp,
                     ref_name='HadSLP', title='', filename='fig.png',
                     xlim=(1860,2080), vline=2015, highlight_range=np.arange(2040,2051)):
    fig = pplt.figure(figsize=(7, 3))

    ax = fig.subplot(xlabel='Year', ylabel='NAO index',title=title)
    ax.format(xlim=xlim, xminorlocator=mw_len, xlocator=20)

    ax.plot(percentiles.year,percentiles.sel(quantile=0.5), color='C1', alpha=1, linewidth=1, label=f'_{exp} median')
    ax.plot(percentiles.year,percentiles.sel(quantile=1), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} max')
    ax.plot(percentiles.year,percentiles.sel(quantile=0), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} min')
    ax.fill_between(percentiles.year, percentiles.sel(quantile=0.05), percentiles.sel(quantile=0.95), color='C1', alpha=0.2, label=f'_{exp} 90%')

    ax.axvline(vline , color='r', ls='dashed', alpha=0.5)
    ax.fill_between(percentiles.year, 0, 1, where = percentiles.year.isin(highlight_range),
                    color='r', ls='dashed', alpha=0.2,label='Target period', transform=ax.get_xaxis_transform())

    ax.plot(obs.year, obs, color='k', alpha=1, linewidth=2, label=ref_name)    

    # plt.plot(nao_20cr_filt.year, nao_20cr_filt.scores, color='k', ls='dashed', alpha=1, linewidth=1)
    members = [mem for mem in ensemble.member.values if f'm{model_index}r' in mem]
    ens_lines = ax.plot(ensemble.year,ensemble.sel(member=members).T, color=model_to_color[model], alpha=0.1, linewidth=0.75);
    for line in ens_lines:
        label = line.get_label()
        line.set_label(f'_{label}')
    ax.plot(ensemble.year,ensemble.sel(member=members).mean('member'), color=model_to_color[model], alpha=0.5, linewidth=1, label=f'{model} mean');

    ax.legend(loc='upper left', fontsize=8, frameon=True)

    fig.savefig(base_path/f'www/scenarios_2050/figures/NAO_time_series/{filename}', dpi=300, bbox_inches='tight')



base_path_search = proj_base/'data/CMIP6_processed'

ref_ds = 'HadSLP'

nao_hadslp = xr.open_dataset(proj_base/f'data/{ref_ds}/{ref_ds}_psl_EOF_full_index_DJFM_1850-2014_1.5deg.nc')
nao_hadslp = nao_hadslp.sel(mode=1).squeeze().scores


cutoff_period_short = 7
cutoff_period_long = 12
lowcut = 1/cutoff_period_long
highcut = 1/cutoff_period_short

mw_len = 8
vf_win_len = 20
# loop over this to generate the different hindcasts:
final_window_year = nao_hadslp.year.max().values
mxlags = 6

# verify not only forward but also backward!
# the problem is anyway that we only have a historical run
# and no "real" hindcast. We also need to keep in mind that we
# know the historical forcing but we don't know the future forcing!!
# So any thus estimated "skill" is only an upper bound!

# based on the above settings, create 30 year hindcasts that can be evaluated
# test different parameter combinations for comparison

for scen in ['ssp245','ssp370']:
    for filt in ['lp','rm']:
        
        models = []
        for exp in ['historical',scen]:
            # Collect files
            file_list = sorted(list(base_path_search.glob(f"{exp}/*/EOF_model/*psl_EOF_{ref_ds}*index*.nc")))
            # Remove expty ones:
            file_list = [filename for filename in file_list if os.path.getsize(filename) > 0]
            # get model name:
            models.append([filename.stem.split('_')[0] for filename in file_list])

        av_models = sorted(list(set(models[0]).intersection(models[1])))
        cmap = cm.get_cmap("tab20", len(av_models))
        model_to_color = {model: cmap(i) for i, model in enumerate(av_models)}

        index_series = []
        index_series_bp = []
        index_series_rm = []
        for ii,model in enumerate(av_models):
            filename_hist = list(base_path_search.glob(f"historical/{model}/EOF_model/*psl_EOF_{ref_ds}*index*.nc"))[0]
            try:
                filename_scen = list(base_path_search.glob(f"{scen}/{model}/EOF_model/*psl_EOF_{ref_ds}*index*.nc"))[0]
            except:
                continue
            ds_hist = preprocess_expand_member_dim(xr.open_dataset(filename_hist))
            ds_scen = preprocess_expand_member_dim(xr.open_dataset(filename_scen))

            common_members = np.intersect1d(ds_hist['member'], ds_scen['member'])

            ds = xr.concat([ds_hist.sel(member=common_members), ds_scen.sel(member=common_members)], dim='year')
            
            ext_mems = ['m'+str(ii)+mem for mem in ds.member.values]

            ds = ds.assign_coords(member=ext_mems)

            for member_idx in range(ds.dims['member']):
                ts_bp = xr.apply_ufunc(butter_bandpass_filter,ds['scores'].isel(member=member_idx),lowcut,highcut,1)
                ts_lp = xr.apply_ufunc(butter_lowpass_filter,ds['scores'].isel(member=member_idx),highcut,1)
                ts_rm = ds['scores'].isel(member=member_idx).rolling(year=mw_len, center=True).mean().dropna('year')
                index_series.append(ts_lp)
                index_series_bp.append(ts_bp)
                index_series_rm.append(ts_rm)

        ensemble_lp = xr.concat(index_series, dim='member')  # shape: (ensemble, time)
        ensemble_bp = xr.concat(index_series_bp, dim='member')  # shape: (ensemble, time)
        ensemble_rm = xr.concat(index_series_rm, dim='member')  # shape: (ensemble, time)

        # Ensure year is sorted and numeric
        years_full = np.arange(ensemble_lp.year.min(), ensemble_lp.year.max() + 1)

        # Reindex with interpolation
        ensemble_lp = ensemble_lp.reindex(year=years_full).interpolate_na(dim='year', method='linear')
        ensemble_bp = ensemble_bp.reindex(year=years_full).interpolate_na(dim='year', method='linear')
        ensemble_rm = ensemble_rm.reindex(year=years_full).interpolate_na(dim='year', method='linear')

        percentiles_lp = ensemble_lp.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')
        percentiles_bp = ensemble_bp.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')
        percentiles_rm = ensemble_rm.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')

        nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, nao_hadslp, lowcut, highcut, 1)
        nao_hadslp_lp = xr.apply_ufunc(butter_lowpass_filter, nao_hadslp, highcut, 1)
        nao_hadslp_rm = nao_hadslp.rolling(year=mw_len, center=True).mean().dropna('year')

        nmems = ensemble_lp.member.size
        nmod = len(av_models)
        # Plotting
        mm=12
        if filt == 'rm':
            plot_time_series(percentiles_rm, ensemble_rm, av_models[mm], mm, nao_hadslp_rm, scen, ref_name=ref_ds,
                            title=f'NAO index in {nmems} CMIP6 ({scen}) members ({nmod} models)', 
                            filename=f'NAO_index_{mw_len}yrm_{scen}_bp{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}y_vfwin_mxlag{mxlags}.png',)
        elif filt == 'lp':
            plot_time_series(percentiles_lp, ensemble_lp, av_models[mm], mm, nao_hadslp_lp, scen, ref_name=ref_ds,
                            title=f'NAO index in {nmems} CMIP6 ({scen}) members ({nmod} models)', 
                            filename=f'NAO_index_lp_{scen}_bp{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}y_vfwin_mxlag{mxlags}.png',)

        # Shift/sync:

        ensemble_bp_shifted = RMSE_shift(nao_hadslp_bp,ensemble_bp,ensemble_bp,final_window_year,years_full,per_len=vf_win_len,max_lag=mxlags)
        ensemble_lp_shifted = RMSE_shift(nao_hadslp_bp,ensemble_bp,ensemble_lp,final_window_year,years_full,per_len=vf_win_len,max_lag=mxlags)
        ensemble_rm_shifted = RMSE_shift(nao_hadslp_bp,ensemble_bp,ensemble_rm,final_window_year,years_full,per_len=vf_win_len,max_lag=mxlags)

        percentiles_lp_shifted = ensemble_lp_shifted.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')
        percentiles_rm_shifted = ensemble_rm_shifted.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')

        # Plotting
        if filt == 'rm':
            plot_time_series(percentiles_rm_shifted, ensemble_rm_shifted, av_models[mm], mm, nao_hadslp_rm, scen, ref_name=ref_ds,
                            title=f'NAO index in CMIP6 ({scen}) models w/ optimal lag (RMSE)', 
                            filename=f'NAO_index_{mw_len}yrm_{scen}_bp{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}y_vfwin_mxlag{mxlags}_RMSE_opt_lagged.png',)
        elif filt == 'lp':
            plot_time_series(percentiles_lp_shifted, ensemble_lp_shifted, av_models[mm], mm, nao_hadslp_lp, scen, ref_name=ref_ds,
                            title=f'NAO index in CMIP6 ({scen}) models w/ optimal lag (RMSE)', 
                            filename=f'NAO_index_lp_{scen}_bp{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}y_vfwin_mxlag{mxlags}_RMSE_opt_lagged.png',)


# fig = pplt.figure(figsize=(7, 3))

# ax = fig.subplot(xlabel='Year', ylabel='NAO index',title=f'NAO index in CMIP6 ({scen}) models')
# ax.format(xlim=(1860,2080), xminorlocator=10, xlocator=20)

# ax.plot(percentiles_lp.year,percentiles_lp.sel(quantile=0.5), color='C1', alpha=1, linewidth=1, label=f'_{exp} median')
# ax.plot(percentiles_lp.year,percentiles_lp.sel(quantile=1), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} max')
# ax.plot(percentiles_lp.year,percentiles_lp.sel(quantile=0), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} min')
# ax.fill_between(percentiles_lp.year, percentiles_lp.sel(quantile=0.05), percentiles_lp.sel(quantile=0.95), color='C1', alpha=0.2, label=f'_{exp} 90%')

# ax.axvline(2015 , color='r', ls='dashed', alpha=0.5)
# ax.fill_between(percentiles_lp.year, -100, 100, where = percentiles_lp.year.isin(np.arange(2040,2051)),
#                 color='r', ls='dashed', alpha=0.2,label='Target period', transform=ax.get_xaxis_transform())

# ax.plot(nao_hadslp_lp.year, nao_hadslp_lp, color='k', alpha=1, linewidth=2, label='HadSLP')    

# # plt.plot(nao_20cr_filt.year, nao_20cr_filt.scores, color='k', ls='dashed', alpha=1, linewidth=1)    
# mm = 16
# model = av_models[mm]
# members = [mem for mem in ensemble_lp.member.values if f'm{mm}r' in mem]
# ens_lines = ax.plot(ensemble_lp.year,ensemble_lp.sel(member=members).T, color=model_to_color[model], alpha=0.1, linewidth=0.75);
# for line in ens_lines:
#     label = line.get_label()
#     line.set_label(f'_{label}')
# ax.plot(ensemble_lp.year,ensemble_lp.sel(member=members).mean('member'), color=model_to_color[model], alpha=0.5, linewidth=1, label=f'{model} mean');

# ax.legend(loc='upper left', fontsize=8, frameon=True)

# fig.savefig(base_path/f'www/scenarios_2050/figures/NAO_time_series/NAO_index_{scen}.png', dpi=300, bbox_inches='tight')


# To synchoronize the NAO decadal variability, take the latest 20-year window of HadSLP
# and compute the RMSE compared to each CMIP6 member, do this for different lags in (-5,5) years
# and pick the lag that minimizes the RMSE for each member individually
# per_len = 20
# check_period = [nao_hadslp_bp.year.max() - per_len,nao_hadslp_bp.year.max()]# end year nee
# max_lag = 6
# lags = np.arange(-max_lag,max_lag+.1,1)
# NAO_window = nao_hadslp_bp.sel(year=slice(*check_period))

# rmse = []
# for lag in lags:
#     RMSE = ((NAO_window.values - ensemble_bp.sel(year=slice(check_period[0]+lag,check_period[1]+lag)).values)**2).mean(axis=1)**.5
#     # compute RMSE:
#     rmse.append(RMSE)

# rmse_min = np.array(rmse).argmin(axis=0)
# best_lags = lags[rmse_min]

# shifted_years = np.array([years_full + lag for lag in best_lags])  # shape (n_member,n_year)
# ensemble_bp_shifted = []
# ensemble_lp_shifted = []
# ensemble_rm_shifted = []
# for ii,mem in enumerate(ensemble_bp.member.values):
#     mem_shift = ensemble_bp.sel(member=mem).sel(year = slice(years_full.min() + best_lags[ii],years_full.max() + best_lags[ii]))
#     mem_shift = mem_shift.assign_coords(year=mem_shift.year - best_lags[ii])
    
#     ensemble_bp_shifted.append(mem_shift.sel(year=slice(years_full.min()+max_lag,years_full.max()-max_lag)))

#     mem_shift_lp = ensemble_lp.sel(member=mem).sel(year = slice(years_full.min() + best_lags[ii],years_full.max() + best_lags[ii]))
#     mem_shift_lp = mem_shift_lp.assign_coords(year=mem_shift_lp.year - best_lags[ii])
    
#     ensemble_lp_shifted.append(mem_shift_lp.sel(year=slice(years_full.min()+max_lag,years_full.max()-max_lag)))

#     mem_shift_rm = ensemble_rm.sel(member=mem).sel(year = slice(years_full.min() + best_lags[ii],years_full.max() + best_lags[ii]))
#     mem_shift_rm = mem_shift_rm.assign_coords(year=mem_shift_rm.year - best_lags[ii])
    
#     ensemble_rm_shifted.append(mem_shift_rm.sel(year=slice(years_full.min()+max_lag,years_full.max()-max_lag)))

# ensemble_bp_shifted = xr.concat(ensemble_bp_shifted,dim='member')
# ensemble_lp_shifted = xr.concat(ensemble_lp_shifted,dim='member')
# ensemble_rm_shifted = xr.concat(ensemble_rm_shifted,dim='member')

# percentiles_lp_shifted = ensemble_lp_shifted.quantile([0,0.05,0.1,.25,0.5,.75,0.9,0.95,1], dim='member')

# fig = pplt.figure(figsize=(7, 3))

# ax = fig.subplot(xlabel='Year', ylabel='NAO index',title=f'NAO index in CMIP6 ({scen}) models w/ optimal lag')
# ax.format(xlim=(1860,2080), xminorlocator=10, xlocator=20)

# ax.plot(percentiles_lp_shifted.year,percentiles_lp_shifted.sel(quantile=0.5), color='C1', alpha=1, linewidth=1, label=f'_{exp} median')
# ax.plot(percentiles_lp_shifted.year,percentiles_lp_shifted.sel(quantile=1), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} max')
# ax.plot(percentiles_lp_shifted.year,percentiles_lp_shifted.sel(quantile=0), color='C1', ls = 'dotted', alpha=1, linewidth=1, label=f'_{exp} min')
# ax.fill_between(percentiles_lp_shifted.year, percentiles_lp_shifted.sel(quantile=0.05), percentiles_lp_shifted.sel(quantile=0.95), color='C1', alpha=0.2, label=f'_{exp} 90%')

# ax.axvline(2015 , color='r', ls='dashed', alpha=0.5)
# ax.fill_between(percentiles_lp_shifted.year, -100, 100, where = percentiles_lp_shifted.year.isin(np.arange(2040,2051)),
#                 color='r', ls='dashed', alpha=0.2,label='Target period', transform=ax.get_xaxis_transform())

# ax.plot(nao_hadslp_lp.year, nao_hadslp_lp, color='k', alpha=1, linewidth=2, label='HadSLP')    

# # plt.plot(nao_20cr_filt.year, nao_20cr_filt.scores, color='k', ls='dashed', alpha=1, linewidth=1)    
# ii = 0
# model = av_models[ii]
# members = [mem for mem in ensemble_lp_shifted.member.values if f'm{ii}r' in mem]
# ens_lines = ax.plot(ensemble_lp_shifted.year,ensemble_lp_shifted.sel(member=members).T, color=model_to_color[model], alpha=0.1, linewidth=0.75);
# for line in ens_lines:
#     label = line.get_label()
#     line.set_label(f'_{label}')
# ax.plot(ensemble_lp_shifted.year,ensemble_lp_shifted.sel(member=members).mean('member'), color=model_to_color[model], alpha=0.5, linewidth=1, label=f'{model} mean');

# ax.legend(loc='upper left', fontsize=8, frameon=True)

# fig.savefig(base_path/f'www/scenarios_2050/figures/NAO_time_series/NAO_index_{scen}_RMSE_opt_lagged.png', dpi=300, bbox_inches='tight')

# Lagged correlation between "_bandpass_" filtered NAO index in obs and CMIP6, note that this eliminates existing trends, too!
# lags = np.arange(0,11,1)
# lagged_corr = []
# hist_ens_subset = ensemble_bp.sel(year=ensemble_bp.year.isin(nao_hadslp_lp.year))
# # match the years of
# for lg in lags:
#     if lg == 0:
#         obs = nao_hadslp_lp.values
#         sim = hist_ens_subset.values
#     else:
#         obs = nao_hadslp_lp.values[lg:]
#         sim = hist_ens_subset.values[:,:-lg]
#     X = np.vstack(([obs,sim]))
#     corr_mat = np.corrcoef(X)
#     lagged_corr.append(corr_mat[0,1:])

# lagged_corr = np.array(lagged_corr)
# # put back into a xr.dataarray
# lagged_corr = xr.DataArray(lagged_corr, dims=['lag', 'member'], coords=[lags, hist_ens_subset.member.values])

# 

# for exp in ['historical','ssp370']:
#     # Load files
#     file_list = sorted(list(base_path_search.glob(f"{exp}/*/EOF_model/*psl_EOF_{ref_ds}*index*.nc")))
#     file_list = [filename for filename in file_list if filename.stem.split('_')[0] in av_models]
#     n_mem = 0
#     index_series = []
#     for filename in file_list:
    
#         ds = preprocess(xr.open_dataset(filename))
#         try:
#             n_mem += ds.sizes['member']
#         except:
#             n_mem += 1
#         model = filename.stem.split('_')[0]

#         ext_mems = [mem for mem in ds.member.values]


#         for member_idx in range(ds.dims['member']):
#             ts = xr.apply_ufunc(butter_bandpass_filter,ds['scores'].isel(member=member_idx),lowcut,highcut,1)
#             ts.member = ts.member.to_list()[0]
#             index_series.append(ts)

#     ensemble = xr.concat(index_series, dim='ensemble')  # shape: (ensemble, time)

#     percentiles.append(ensemble.quantile([0,0.1, 0.5, 0.9,1], dim='ensemble'))

# percentiles = xr.concat(percentiles, dim='year')
