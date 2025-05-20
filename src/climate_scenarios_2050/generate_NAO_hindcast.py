# produce hindcast set based on input sync properties (frequency band, window, maximum lag)
# 

import xarray as xr
import numpy as np
from climate_scenarios_2050.utils import preprocess_expand_member_dim
from climate_scenarios_2050.cmip6_nao import butter_bandpass_filter, butter_lowpass_filter
from climate_scenarios_2050.config import *
import pickle

def RMSE_shift(
        obs,
        comp_ensemble_sim,
        shift_ensemble_sim,
        end_year,
        years_full,
        per_len=20,
        max_lag=6,
    ):
        
    check_period = [end_year - per_len, end_year]
    lags = np.arange(-max_lag,max_lag+.1,1,dtype=int)

    obs_window = obs.sel(year=slice(*check_period))

    rmse = []
    for lag in lags:
        RMSE = ((obs_window.values - comp_ensemble_sim.sel(year=slice(check_period[0]+lag,check_period[1]+lag)).values)**2).mean(axis=1)**.5
        # compute RMSE:
        rmse.append(RMSE)

    rmse_min = np.array(rmse).argmin(axis=0)
    best_lags = lags[rmse_min]

    ensemble_shifted = []
    for ii,mem in enumerate(comp_ensemble_sim.member.values):
        mem_shift = shift_ensemble_sim.sel(member=mem).sel(year = slice(years_full.min() + best_lags[ii],years_full.max() + best_lags[ii]))
        mem_shift = mem_shift.assign_coords(year=mem_shift.year - best_lags[ii])
        
        ensemble_shifted.append(mem_shift.sel(year=slice(years_full.min()+max_lag,years_full.max()-max_lag)))

    ensemble_shifted = xr.concat(ensemble_shifted,dim='member')
    
    return ensemble_shifted

def load_model_ensemble_NAO(subsampling,scen,ref_ds):
    """
    load a full (1850 - 2100) time series of yearly NAO index, using members from the 
    scenario `scen`. NAO is the projection of the SLP from the models onto the interannual EOF1
    of reference dataset `ref_ds`
    `subsampling` should be a list of CMIP6 models to use out of the available ones
    """

    base_path_search = proj_base/'data/CMIP6_processed'

    # load unique model identifier for naming ensemble members:
    with open(model_id_file,'rb') as f:
        model_id = pickle.load(f)

    # get a list of all available models:
    models = []
    for exp in ['historical',scen]:
        # Collect files
        file_list = sorted(list(base_path_search.glob(f"{exp}/*/EOF_model/*psl_EOF_{ref_ds}*index*.nc")))
        # Remove expty ones:
        file_list = [filename for filename in file_list if os.path.getsize(filename) > 0]
        # get model name:
        models.append([filename.stem.split('_')[0] for filename in file_list])
    # only select those member for which both historical and scenario simulations exist:
    av_models = sorted(list(set(models[0]).intersection(models[1])))

    if subsampling is None:
        model_selection = av_models
    elif 'varsplit' in subsampling:
        obs_split = xr.open_dataarray(proj_base/f'data/{ref_ds}/{subsampling}_{ref_ds}_psl_EOF_full_index_DJFM_1850-2014_1.5deg.nc').sel(mode=1)
        # subselect the models based on the amount of low-frequency NAO variance:
        model_selection = []
        for model in av_models:
            vsplit = xr.open_dataarray(base_path_search/f'historical/{model}/EOF_model/{subsampling}_{model}_psl_EOF_{ref_ds}_index_DJFM_1850-2014_1.5deg.nc').sel(mode=1)
            # if the ratio in the observations is more than 10% larger than in the simulations, drop the model
            # print('{:} obs: {:1.2f} sim: {:1.2f}'.format(model,obs_split.values, vsplit.values))
            criterion = (obs_split - vsplit) < .1
            if criterion:
                model_selection.append(model)

    else:
        print('{:} subsampling not implemented yet'.format(subsampling))
        return

    index_series = []
    for ii,model in enumerate(model_selection):
        filename_hist = list(base_path_search.glob(f"historical/{model}/EOF_model/{model}*psl_EOF_{ref_ds}*index*.nc"))[0]
        try:
            filename_scen = list(base_path_search.glob(f"{scen}/{model}/EOF_model/{model}*psl_EOF_{ref_ds}*index*.nc"))[0]
        except:
            continue
        ds_hist = preprocess_expand_member_dim(xr.open_dataset(filename_hist))
        ds_scen = preprocess_expand_member_dim(xr.open_dataset(filename_scen))

        common_members = np.intersect1d(ds_hist['member'], ds_scen['member'])

        ds = xr.concat([ds_hist.sel(member=common_members), ds_scen.sel(member=common_members)], dim='year')
        
        ext_mems = [model_id[model]+mem for mem in ds.member.values]

        index_series.append(ds.assign_coords(member=ext_mems))
            
    return xr.concat(index_series,dim='member')


if __name__ == '__main__':

    # define the parameter sets to try 
    parameter_sets = [
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        {'subsampling' : 'varsplit_7y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
    ]

    for hc_param_set in parameter_sets:
        # create the path to save the hindcast to:
        hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
        hc_path.mkdir(parents=True,exist_ok=True)

        # take inverse of periods to get cutoff frequencies for bandpass filter:
        lowcut_freq = 1/hc_param_set['cutoff_period_long']
        highcut_freq = 1/hc_param_set['cutoff_period_short']
        
        # load the full model ensemble NAO (need to know scenario and reference/origin of pattern):
        NAO_model_ensemble = load_model_ensemble_NAO(hc_param_set['subsampling'],hc_param_set['scen'],hc_param_set['ref_ds'])
        
        # Ensure year is sorted and numeric
        years_full = np.arange(NAO_model_ensemble.year.min(), NAO_model_ensemble.year.max() + 1)
        # Interpolate to get a value for 2014
        NAO_model_ensemble = NAO_model_ensemble.reindex(year=years_full).interpolate_na(dim='year', method='linear')

        # band-pass filter the NAO ensemble for synchronization:
        ensemble_bp = xr.apply_ufunc(
            butter_bandpass_filter,NAO_model_ensemble['scores'],lowcut_freq,highcut_freq,1,
            input_core_dims=[['year'],[],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )

        # Reindex with interpolation (fill missing 2014)
        # ensemble_bp = ensemble_bp.reindex(year=years_full).interpolate_na(dim='year', method='linear')

        nao_vf = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(hc_param_set['ref_ds']))
        nao_vf = nao_vf.sel(mode=1).squeeze().scores

        nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, nao_vf, lowcut_freq, highcut_freq, 1)

        # generate list of possible initialization years (limited by sync'ing window length):
        init_years = np.arange(nao_hadslp_bp.year.values.min()+hc_param_set['vf_win_len']+hc_param_set['mxlags'],nao_hadslp_bp.year.values.max()+1,1)

        print('generating {} hindcasts'.format(len(init_years)))

        # Shift/sync:
        for init_year in init_years:
            print(init_year)
        # init_year = 2015 #nao_hadslp.year.max().values
            NAO_sync_initialized = RMSE_shift(nao_hadslp_bp,ensemble_bp,NAO_model_ensemble['scores'],init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'])
            # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
            lt_coord = NAO_sync_initialized.year.values - init_year
            lt_coord[np.where(lt_coord<0)] = 0
            # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
            lt_coord_neg = NAO_sync_initialized.year.values - (init_year - hc_param_set['vf_win_len']-1)
            lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
            NAO_sync_initialized = NAO_sync_initialized.assign_coords({'lead_time':('year',lt_coord)}).drop('mode')
            NAO_sync_initialized = NAO_sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
            # save hindcast:
            hc_name = 'hc_{}.nc'.format(init_year)
            NAO_sync_initialized.to_netcdf(hc_path/hc_name)