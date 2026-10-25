# produce hindcast set based on input sync properties (frequency band, window, maximum lag)
# 

import xarray as xr
import numpy as np
from climate_scenarios_2050.utils import subsample
from climate_scenarios_2050.subsampling_list import Smith_subsampling
from climate_scenarios_2050.config import *

from joblib import Parallel,delayed

# Don't use this:
# def ens_shift(
#         obs,
#         comp_ensemble_sim,
#         shift_ensemble_sim,
#         end_year,
#         years_full,
#         per_len=20,
#         max_lag=6,
#         measure='RMSE'
#     ):
    
#     check_period = [end_year - per_len, end_year]
#     lags = np.arange(-max_lag,max_lag+.1,1,dtype=int)

#     obs_window = obs.sel(year=slice(*check_period))

#     score = []
#     for lag in lags:
#         lagged_mod = comp_ensemble_sim.sel(year=slice(check_period[0]+lag,check_period[1]+lag))
#         # compute RMSE:
#         if measure == "RMSE":
#             SCORE = np.nanmean(((obs_window - lagged_mod).values**2),axis=0)**.5
#         # correlation:
#         elif measure == "CORR":
#             # SCORE = [np.corrcoef(obs_window.values[~np.isnan(obs_window.values)],mem[~np.isnan(obs_window.values)])[0,1] for mem in lagged_mod.values]
#             SCORE = xr.corr(obs_window,lagged_mod,dim="year").values
#         score.append(SCORE)

#     if measure in ["RMSE"]:
#         score_best = np.nanargmin(np.array(score),axis=0)
#     elif measure in ["CORR"]:
#         score_best = np.nanargmax(np.array(score),axis=0)
#     best_lags = lags[score_best]


#     ensemble_shifted = []
#     for ii,mem in enumerate(comp_ensemble_sim.member.values):
#         if 'year' in list(shift_ensemble_sim.coords):
#             mem_shift = shift_ensemble_sim.sel(member=mem).sel(year = slice(years_full.min() + best_lags[ii],years_full.max() + best_lags[ii]))
#             mem_shift = mem_shift.assign_coords(year=mem_shift.year - best_lags[ii])
            
#             ensemble_shifted.append(mem_shift.sel(year=slice(years_full.min()+max_lag,years_full.max()-max_lag)))
#         elif 'time' in list(shift_ensemble_sim.coords):
#             mem_shift = shift_ensemble_sim.sel(member=mem).sel(time=slice(str(years_full.min() + best_lags[ii]),str(years_full.max() + best_lags[ii])))
#             mem_shift = mem_shift.assign_coords(time=pd.to_datetime(mem_shift.time) - pd.DateOffset(years=best_lags[ii]))
            
#             ensemble_shifted.append(mem_shift.sel(time=slice(str(years_full.min()+max_lag),str(years_full.max()-max_lag))))

#     ensemble_shifted = xr.concat(ensemble_shifted,dim='member')
    
#     return ensemble_shifted

def ens_subsample(
        obs,
        comp_ensemble_sim,
        shift_ensemble_sim,
        end_year,
        # years_full,
        per_len=20,
        M=50,
        measure = "RMSE",
    ):
    """
    subsample the ensemble and only keep the M best members (by RMSE)
    """
    if per_len == "all":
        check_period = [obs.year.min().values, end_year]
    else:
        check_period = [end_year - per_len, end_year]

    # re-center data over the check_period:
    obs_centr = obs.sel(year=slice(*check_period)) - obs.sel(year=slice(*check_period)).mean()
    # re-center each model separately:
    comp_ensemble_sim_sel = comp_ensemble_sim.sel(year=slice(*check_period))
    ens_centr = comp_ensemble_sim_sel.groupby("m_id") - comp_ensemble_sim_sel.groupby('m_id').mean("member")

    if measure == "RMSE":
        score = ((obs_centr - ens_centr)**2).mean(dim="year",skipna=True)**.5
    elif measure == "CORR":
        score = xr.corr(obs_centr, ens_centr,dim="year")
    
    # choose M best members:
    if measure in ["RMSE"]:
        ensemble_subsampled = shift_ensemble_sim.sortby(score).isel(member=slice(M))
    elif measure in ["CORR"]:
        ensemble_subsampled = shift_ensemble_sim.sortby(score,ascending=False).isel(member=slice(M))
        
    return ensemble_subsampled

def generate_hindcast(init_year,index_int_var_obs,sync_indx_mod,index_int_var_model,years_full,hc_param_set,measure,roll,selection,M=50):

    print(init_year)

    #-----CUT OFF everything after init_year-----#
    index_int_var_obs_rem = index_int_var_obs.sel(year=slice(init_year)) # includes `init_year`!

    #-----FILTER REMAINDER-----#
    # nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, index_int_var_obs_rem, lowcut_freq, highcut_freq, 1, True)
    # nao_hadslp_bp = xr.apply_ufunc(butter_lowpass_filter, index_int_var_obs_rem, highcut_freq, 1, True)
    # sync_indx_obs = nao_hadslp_bp.sel(year=slice(index_int_var_obs.year.min().values, init_year - hc_param_set["cutoff_period_long"]//2))
    sync_indx_obs = index_int_var_obs_rem.rolling(year=roll,center=True,min_periods=roll//4).mean(skipna=True)

    # init_year = 2015 #nao_hadslp.year.max().values
    # NAO_sync = ens_shift(sync_indx_obs,sync_indx_mod,index_int_var_model,init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'],measure="RMSE")
    # NAO_sync_initialized = ens_subsample(rm_indx_obs,rm_indx_mod,NAO_sync,init_year,per_len=hc_param_set['vf_win_len'],M=M,measure=measure)
    # if selection == "shift":
    #     NAO_sync_initialized = ens_shift(sync_indx_obs,sync_indx_mod,index_int_var_model,init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'],measure=measure)
    # elif selection == "subsample":
        # try a different method:
    NAO_sync_initialized = ens_subsample(sync_indx_obs,sync_indx_mod,index_int_var_model,init_year,per_len=hc_param_set['vf_win_len'],M=M,measure=measure)
    # possible second stage of matching, e.g. on amplitude (RMSE) after matching phase (CORR)
    # NAO_sync_initialized = ens_subsample(sync_indx_obs,sync_indx_mod.sel(member=NAO_sync_initialized.member),NAO_sync_initialized,init_year,per_len=hc_param_set['vf_win_len'],M=M//2,measure="RMSE")
    
    # make a new dimension associated with a generic member dimension in order to be able to load multiple hindcasts at once later
    NAO_sync_initialized = NAO_sync_initialized.assign_coords(mem=("member",np.arange(len(NAO_sync_initialized.member)))).rename({"member":"member_id"}).swap_dims({"member_id":"mem"}).rename({"mem":"member"})
    
    # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
    lt_coord = NAO_sync_initialized.year.values - init_year
    lt_coord[np.where(lt_coord<0)] = 0
    # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
    if hc_param_set['vf_win_len'] != "all":
        lt_coord_neg = NAO_sync_initialized.year.values - (init_year - hc_param_set['vf_win_len']-1)
        lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
    NAO_sync_initialized = NAO_sync_initialized.assign_coords({'lead_time':('year',lt_coord)}).drop('mode')
    NAO_sync_initialized = NAO_sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
    NAO_sync_initialized = NAO_sync_initialized.assign_coords(member_id=(("member","init_year"),np.tile(NAO_sync_initialized.member_id.values,[1,1]).T))
    # save hindcast:
    hc_name = 'hc_{}.nc'.format(init_year)
    NAO_sync_initialized.to_netcdf(hc_path/hc_name)
    return NAO_sync_initialized

if __name__ == '__main__':

    # define the parameter sets to try:
    parameter_sets = [
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : 'varsplit_7y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : 'varsplit_20y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 75, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : 'NAO_AR1_D2.5', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : 'varsplit_7y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : 'NAO_AR1_D2.5', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : "all", 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 60, 'mxlags' : 30,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
        
    ]

    for hc_param_set in parameter_sets:
        # index (normally NAO):
        index_name = "NAO"
        measure = "CORR" # "RMSE"/"CORR"-zz
        roll = 41 # Can we get a better idea for the period from fft?
        selection = "subsample" # "shift"/"subsample"
        M = 100

        # create the path to save the hindcast to:
        hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
        hc_path.mkdir(parents=True,exist_ok=True)
        # empty the directory:
        for hc_fi in list(hc_path.glob("*.nc")):
            hc_fi.unlink()



        # filter applied to forced signal (which is removed):
        # only choose the unfiltered version (`f-None`) since the others are prone to removing part of the multi-decadal variability!
        filter_ext = "f-None"

        # take inverse of periods to get cutoff frequencies for bandpass filter:
        # lowcut_freq = 1/hc_param_set['cutoff_period_long']
        # highcut_freq = 1/hc_param_set['cutoff_period_short']


        # Get indices for synchronization:
        # load the model index natural variability:
        index_int_var_model = xr.open_dataarray(CMIP6_base/"internal/internal-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))
        # forced signal (per model)
        # forced_var_model = xr.open_dataarray(CMIP6_base/"forced/forced-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))
        # index_full_model = forced_var_model.sel(m_id=index_int_var_model.m_id) + index_int_var_model
        index_full_model = index_int_var_model
        # subsample the model?
        if hc_param_set["subsampling"] is not None:
            index_full_model = subsample(index_full_model,hc_param_set["subsampling"])

        years_full = np.arange(index_int_var_model.year.min(), index_int_var_model.year.max() + 1)

        # if hc_param_set['subsampling'] is not None:
        #     print("need to implement subsampling")
        #     continue

        # load the observed index variability:
        if hc_param_set["ref_ds"] == "ERA5":
            obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
        else:
            obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
        index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))
        # index_forced_var_obs = xr.open_dataarray(obs_ind_dir/"forced-{1}-variability_{0}.nc".format(filter_ext,index_name))
        # index_full_obs = index_forced_var_obs + index_int_var_obs
        index_full_obs = index_int_var_obs

        #--------FILTERING----------#
        # make sure to take out the nans by using nancut = True
        # band-pass filter the NAO ensemble for synchronization:
        # ensemble_bp = xr.apply_ufunc(
        #     butter_lowpass_filter,index_full_model,highcut_freq,1,True,
        #     input_core_dims=[['year'],[],[],[]],output_core_dims=[['year']],
        #     dask='allowed',vectorize=True,
        # )

        # nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, index_int_var_obs, lowcut_freq, highcut_freq, 1)
        #--------END FILTERING----------#

        # generate list of possible initialization years (limited by sync'ing window length):
        if selection == "shift":
            init_years = np.arange(index_int_var_obs.year.values.min()+hc_param_set['vf_win_len']+hc_param_set['mxlags'],index_int_var_obs.year.values.max()+1,1)
        elif selection == "subsample":
            # if hc_param_set['vf_win_len'] == "all":
            init_years = np.arange(index_int_var_obs.year.values.min() + roll + roll//2,index_int_var_obs.year.values.max()+1,1)
            # else:
            #     init_years = np.arange(index_int_var_obs.year.values.min()+hc_param_set['vf_win_len']//2,index_int_var_obs.year.values.max()+1,1)

        print('generating {} hindcasts'.format(len(init_years)))

        # Shift/sync:
        # wgts = signal.sawtooth(np.linspace(0,1,roll)*np.pi*2,.5)+1
        # weight = xr.DataArray(wgts, dims=['window'])
        sync_indx_mod = index_full_model.rolling(year=roll,center=True,min_periods=roll//4).mean(skipna=True)
        index_full_model = index_full_model.sel(member=sync_indx_mod.member)
        # sync_indx_mod = ensemble_bp

        # generate in parallel
        hc_gen = Parallel(n_jobs=-5)(
            delayed(generate_hindcast)(init_year,index_full_obs,sync_indx_mod,index_full_model,years_full,hc_param_set,measure,roll,selection,M) for init_year in init_years
        )

        # "Slow" loop version
        # for init_year in init_years:
        #     print(init_year)

        #     #-----CUT OFF everything after init_year-----#
        #     index_int_var_obs_rem = index_int_var_obs.sel(year=slice(init_year))

        #     #-----FILTER REMAINDER-----#
        #     nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, index_int_var_obs_rem, lowcut_freq, highcut_freq, 1, True)
        #     # sync_indx_obs = index_int_var_obs_rem.rolling(year=20,center=True).mean()
        #     sync_indx_obs = nao_hadslp_bp
        #     sync_indx_mod = index_int_var_model.rolling(year=20,center=True).mean()

        #     # init_year = 2015 #nao_hadslp.year.max().values
        #     # NAO_sync_initialized = RMSE_shift(nao_hadslp_bp,ensemble_bp,index_int_var_model,init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'])
        #     # try a different method:
        #     NAO_sync_initialized = ens_shift(sync_indx_obs,ensemble_bp,index_int_var_model,init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'],measure=measure)

            
        #     # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
        #     lt_coord = NAO_sync_initialized.year.values - init_year
        #     lt_coord[np.where(lt_coord<0)] = 0
        #     # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
        #     lt_coord_neg = NAO_sync_initialized.year.values - (init_year - hc_param_set['vf_win_len']-1)
        #     lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
        #     NAO_sync_initialized = NAO_sync_initialized.assign_coords({'lead_time':('year',lt_coord)}).drop('mode')
        #     NAO_sync_initialized = NAO_sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
        #     # save hindcast:
        #     hc_name = 'hc_{}.nc'.format(init_year)
        #     NAO_sync_initialized.to_netcdf(hc_path/hc_name)
