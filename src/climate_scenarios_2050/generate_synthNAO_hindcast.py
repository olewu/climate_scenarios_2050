
# make the analogue ensemble but fitted to synthetic AR(1) time series of the NAO

import xarray as xr
import numpy as np

from climate_scenarios_2050.utils import subsample
from climate_scenarios_2050.utils import load_model_ensemble_NAO
from climate_scenarios_2050.config import *

from joblib import Parallel,delayed


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
    ensemble_subsampled = []
    for obs_mem_ix in score.syobs_member:
        if measure in ["RMSE"]:
            ens_best = shift_ensemble_sim.sortby(score.sel(syobs_member=obs_mem_ix)).isel(member=slice(M))
        elif measure in ["CORR"]:
            ens_best = shift_ensemble_sim.sortby(score.sel(syobs_member=obs_mem_ix),ascending=False).isel(member=slice(M))
        # make a new dimension associated with a generic member dimension in order to be able to load multiple hindcasts at once later
        ens_best = ens_best.assign_coords(mem=("member",np.arange(len(ens_best.member)))).rename({"member":"member_id"}).swap_dims({"member_id":"mem"}).rename({"mem":"member"})
        ensemble_subsampled.append(ens_best)

    return xr.concat(ensemble_subsampled,dim=score.syobs_member)


def generate_hindcast(init_year,index_int_var_obs,sync_indx_mod,index_int_var_model,hc_param_set,measure,roll,M=50):

    print(init_year)

    #-----CUT OFF everything after init_year-----#
    index_int_var_obs_rem = index_int_var_obs.sel(year=slice(init_year))

    #-----FILTER REMAINDER-----#
    sync_indx_obs = index_int_var_obs_rem.rolling(year=roll,center=True,min_periods=roll//4).mean()

    NAO_sync_initialized = ens_subsample(sync_indx_obs,sync_indx_mod,index_int_var_model,init_year,per_len=hc_param_set['vf_win_len'],M=M,measure=measure)
    
    # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
    lt_coord = NAO_sync_initialized.year.values - init_year
    lt_coord[np.where(lt_coord<0)] = 0
    # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
    if hc_param_set['vf_win_len'] != "all":
        lt_coord_neg = NAO_sync_initialized.year.values - (init_year - hc_param_set['vf_win_len']-1)
        lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
    NAO_sync_initialized = NAO_sync_initialized.assign_coords({'lead_time':('year',lt_coord)}).drop('mode')
    NAO_sync_initialized = NAO_sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
    # save hindcast:s
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
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : "all", 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
        # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
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
        # create the path to save the hindcast to:
        hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_synthetic-{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)
        hc_path.mkdir(parents=True,exist_ok=True)
        # empty the directory:
        for hc_fi in list(hc_path.glob("*.nc")):
            hc_fi.unlink()


        # index (normally NAO):
        index_name = "NAO"
        measure = "CORR" # "RMSE"/"CORR"-
        roll = 41 # where does this come from? Can we get an idea for the period from fft?
        M = 100

        # filter applied to forced signal (which is removed):
        filter_ext = "f-None" # "f-butter-lp-co0.2-or4-pdNone"

        # take inverse of periods to get cutoff frequencies for bandpass filter:
        lowcut_freq = 1/hc_param_set['cutoff_period_long']
        highcut_freq = 1/hc_param_set['cutoff_period_short']


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

        # load the synthetic index variability:
        obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/synthetic_ar1_{:}_nao.nc".format(hc_param_set["ref_ds"]))
        index_int_var_obs = xr.open_dataarray(obs_ind_dir)
        # index_forced_var_obs = xr.open_dataarray(obs_ind_dir/"forced-{1}-variability_{0}.nc".format(filter_ext,index_name))
        # index_full_obs = index_forced_var_obs + index_int_var_obs
        index_full_obs = index_int_var_obs


        # generate list of possible initialization years (limited by sync'ing window length):
        init_years = np.arange(index_int_var_obs.year.values.min() + roll + roll//2,index_int_var_obs.year.values.max()+1,1)
        
        print('generating {} hindcasts'.format(len(init_years)))

        # Shift/sync:
        sync_indx_mod = index_full_model.rolling(year=roll,center=True,min_periods=roll//4).mean(skipna=True)

        # generate in parallel
        hc_gen = Parallel(n_jobs=-5)(
            delayed(generate_hindcast)(init_year,index_full_obs,sync_indx_mod,index_full_model,hc_param_set,measure,roll,M) for init_year in init_years
        )
