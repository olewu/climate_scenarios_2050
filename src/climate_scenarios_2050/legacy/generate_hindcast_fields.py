# produce hindcast set based on input sync properties (frequency band, window, maximum lag)
# 

import xarray as xr
import numpy as np
import numcodecs
from climate_scenarios_2050.cmip6_nao import butter_bandpass_filter
from climate_scenarios_2050.config import *
from climate_scenarios_2050.generate_NAO_hindcast import ens_shift
# from dask.distributed import Client

# client = Client()

blosc = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

CMIP6_hc_base = Path("/datalake/NS9873K/DATA/CMIP6_hindcast")
CMIP6_base = Path("/datalake/NS9873K/DATA/CMIP6")

if __name__ == '__main__':

    for variable in ["sfcWind","pr"]: # ,,"tas"

        # define the parameter sets to try 
        parameter_sets = [
            {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
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
        ]

        which_subs = 0

        hc_param_set = parameter_sets[0]

        if variable == "pr":
            int_meth = "conservative"
        else:
            int_meth = "bilinear"

        #-------------------------------------#
        # Load index:
        #-------------------------------------#

        # index (normally NAO):
        index_name = "NAO"

        # filter applied to forced signal (which is removed):
        filter_ext = "f-None" # "f-butter-lp-co0.2-or4-pdNone"

        # take inverse of periods to get cutoff frequencies for bandpass filter:
        lowcut_freq = 1/hc_param_set['cutoff_period_long']
        highcut_freq = 1/hc_param_set['cutoff_period_short']


        # Get indices for synchronization:
        # load the model index natural variability:
        index_int_var_model = xr.open_dataarray(CMIP6_base/"internal/internal-{2}-variability-hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))

        if hc_param_set['subsampling'] is not None:
            print("need to implement subsampling")
            pass

        # load the observed index variability:
        obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
        index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))

        years_full = np.arange(index_int_var_model.year.min(), index_int_var_model.year.max() + 1)

        #-------------------------------------#
        # Load model simulations
        #-------------------------------------#

        unforced_path = CMIP6_base/"internal"
        forcing = hc_param_set["scen"]
        unforced_file = f"internal-{variable}-variability_hist-{forcing}_{filter_ext}.zarr"
        # unforced_path.mkdir(exist_ok=True,parents=True)
        # if not (unforced_path/unforced_file).exists():
            
        #     # load the field to generate the hindcast for
        #     # it can be used for all parameter values other than `subsampling`!
        #     # only load if the subsampling method changes
        #     model_ens = load_model_ensemble(variable,scen=hc_param_set['scen'])
        #     model_ens = model_ens.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in model_ens.member.values]))

        #     # compute the forced signal as multi-model ensemble mean:
        #     forced_signal_path = data_paths['processed'].parent/"CMIP6_forced_signal"
        #     if not (forced_signal_path/f"{variable}_MMEM.nc").exists():
        #         forced_signal_path.mkdir(exist_ok=True,parents=True)
        #         mmem_raw = model_ens.chunk({"member":-1}).mean("member",skipna=True)[variable]
        #         # does the seasonal cycle need to be removed explicitly?
        #         mmem_raw_deseas = mmem_raw.groupby("time.month") - mmem_raw.groupby("time.month").mean() + mmem_raw.mean("time")
        #         # filter out frequencies > 1/(5 years):
        #         mmem_lp = xr.apply_ufunc(
        #             butter_lowpass_filter,mmem_raw_deseas,1/5,12,4,60,
        #             input_core_dims=[['time'],[],[],[],[]],output_core_dims=[['time']],
        #             dask='allowed',vectorize=True,
        #         )

        #         mmem_lp.to_netcdf(forced_signal_path/f"{variable}_MMEM.nc",mode="w")

        #     mmem = xr.open_dataarray(forced_signal_path/f"{variable}_MMEM.nc")

        #     # find common members in NAO and field:
        #     common_mems = np.intersect1d(index_int_var_model.member.values,model_ens.member.values)
        #     model_ens = model_ens.sel(member=common_mems)

        #     # project onto each single-model ensemble mean:
        #     model_ens = model_ens.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in model_ens.member.values]))
        #     mmem_demean = mmem - mmem.mean("time")
        #     model_ens_demean = model_ens.groupby("m_id") - model_ens.mean("time").groupby("m_id").mean("member")
        #     forced_signal = xr.cov(model_ens_demean[variable].groupby("m_id").mean("member"),mmem_demean,dim="time")/mmem_demean.var("time",skipna=True) * mmem_demean

        #     # Subtract the projcetion of the mmem onto the smem as well as the long term mean to get internal variability around the forced signal
        #     model_ens_unforced = (model_ens_demean - forced_signal.sel(m_id=model_ens.m_id)).drop_vars(["m_id","month"])
        #     # note that the pure mmem of this is not zero (because it still contains variability < 5 years period, including seasona cycle)!
        #     model_ens_unforced = model_ens_unforced.assign_coords(member=model_ens_unforced.member.astype(object))

        #     model_ens_unforced.chunk({"member":40,"time":500,"lon":30,"lat":30}).to_zarr(unforced_path/unforced_file,mode="w")

        model_ens_unforced = xr.open_zarr(unforced_path/unforced_file,chunks={})

        print("{:} ensemble members".format(len(model_ens_unforced.member)))

        # find common members in NAO and field:
        common_mems = np.intersect1d(index_int_var_model.member.values,model_ens_unforced.member.values)
        index_int_var_model = index_int_var_model.sel(member=common_mems)

        # band-pass filter the NAO ensemble for synchronization:
        ensemble_bp = xr.apply_ufunc(
            butter_bandpass_filter,index_int_var_model,lowcut_freq,highcut_freq,1,
            input_core_dims=[['year'],[],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )

        # filter the observed natural NAO variability
        nao_hadslp_bp = xr.apply_ufunc(butter_bandpass_filter, index_int_var_obs, lowcut_freq, highcut_freq, 1)

        # generate list of possible initialization years (limited by sync'ing window length):
        init_years = np.arange(nao_hadslp_bp.year.values.min()+hc_param_set['vf_win_len']+hc_param_set['mxlags'],nao_hadslp_bp.year.values.max()+1,1)

        print('generating {} hindcasts'.format(len(init_years)))

        # Shift/sync:
        hc_dir = CMIP6_hc_base/"hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx".format(**hc_param_set)
        hc_dir.mkdir(exist_ok=True,parents=True)
        tmp_dir = CMIP6_hc_base/"tmp/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx".format(**hc_param_set)
        tmp_dir.mkdir(exist_ok=True,parents=True)
        outfiles = []
        for init_year in init_years:
            # if init_year < 2012:
            #     continue

            print(init_year)
        # init_year = 2015 #nao_hadslp.year.max().values
            sync_initialized = ens_shift(nao_hadslp_bp,ensemble_bp,model_ens_unforced[variable],init_year,years_full,per_len=hc_param_set['vf_win_len'],max_lag=hc_param_set['mxlags'])
            
            # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
            lt_coord = sync_initialized.time.dt.year.values - init_year
            lt_coord[np.where(lt_coord<0)] = 0
            # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
            lt_coord_neg = sync_initialized.time.dt.year.values - (init_year - hc_param_set['vf_win_len']-1)
            lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
            sync_initialized = sync_initialized.assign_coords({'lead_time':('time',lt_coord)})
            sync_initialized = sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
            sync_initialized = sync_initialized.sel(time=slice(str(init_year),str(init_year+35)))
            sync_initialized = sync_initialized.swap_dims({"time":"lead_time"})
            sync_initialized = sync_initialized.assign_coords(member=sync_initialized.member.astype(object))

            # save temporarily as nc:
            outfile_loc = tmp_dir/f"{variable}_{init_year}.zarr"
            encoding = {
                variable: {"dtype": "float32", "compressor": blosc},
            }
            
            sync_initialized.to_dataset(name=variable).chunk({"lat":60,"lon":60,"lead_time":144,"member":20,"init_year":1}).to_zarr(outfile_loc,mode="w",encoding=encoding)
            outfiles.append(outfile_loc)
            # hindcast_ensemble.append(sync_initialized.to_dataset(name=variable))

        # open the entire hindcast database and store as zarr
        hindcast_ensemble = xr.open_mfdataset(outfiles,combine="nested",concat_dim="init_year",engine="zarr")
        # hindcast_ensemble = xr.concat(hindcast_ensemble,dim="init_year")
        hindcast_ensemble = hindcast_ensemble.chunk({"lat":60,"lon":60,"lead_time":144,"member":20,"init_year":1})
        hindcast_ensemble = hindcast_ensemble.assign_coords(member=hindcast_ensemble.member.astype(object))
    
        # save hindcast:
        hc_name = 'hc_{}_{}-{}.zarr'.format(variable,init_years[0],init_years[-1])
        hindcast_ensemble.to_zarr(hc_dir/hc_name,mode="w",encoding=encoding)

        # delete the single files:
        # for outf in outfiles:
        #     outf.unlink()