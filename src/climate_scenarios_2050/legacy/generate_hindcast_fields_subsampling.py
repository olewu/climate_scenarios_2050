# produce hindcast set based on input sync properties (frequency band, window, maximum lag)
# 

import xarray as xr
import numpy as np
import numcodecs
from climate_scenarios_2050.config import *
from climate_scenarios_2050.generate_NAO_hindcast import ens_subsample

blosc = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

CMIP6_hc_base = Path("/datalake/NS9873K/DATA/CMIP6_hindcast")
CMIP6_base = Path("/datalake/NS9873K/DATA/CMIP6")

def generate_hindcast_fields(init_year,index_int_var_obs,sync_indx_mod,model_field,variable,years_full,hc_param_set,measure,roll,selection,M=50):

    #-----CUT OFF everything after init_year-----#
    index_int_var_obs_rem = index_int_var_obs.sel(year=slice(init_year)) # includes `init_year`!

    #-----FILTER REMAINDER-----#
    sync_indx_obs = index_int_var_obs_rem.rolling(year=roll,center=True,min_periods=roll//4).mean()

    print(init_year)
    
    sync_initialized = ens_subsample(sync_indx_obs,sync_indx_mod,model_field[variable],init_year,per_len=hc_param_set['vf_win_len'],M=M,measure=measure)
    
    # add a lead time dimension (set zero everywhere there is overlap with the syncing window):
    lt_coord = sync_initialized.time.dt.year.values - init_year
    lt_coord[np.where(lt_coord<0)] = 0
    # define a negative lead time coordinate as well to be able to verify "backward-looking" hindcasts
    if hc_param_set["vf_win_len"] != "all":
        lt_coord_neg = sync_initialized.time.dt.year.values - (init_year - hc_param_set['vf_win_len']-1)
        lt_coord[np.where(lt_coord_neg<0)] = lt_coord_neg[np.where(lt_coord_neg<0)]
    sync_initialized = sync_initialized.assign_coords({'lead_time':('time',lt_coord)})
    sync_initialized = sync_initialized.expand_dims(dim={'init_year':1}).assign_coords({'init_year':('init_year',[init_year])})
    sync_initialized = sync_initialized.sel(time=slice(str(init_year),str(init_year+35)))
    sync_initialized = sync_initialized.swap_dims({"time":"lead_time"})
    # sync_initialized = sync_initialized.assign_coords(member=sync_initialized.member.astype(object))

    # save temporarily as nc:
    outfile_loc = tmp_dir/f"{variable}_{init_year}.zarr"
    
    encoding = {
        variable: {"dtype": "float32", "compressor": blosc},
        'member_id': {'chunks': (20,)}
    }
    
    # save the single initialization:
    sync_initialized.to_dataset(name=variable).chunk({"lat":60,"lon":60,"lead_time":144,"member":20,"init_year":1}).to_zarr(outfile_loc,mode="w",encoding=encoding)
    
    # return the outfile name:
    return outfile_loc
    
def obj2str(ds):
    dsc = ds.copy()
    dsc = dsc.assign_coords(member_id=ds.member_id.astype(str))
    return dsc

if __name__ == '__main__':

    for variable in ["sfcWind","pr","tas"]: # 

        print(variable)
        # define the parameter sets to try 
        parameter_sets = [
            {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
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
        measure = "CORR" # "RMSE"/"CORR"-
        roll = 41 # where does this come from? Can we get an idea for the period from fft?
        selection = "subsample" # "shift"/"subsample"
        M = 100

        # filter applied to forced signal (which is removed):
        filter_ext = "f-None" # "f-butter-lp-co0.2-or4-pdNone"

        # take inverse of periods to get cutoff frequencies for bandpass filter:
        lowcut_freq = 1/hc_param_set['cutoff_period_long']
        highcut_freq = 1/hc_param_set['cutoff_period_short']


        # Get indices for synchronization:
        # load the model index natural variability:
        index_int_var_model = xr.open_dataarray(CMIP6_base/"internal/internal-{2}-variability_hist-{0}_{1}.nc".format(hc_param_set["scen"],filter_ext,index_name))

        if hc_param_set['subsampling'] is not None:
            print("need to implement subsampling")
            pass

        # load the observed index variability:
        if hc_param_set["ref_ds"] == "ERA5":
            obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
        else:
            obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
        index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))

        years_full = np.arange(index_int_var_model.year.min(), index_int_var_model.year.max() + 1)

        #-------------------------------------#
        # Load model simulations
        #-------------------------------------#

        unforced_path = CMIP6_base/"internal"
        forcing = hc_param_set["scen"]
        unforced_file = f"internal-{variable}-variability_hist-{forcing}_{filter_ext}.zarr"
        
        model_ens_unforced = xr.open_zarr(unforced_path/unforced_file,chunks={})

        print("{:} ensemble members".format(len(model_ens_unforced.member)))

        # find common members in NAO and field:
        common_mems = np.intersect1d(index_int_var_model.member.values,model_ens_unforced.member.values)
        index_int_var_model = index_int_var_model.sel(member=common_mems)
        sync_indx_mod = index_int_var_model.rolling(year=roll,center=True,min_periods=roll//4).mean(skipna=True)

        model_ens_unforced_reduced = model_ens_unforced.sel(member=common_mems)

        # generate list of possible initialization years (limited by sync'ing window length):
        init_years = np.arange(index_int_var_obs.year.values.min() + roll + roll//2,index_int_var_obs.year.values.max()+1,1)

        print('generating {} hindcasts'.format(len(init_years)))

        # Shift/sync:
        hc_dir = CMIP6_hc_base/"hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_{vf_win_len}yvf_roll{roll}y".format(scen=hc_param_set["scen"],subsampling=hc_param_set["subsampling"],ref_ds=hc_param_set["ref_ds"],vf_win_len=hc_param_set["vf_win_len"],roll=roll)
        hc_dir.mkdir(exist_ok=True,parents=True)
        tmp_dir = CMIP6_hc_base/"tmp/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_{vf_win_len}yvf_roll{roll}y".format(scen=hc_param_set["scen"],subsampling=hc_param_set["subsampling"],ref_ds=hc_param_set["ref_ds"],vf_win_len=hc_param_set["vf_win_len"],roll=roll)
        tmp_dir.mkdir(exist_ok=True,parents=True)

        # generate the single hindcast files:
        outfiles = []
        for init_year in init_years:
            outfiles.append(generate_hindcast_fields(init_year,index_int_var_obs,sync_indx_mod,model_ens_unforced_reduced,variable,years_full,hc_param_set,measure,roll,selection,M))

        # open the entire hindcast database and store as zarr
        hindcast_ensemble = xr.open_mfdataset(outfiles,combine="nested",concat_dim="init_year",engine="zarr",preprocess=obj2str)
        # hindcast_ensemble = xr.concat(hindcast_ensemble,dim="init_year")
        hindcast_ensemble = hindcast_ensemble.chunk({"lat":60,"lon":60,"lead_time":144,"member":20,"init_year":1})
        hindcast_ensemble = hindcast_ensemble.assign_coords(member=hindcast_ensemble.member.astype(object))
    
        # save hindcast:
        hc_name = 'hc_{}_{}-{}.zarr'.format(variable,init_years[0],init_years[-1])
        
        encoding = {
            variable: {"dtype": "float32", "compressor": blosc},
            'member_id': {'chunks': (20,)}
        }
        hindcast_ensemble.to_zarr(hc_dir/hc_name,mode="w",encoding=encoding)

        # delete the single files:
        # for outf in outfiles:
        #     outf.unlink()