# Split models and observations into forced signal and natural varibility

# forced signal: measured by multi-model ensemble mean (MMEM)
# include de-seasonalizing (for monthly data) and filtering
#   this signal is then projected onto the single-model ensemble mean (SMEM)
#   for each model or the observations (remember to normalize/subtract the mean) and
#   subtracted from each simulation to get the natural variability

# Do this both for the indexes (NAO, ?)
# and in the same way, grid point by grid point for fields
# for fields should ideally use optimal fingerprinting (not yet implemented)

# save forced signal and natural variability for each 

import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from climate_scenarios_2050.cmip6_nao import butter_lowpass_filter
from climate_scenarios_2050.config import *
from climate_scenarios_2050.utils import load_model_ensemble_NAO
import re

# from dask.distributed import Client

# client = Client()


CMIP6_forced = CMIP6_base/"forced"
CMIP6_forced.mkdir(exist_ok=True,parents=True)
CMIP6_internal = CMIP6_base/"internal"
CMIP6_internal.mkdir(exist_ok=True,parents=True)

def generate_to_grid(grid_spacing, method="bilinear", to_stdgrid=False):
    if method == 'conservative':
        try:
            lon_b = np.concatenate([
                ds_in.lon_bnds[0, :, 0].values[:1],  # first lower bound
                ds_in.lon_bnds[0, :, 1].values       # all upper bounds
            ])

            lat_b = np.concatenate([
                ds_in.lat_bnds[0, :, 0].values[:1],
                ds_in.lat_bnds[0, :, 1].values
            ])            

            ds_in = ds_in.assign_coords({
                'lon_b': (('lon_b',), lon_b),
                'lat_b': (('lat_b',), lat_b)
            })
        except:
            pass

    if to_stdgrid:

        lat_out_half = np.arange(grid_spacing['lat']*1/2, 90.1,grid_spacing['lat'])
        lat_out = np.concatenate([np.flip(-lat_out_half),lat_out_half])
        lon_out = np.arange(-180+grid_spacing['lon']*1/2, 180, grid_spacing['lon'])
    else:
        lat_out_half = np.arange(grid_spacing['lat']*1/2, 90.1,grid_spacing['lat'])
        lat_out = np.concatenate([np.flip(-lat_out_half),lat_out_half])
        lon_out = np.arange(grid_spacing['lon']*1/2, 360, grid_spacing['lon'])
    
    return xr.Dataset({"lat": (["lat"], lat_out),"lon": (["lon"], lon_out),})


def load_model_ensemble(variable,scen,interpolate_grid=None,method="bilinear"):
    """
    load a full (1850 - 2100) time series of monthly global fields, using members from the 
    scenario `scen`.
    If subsampling is wanted, do it to the hindcasts instead of generating another hindcast dataset
    """

    # get a list of all available models:
    models = []; model_files = []
    for exp in ['historical',scen]:
        # Collect the regridded files
        file_list = sorted(list(data_paths["processed"].glob(f"{exp}/*/{variable}/{variable}*.zarr")))
        # Remove empty ones:
        file_list = [filename for filename in file_list if os.path.getsize(filename) > 0]
        model_files.append(file_list)
        # get model name:
        models.append([filename.stem.split('_')[2] for filename in file_list])
    # only select those member for which both historical and scenario simulations exist:
    av_models = sorted(list(set(models[0]).intersection(models[1])))
    
    model_selection = av_models
    
    # subsample model files:
    model_files_zarr = []
    for file_list_fc in model_files:
        model_files_zarr.append([fl for fl in file_list_fc if fl.parent.parent.stem in model_selection])


    # load historical and scenario:
    ds_hist = xr.open_mfdataset(model_files_zarr[0],engine="zarr",concat_dim="member",combine="nested")
    ds_hist = ds_hist.sel(time=slice("1850","2014"))
    ds_scen = xr.open_mfdataset(model_files_zarr[1],engine="zarr",concat_dim="member",combine="nested")
    ds_scen = ds_scen.sel(time=slice("2015","2100"))

    # find common ensemble members and subsample:
    common_members = np.intersect1d(ds_hist.member.values, ds_scen.member.values)

    # then merge those members by time:
    ds = xr.concat([ds_hist.sel(member=common_members),ds_scen.sel(member=common_members)],dim='time')
    
    # interpolate to coarser grid"
    if interpolate_grid is not None:
        to_grid = generate_to_grid(grid_spacing=interpolate_grid,method=method)
        regridder = xe.Regridder(ds, to_grid, method=method, ignore_degenerate=True, periodic=True)
        ds = regridder(ds)


    return ds.assign_coords(time=pd.DatetimeIndex(ds.time.values))

def split_variability_index(index_model_ensemble,index_obs,forcing,index_name,obs_ds,filt=None):
    """
    index input must be xr.DataArray
    split the variability of an index time series into 
    forced and internal/natural variability
    """
        
    # get a file name extension based on the filter:
    filter_ext = filter_dict_to_ext(filt)
    
    # get the location of observations:
    if obs_ds == "ERA5":
        obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
    else:
        obs_ind_dir = proj_base/f"data/{obs_ds}"
    
    # Ensure year is sorted and numeric
    years_full = np.arange(index_model_ensemble.year.min(), index_model_ensemble.year.max() + 1)
    # Interpolate to get a value for 2014
    index_model_ensemble = index_model_ensemble.reindex(year=years_full).interpolate_na(dim='year', method='linear')
    index_model_ensemble = index_model_ensemble.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in index_model_ensemble.member.values]))
    index_model_ensemble = index_model_ensemble.assign_coords(member=index_model_ensemble.member.astype(object))

    # compute the forced signal as the (low-pass-filtered) MMEM:
    mmem_unfiltered = index_model_ensemble.mean('member')
    if filt is None:
        mmem = mmem_unfiltered
    elif isinstance(filt,dict):
        mmem = xr.apply_ufunc(
            butter_lowpass_filter,mmem_unfiltered,filt["cutoff_freq"],filt["fs"],filt["order"],filt["pad"],
            input_core_dims=[['year'],[],[],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )
    else:
        print("provide None or an appropriate filter dictionary")

    # normalize the MMEM signal to mean 0 over the entire period:
    mmem_normed = mmem - mmem.mean("year")
    # normalize the single simulations with the SMEM over the entire period:
    # single model mean over entire period:
    smem_mean = index_model_ensemble.mean("year").groupby("m_id").mean("member")
    index_model_ensemble_normed = index_model_ensemble.groupby("m_id") - smem_mean

    # project the normalized MMEM onto the normalized SMEMs (which are just averages over one single member in some cases)
    # i.e. scale the MMEM by the ratio of the covariance of MMEM and SMEM and the variance of MMEM
    forced_signal_single_model = (xr.cov(index_model_ensemble_normed,mmem_normed,dim="year")/mmem_normed.var()).groupby('m_id').mean("member") * mmem_normed

    # remove the scaled signals by subtracting the respective model's signal from each simulation:
    index_model_ensemble_int = index_model_ensemble_normed - forced_signal_single_model.sel(m_id=index_model_ensemble.m_id)

    # save both the single model (incl. observations) signals and natural variability:
    # first add back the entire period average to the single model signal, then save:
    (forced_signal_single_model + smem_mean).to_netcdf(CMIP6_forced/f"forced-{index_name}-variability_hist-{forcing}_{filter_ext}.nc",mode="w")

    # natural variability model:
    index_model_ensemble_int.to_netcdf(CMIP6_internal/f"internal-{index_name}-variability_hist-{forcing}_{filter_ext}.nc",mode="w")

    #---------------
    # Treat observations as if they were another single-member ensemble
    #---------------
    
    # normalize:
    mmem_normed_obsper = mmem.sel(year=index_obs.year) - mmem.sel(year=index_obs.year).mean("year")
    index_obs_normed = index_obs - index_obs.mean()

    # compute the forced signal in the observations by regressing NAO onto the MMM
    forced_signal_obs = xr.cov(index_obs_normed.sel(year=mmem_normed_obsper.sel(year=index_obs_normed.year).year),mmem_normed_obsper.sel(year=index_obs_normed.year))/mmem_normed_obsper.sel(year=index_obs_normed.year).var() * mmem_normed_obsper
    
    # remove the forced signal from observations:
    index_obs_int = index_obs_normed - forced_signal_obs

    # Save the split variability:
    # forced signal observations + constant:
    (forced_signal_obs + index_obs.mean()).to_netcdf(obs_ind_dir/f"forced-{index_name}-variability_{filter_ext}.nc",mode="w")
    
    # natural variability observations:
    index_obs_int.to_netcdf(obs_ind_dir/f"internal-{index_name}-variability_{filter_ext}.nc",mode="w")

    return

def filter_dict_to_ext(filter_dict):

    if filter_dict is None:
        filter_ext = "f-None"
    elif isinstance(filter_dict,dict):
        filter_ext = "f-butter-lp-co{cutoff_freq:}-or{order:}-pd{pad:}".format(**filter_dict)
    else:
        print("please provide valid dict or None")

    return filter_ext

def split_variability_field(model_ens_field,variable,forcing,filt):

    # get a file name extension based on the filter:
    filter_field_ext = filter_dict_to_ext(filt)
    # unforced_file = f"internal-{variable}-variability_hist-{forcing}_{filter_field_ext}.zarr"
    unforced_file = f"internal-{variable}-variability_hist-{forcing}.zarr"

    # forced_signal_file = f"forced-{variable}-variability_hist-{forcing}_{filter_field_ext}.zarr"
    forced_signal_file = f"forced-{variable}-variability_hist-{forcing}.zarr"
    # mmem_file = f"mmem-{variable}_hist-{forcing}_{filter_field_ext}.nc"
    mmem_file = f"mmem-{variable}_hist-{forcing}.nc"

    # load the field to generate the hindcast for
    # it can be used for all parameter values other than `subsampling`!
    # only load if the subsampling method changes
    # model_ens_field = model_ens_field.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in model_ens_field.member.values]))

    # estimate the mmem forced signal:
    if not (CMIP6_forced/mmem_file).exists():
        # compute the forced signal as multi-model ensemble mean:
        mmem_raw = model_ens_field.chunk({"member":-1}).mean("member",skipna=True)[variable]
        # explicitely remove the seasonal cycle (problematic if there is any trend in the data!)
        # mmem_raw_deseas = mmem_raw.groupby("time.month") - mmem_raw.groupby("time.month").mean() + mmem_raw.mean("time")
        # mmem_raw_deseas = mmem_raw_deseas.drop_vars(["month"])

        # "safest" is always no filtering (or deseasonalizing) but note that
        # for monthly data, the seasonal cycle is in the forced signal!
        if filt is None:
            mmem = mmem_raw.compute()
        elif isinstance(filt,dict):
            # filter out remaining "high" frequencies:
            mmem = xr.apply_ufunc(
                butter_lowpass_filter,mmem_raw,filt["cutoff_freq"],filt["fs"],filt["order"],filt["pad"],
                input_core_dims=[['time'],[],[],[],[]],output_core_dims=[['time']],
                dask='allowed',vectorize=True,
            )
        else:
            print("provide None or an appropriate filter dictionary")

        # save the MMEM, it needs to be used in estimating the signal in observations:
        mmem.to_netcdf(CMIP6_forced/mmem_file,mode="w")
    else:
        mmem = xr.open_dataarray(CMIP6_forced/mmem_file)

    # assign a model coordinate
    # model_ens_field = model_ens_field.assign_coords(m_id=("member", [re.match(r'(m\d+)', m).group(1) for m in model_ens_field.member.values]))

    # subtract mean over entire period from MMEM:
    mmem_demean = mmem - mmem.mean("time")
    # also subtract from full model field (long-term single model ensemble mean)
    smem_mean = model_ens_field.mean("time").groupby("model_id").mean("member")
    # to get anomalies wrt to each models mean
    model_ens_demean = model_ens_field.groupby("model_id") - smem_mean
    # project onto each single-model ensemble mean: 
    forced_signal = (xr.cov(model_ens_demean[variable],mmem_demean,dim="time")/mmem_demean.var("time",skipna=True)).groupby("model_id").mean("member") * mmem_demean

    full_forced_signal = (forced_signal + smem_mean).chunk({"model_id":-1,"time":500,"lon":30,"lat":30})
    full_forced_signal.to_zarr(CMIP6_forced/forced_signal_file,mode="w")
    # load after saving since the whole graph had to be computed in the previous saving step
    full_forced_signal = xr.open_zarr(CMIP6_forced/forced_signal_file)

    # Subtract the projcetion of the mmem onto the smem as well as the long term mean to get internal variability around the forced signal
    model_ens_unforced = (model_ens_field - full_forced_signal.sel(model_id=model_ens_field.m_id)).drop_vars(["m_id"])
    # note that the pure mmem of this is not zero (because it still contains variability < 5 years period, including seasona cycle)!
    model_ens_unforced = model_ens_unforced.assign_coords(member=model_ens_unforced.member.astype(object))

    model_ens_unforced.chunk({"member":40,"time":500,"lon":30,"lat":30}).to_zarr(CMIP6_internal/unforced_file,mode="w")

    return


if __name__ == "__main__":

    which_index = "NAO"
    obs_ds="HadSLP"
    ds_var = "scores"
    # don't filter the NAO to retain peaks related to volcanic eruptions 
    # (at the cost of a leakge of internal variability into the mmem and a less smooth signal component)
    filter_NAO = None #dict(cutoff_freq=1/5,fs=1,order=4,pad=None)
    filter_field = None #dict(cutoff_freq=.6,fs=12,order=4,pad=None)

    FORCINGS = ["ssp119", "ssp126", "ssp585"] # ["ssp245","ssp370"]

    for forcing in FORCINGS:
        # Split NAO variability
        # Load model ensemble
        NAO_model_ensemble = load_model_ensemble_NAO(None,forcing,ref_ds=obs_ds)[ds_var] # choose None for subsampling to retain all ensemble members/models
        # Load observations
        if obs_ds == "ERA5":
            NAO_obs = xr.open_dataset('/datalake/NS9873K/DATA/SFE/ERA5_processed/mslp_EOF/{0:s}_psl_EOF_full_index_DJFM_1939-2024_1.5deg.nc'.format(obs_ds))
        else:
            NAO_obs = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2014_1.5deg.nc'.format(obs_ds))
        NAO_obs = NAO_obs.sel(mode=1).squeeze()[ds_var]

        split_variability_index(
            index_model_ensemble = NAO_model_ensemble,
            index_obs = NAO_obs,
            forcing = forcing,
            index_name = which_index,
            obs_ds = obs_ds,
            filt = filter_NAO,
        )

        # do the same for the fields (sfcWind, pr, tas)
        filter_field_ext = filter_dict_to_ext(filter_field)
        for variable in ["sfcWind"]: #  "pr", ,"tas"
            unforced_file = f"internal-{variable}-variability_hist-{forcing}.zarr"
            if not (CMIP6_internal/unforced_file).exists():
                # model_ens_field = load_model_ensemble(variable,scen=forcing)
                model_ens_field = xr.open_zarr(f"/datalake/NS9873K/DATA/CMIP6/wind_db/{variable}_Amon_{forcing}_multimodel.zarr")
                print(f"computing internal variability for {variable} in {forcing}")
                split_variability_field(
                    model_ens_field = model_ens_field,
                    variable = variable,
                    forcing = forcing,
                    filt = filter_field,
                )
            else:
                print("skipping")
