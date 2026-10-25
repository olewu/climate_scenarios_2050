# Pull wind time series for Smøla from the CMIP6 climate model runs"

import xarray as xr
import numpy as np
import pandas as pd
import re
import cftime
from pathlib import Path
from climate_scenarios_2050.config import *
from climate_scenarios_2050.utils import preprocess_expand_member_dim_
from climate_scenarios_2050.interpolate import interpolate_grid
import numcodecs
import shutil
numcodecs.blosc.set_nthreads(4)

# Compressor:
compressor = numcodecs.Blosc(
    cname="zstd",     # algorithm
    clevel=5,         # compression level (1–9)
    shuffle=numcodecs.Blosc.BITSHUFFLE
)

def get_chunk_tuple(da, chunk_dict):
    """
    Converts a chunk dictionary {'dim': size, ...} into a tuple of integers
    ordered exactly like the DataArray's dimensions.
    """
    return tuple(chunk_dict[dim] for dim in da.dims)

def get_interp_ds(file_list,interp_meth="bilinear"):
    
    ds_common_grid = interpolate_grid(
        file_list,
        grid_spacing = grid_spacing,
        variable = variable,
        preprocess=preprocess_expand_member_dim_,
        engine='netcdf4',
        method=interp_meth,
        concat_dim='member',
        to_stdgrid=True,
    )

    ds_common_grid = ds_common_grid.sortby("time")
    try:
        mask = (ds_common_grid.time >= pd.Timestamp('1850-01-01')) & (ds_common_grid.time <= pd.Timestamp('2099-12-31'))
        ds_common_grid = ds_common_grid.where(mask, drop=True)
    except:
        pass
    ds_common_grid = ds_common_grid.convert_calendar("noleap", use_cftime=True, align_on="year")
    ds_common_grid = ds_common_grid.assign_coords(
        time=[cftime.DatetimeNoLeap(t.year, t.month, 1) for t in ds_common_grid.time.values]
    )
    ds_common_grid = ds_common_grid.drop_duplicates(dim="time", keep="first")
    ds_common_grid = ds_common_grid.sel(
        time=slice(
            cftime.DatetimeNoLeap(1850, 1, 1),
            cftime.DatetimeNoLeap(2099, 12, 31),
        )
    )

    return ds_common_grid

gsp = 1.5
grid_spacing = {'lat':gsp,'lon':gsp}

base_path_search = proj_base/'data/CMIP6'

with open(model_id_file,'rb') as f:
    M_ID = pickle.load(f)

output_path = Path("/datalake/NS9873K/DATA/CMIP6/wind_db")
OPTIMAL_CHUNKS = {'time': 1000, 'lat': 1, 'lon': 1, 'member': -1}

for variable in ["psl"]: # "sfcWind","tas"
    for scenario in ["ssp245","ssp370"]: 

        if (variable == "sfcWind") and (scenario == "ssp245"):
            continue
        output_name = f"{variable}_Amon_{scenario}_multimodel.zarr"
        output_name_raw = f"{variable}_Amon_{scenario}_multimodel_fragm.zarr"

        # get a list of all available models:
        models = []
        for exp in ['historical',scenario]:
            # Collect files
            file_list = sorted(list(base_path_search.glob(f"{exp}/*/{variable}/{variable}_Amon_*.nc")))
            # Remove empty ones:
            file_list = [filename for filename in file_list if os.path.getsize(filename) > 0]
            # get model name:
            models.append([filename.parent.parent.stem for filename in file_list])
        # only select those member for which both historical and scenario simulations exist:
        model_selection = sorted(list(set(models[0]).intersection(models[1])))

        ensemble_pattern = r'(r\d+i\d+p\d+f\d+)'

        # algin time axis first:
        all_times = []

        for model in model_selection:
            print(model,M_ID[model])

            filename_hist = list(base_path_search.glob(f"historical/{model}/{variable}/{variable}*{model}*.nc"))
            filename_scen = list(base_path_search.glob(f"{scenario}/{model}/{variable}/{variable}*{model}*.nc"))

            if (variable == "psl") and (model == "EC-Earth3-Veg"):
                filename_scen = [fn for fn in filename_scen if "230012" not in fn.name]

            ds_hist = get_interp_ds(filename_hist)
            ds_scen = get_interp_ds(filename_scen)

            ds = xr.concat([ds_hist, ds_scen], dim="time", join="outer").sortby("time")

            all_times.append(ds.time.values)
        
        global_time = np.unique(np.concatenate(all_times))
        global_time = np.sort(global_time)

        first = True
        for model in model_selection:
            print(model,M_ID[model])
            filename_hist = list(base_path_search.glob(f"historical/{model}/{variable}/{variable}*{model}*.nc"))
            try:
                filename_scen = list(base_path_search.glob(f"{scenario}/{model}/{variable}/{variable}*{model}*.nc"))
                if (variable == "psl") and (model == "EC-Earth3-Veg"):
                    filename_scen = [fn for fn in filename_scen if "230012" not in fn.name]
            except:
                continue
            
            ens_id_hist = [re.search(ensemble_pattern, filename.stem).group(1) for filename in filename_hist]
            ens_id_scen = [re.search(ensemble_pattern, filename.stem).group(1) for filename in filename_scen]

            common_mems = list(set(ens_id_scen) & set(ens_id_hist))

            hist_files_common = [filenm for filenm in filename_hist if re.search(ensemble_pattern, filenm.stem).group(1) in common_mems]
            scen_files_common = [filenm for filenm in filename_scen if re.search(ensemble_pattern, filenm.stem).group(1) in common_mems]

            interp_ds_hist = get_interp_ds(hist_files_common)
            interp_ds_scen = get_interp_ds(scen_files_common)

            ds = xr.concat(
                [interp_ds_hist, interp_ds_scen],
                dim="time",
                join="outer"
            ).sortby("time")

            _, index = np.unique(ds.time.values, return_index=True)
            ds = ds.isel(time=np.sort(index))
            
            ds = ds.reindex(time=global_time)

            # extend the member index to include a decoding of the model:
            ds = ds.assign_coords(model_id=("member",[M_ID[model]]*len(ds.member)))
            ds = ds.assign_coords(model_id=ds.model_id.astype("U3"))
            ds = ds.assign_coords(member_id=("member",[M_ID[model]+mem for mem in ds.member.values])).swap_dims({"member":"member_id"}).drop("member").rename({"member_id":"member"})
            ds = ds.assign_coords(member=ds.member.astype("U13"))

            ds = ds.chunk({'time': 100, 'lat': 30, 'lon': 60, 'member': -1}).to_dataset(name=variable)
            
            encoding = {var: {"compressor": compressor, 'dtype': 'float32'} for var in ds.data_vars}

            if "height" in ds.coords:
                ds = ds.drop("height")
            
            if first:
                ds.to_zarr(output_path/output_name_raw, mode="w", encoding=encoding)
                first = False
            else:
                ds.to_zarr(output_path/output_name_raw, mode="a", append_dim="member",)

            ds.close()

        # the final zarr store is fragmented along `member`. Open, re-chunk and save to a new store
        ds_raw = xr.open_zarr(output_path/output_name_raw, consolidated=True)

        # TODO: check CESM (all versions)?? SerializationWarnings
        # Re-chunk and save to a NEW store
        OUTPUT_PATH = output_path/output_name

        # This operation reads the old chunks and writes new ones
        ds_optimized = ds_raw.chunk(OPTIMAL_CHUNKS)
        ds_optimized.to_zarr(
            OUTPUT_PATH, 
            mode='w',
            encoding={
                variable: {
                    'compressor': compressor,
                    'dtype': 'float32',
                }
            }
        )

        shutil.rmtree(output_path/output_name_raw)



###-------------------START: merge all variables for one scenario into single zarr store------------------###
# put all (non NAO) variables into a single data store
# pad the member dimension with NaNs where members exist for one variable but not the other
# include some logic to drop NaN members in the analog selection
# for scenario in ["ssp245","ssp370"]:
#     print(scenario)
#     scenario_store_name = output_path/f"Amon_{scenario}_multimodel.zarr"
#     if scenario_store_name.exists():
#         with xr.open_zarr(scenario_store_name) as exstr:
#             existing_variables = [v for v in exstr.data_vars]
#     else:
#         existing_variables = []
#     stores_to_merge = list(output_path.glob(f"*{scenario}*multimodel.zarr"))
#     # open all stores and extract the member dimension:
#     mems = []
#     for store in stores_to_merge:
#         with xr.open_zarr(store) as ds:
#             mems.append(ds.member.values)

#     # find max set of members:    
#     max_members = list(set([item for sublist in mems for item in sublist]))
#     for store in stores_to_merge:
#         variable = store.stem.split('_')[0]
#         if (store == scenario_store_name) or (variable in existing_variables):
#             continue
#         print(variable)
#         ds = xr.open_zarr(store)
#         current_members = ds.member.values
#         missing_members = list(set(max_members) - set(current_members))
#         missing_model_id = [re.match(r"(m\d{1,3})",mm).group(1) for mm in missing_members]
#         # nan pad NAO index:
#         padding_dims = {dim: size for dim, size in ds.sizes.items() if dim != 'member'}
#         padding = xr.DataArray(
#             np.full((len(missing_members),) + tuple(padding_dims.values()), 
#                     np.nan),
#             dims=['member'] + list(padding_dims.keys()),
#             coords={
#                 'member': ("member",missing_members),
#                 'model_id' : ("member",missing_model_id)
#             }
#         ).assign_coords({coord:ds[coord].values for coord in ds.coords if coord not in ["member","model_id"]})
#         da_scen = xr.concat([ds[variable], padding], dim='member').sortby("member")
#         ds_scen = xr.Dataset(
#             data_vars={variable:da_scen}
#         ).assign_coords({"model_id":("member",da_scen.model_id.values)}).chunk(OPTIMAL_CHUNKS)
        
#         encoding = {var: {"compressor": compressor, 'dtype': 'float32'} for var in ds_scen.data_vars}

#         if scenario_store_name.exists():
#             # append if store already exists:
#             ds_scen.to_zarr(scenario_store_name, mode="a", encoding=encoding)
#         else:
#             ds_scen.to_zarr(scenario_store_name, mode="w", encoding=encoding)

# print(scenario)

# scenario_store_name = output_path / f"Amon_{scenario}_multimodel.zarr"

# stores_to_merge = list(output_path.glob(f"*{scenario}*multimodel.zarr"))

# # create/open output dataset lazily
# if scenario_store_name.exists():
#     out_ds = xr.open_zarr(scenario_store_name, consolidated=False)
#     existing_variables = [v for v in out_ds.data_vars]
# else:
#     out_ds = xr.Dataset()
#     existing_variables = []

# for store in stores_to_merge:
#     variable = store.stem.split('_')[0]
#     if (store == scenario_store_name) or (variable in existing_variables):
#         continue
#     print("processing:", variable)

#     ds = xr.open_zarr(store, consolidated=False)[[variable]]

#     # just write variable as-is into shared store
#     mode = "a" if scenario_store_name.exists() else "w"

#     if mode == "w":
#         # first variable creates structure
#         ds.to_zarr(
#             scenario_store_name,
#             mode="w",
#             encoding={
#                 variable: {
#                     "compressor": compressor,
#                     "dtype": "float32"
#                 }
#             },
#             consolidated=True
#         )
#     else:
#         # append only works after member dimension exists
#         ds.to_zarr(
#             scenario_store_name,
#             mode="a",
#             append_dim="member",
#             encoding={
#                 variable: {
#                     "compressor": compressor,
#                     "dtype": "float32"
#                 }
#             },
#             consolidated=True
#         )

###-------------------END------------------###

# for scenario in ["ssp245","ssp370"]: 

#     ds = xr.open_zarr(
#         output_path/f"Amon_{scenario}_multimodel.zarr",
#         consolidated=True,   # set True if already consolidated
#     )

#     ds_rechunked = ds.chunk(OPTIMAL_CHUNKS)

#     for var in ds_rechunked.data_vars:
#         ds_rechunked[var].encoding.pop("chunks", None)
#         ds_rechunked[var].encoding.pop("preferred_chunks", None)

#     # Write new optimized store
#     ds_rechunked.to_zarr(
#         output_path/f"Amon_{scenario}_multimodel_optimized.zarr",
#         mode="w",
#         consolidated=True,
#         zarr_version=2,
#     )

#     print("Optimized Zarr store written successfully.")


# fix faulty member ("m24r4i1p1f1") in zarr stores:
# import zarr
# import numpy as np

# for scenario in ["ssp245","ssp370"]:
#     zarr_array = zarr.open(f'/datalake/NS9873K/DATA/CMIP6/wind_db/tas_Amon_{scenario}_multimodel.zarr', mode='r+')

#     faulty_ens_mem = int(np.where(zarr_array.member[:] == "m24r4i1p1f1")[0][0])
#     ts = zarr_array.tas[faulty_ens_mem, :, 0, 0]
#     bad_t = np.where(ts == 0)[0]
#     zarr_array.tas[faulty_ens_mem, bad_t, :, :] = np.nan
