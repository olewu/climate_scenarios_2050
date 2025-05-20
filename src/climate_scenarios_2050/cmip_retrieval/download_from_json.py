"""
Downloads cmip6 sst data from an esgf node.
"""

# from pyesgf.search import SearchConnection
import os
import re
# import requests
import subprocess
import xarray      as xr
import xesmf       as xe
import numpy       as np
from climate_scenarios_2050   import config
from collections   import defaultdict
import json
from pathlib import Path
# from datetime import datetime

# Input --------------------------------------------------------------
write2file     = True
project        = 'CMIP6'
experiment_id  = 'ssp370'
frequency      = 'mon'
variable       = 'psl'
time_range     = np.array((185001,201412)) 
new_grid       = '1x1'
path_in        = {
    'historical': config.proj_base/'data/CMIP6/historical/url_repo/' / variable,
    'ssp245':  config.proj_base/'data/CMIP6/ssp245/url_repo/' / variable,
    'ssp370':  config.proj_base/'data/CMIP6/ssp370/url_repo/' / variable,
}
path_out       = {
    'historical': config.dirs['processed_cmip6_esgf_historical'],
    'ssp245':  config.dirs['processed_cmip6_esgf_ssp245'],
    'ssp370':  config.dirs['processed_cmip6_esgf_ssp370'],
}
# --------------------------------------------------------------------


def download_esgf_data(file_info,path_out):
    return subprocess.run(["wget", "-c", "-P", path_out, file_info['url']])


def regrid_esgf_data_and_extract_nino34(file_info,files_info_regrid,path_out_local,variable,new_grid,time_range):
    
    # open your source dataset
    filename_in  = path_out_local + file_info['filename']
    filename_out = path_out_local + 'nino34_regrid_' + new_grid + '_' + file_info['filename']
    ds_in        = xr.open_dataset(filename_in)
    
    # xESMF expects the coordinate names to be exactly 'lat' and 'lon'.
    # So rename them if needed:
    if 'latitude' in ds_in.variables:
        ds_in = ds_in.rename({"latitude": "lat"})
    if 'longitude' in ds_in.variables:    
        ds_in = ds_in.rename({"longitude": "lon"})
        
    # define your target grid
    if new_grid == '1x1':
        lat_out = np.arange(-90, 91, 1.0)
        lon_out = np.arange(0, 360, 1.0)
        
    ds_out = xr.Dataset({"lat": (["lat"], lat_out),"lon": (["lon"], lon_out),})

    # create the regridder 
    # "bilinear" is the standard choice, but you can also use
    # conservative, patch, nearest_s2d, nearest_d2s, etc.
    # If your data are global and cross the 0/360 boundary, set periodic=True.   
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", ignore_degenerate=True, periodic=True)

    # apply the regridder to the variable
    output = regridder(ds_in[variable])

    # extract nino3.4 region ssts (5S-5N,120-170W)
    output = output.sel(lat=slice(-5,5),lon=slice(190,240))

    # extract times within range
    time_range_dates = [f"{str(x)[:4]}-{str(x)[4:]}" for x in time_range]
    output           = output.sel(time=slice(time_range_dates[0],time_range_dates[-1]))

    # update output filename 
    if output.time.size == 0:
        raise ValueError(f"No data in selected time range {time_range_dates[0]} to {time_range_dates[-1]}.")

    start_str             = str(output.time[0].dt.strftime("%Y%m").item())
    end_str               = str(output.time[-1].dt.strftime("%Y%m").item())
    new_time_str          = f"{start_str}-{end_str}"
    base_out, ext_out     = os.path.splitext(filename_out)
    pattern               = r"\d{6}-\d{6}"  # looks for YYYYMM-YYYYMM
    updated_base_out      = re.sub(pattern, new_time_str, base_out)    
    filename_out          = updated_base_out + ext_out

    # update json dict with new output filename
    old_name = os.path.basename(filename_in)
    new_name = os.path.basename(filename_out)
    for i, filename in enumerate(files_info_regrid):
        if filename == old_name:
            files_info_regrid[i] = new_name
    
    # output regridded file 
    output.name = variable
    output      = output.assign_attrs(standard_name='sea surface temperature',units='degC')
    output.to_netcdf(filename_out)

    # delete raw data
    os.remove(filename_in)

    return files_info_regrid



def combine_files_in_time_by_ensemble_member(files_info_regrid, path_out):
    """
    Combine files that share the same ensemble-member (variant label) but
    differ in time into a single NetCDF file. The ensemble-member is extracted
    from the filename using a regex that matches rX iX pX fX (e.g. r10i1p2f1).

    For each ensemble group:
      - If there is only one file, keep that filename.
      - If there are multiple files, combine them using xarray.open_mfdataset,
        write the merged file to disk, and replace the individual filenames with
        the new combined filename.
    
    Parameters
    ----------
    files_info_regrid : list of str
        List of filenames (e.g., ['tos_Omon_..._185001-185012.nc', ...]).
    path_out : str
        Output directory (should end with a slash or be used with os.path.join).
     
    Returns
    -------
    updated_files_info : list of str
        A new list of filenames in which individual files in a group have been replaced
        by the combined filename.
    """
     
    # Regex pattern to capture the ensemble member, e.g. "r1i1p1f1"
    ensemble_pattern = r'(r\d+i\d+p\d+f\d+)'
    
    # 1) Group files by ensemble member
    grouped_files = defaultdict(list)
    for filename in files_info_regrid:
        match = re.search(ensemble_pattern, filename)
        if match:
            ensemble_member = match.group(1)
        else:
            ensemble_member = 'unknown_ensemble'
        grouped_files[ensemble_member].append(filename)
    
    # 2) Process each group
    updated_files_info = []
    for ensemble_member, file_group in grouped_files.items():
        if len(file_group) == 1:
            # Only one file for this ensemble; keep it as-is.
            updated_files_info.append(file_group[0])
        else:
            # Build full paths for opening
            full_paths = [os.path.join(path_out, fname) for fname in file_group]
            ds_merged = xr.open_mfdataset(full_paths, combine='by_coords').compute()
            
            # Create new filename by combining the time ranges
            start_str = str(ds_merged.time[0].dt.strftime("%Y%m").item())
            end_str   = str(ds_merged.time[-1].dt.strftime("%Y%m").item())
            
            # Extract the common prefix from the first filename.
            # Assumes filenames match pattern: "<prefix>_<start>-<end>.nc"
            prefix_re = re.compile(r'^(.*_)\d{6}-\d{6}(\.nc)$')
            match     = prefix_re.search(file_group[0])
            if match:
                prefix = match.group(1)
            else:
                raise ValueError("First filename does not match the expected pattern.")
            
            # Construct the new filename.
            new_filename = prefix + f'{start_str}-{end_str}.nc'
            new_filepath = os.path.join(path_out, new_filename)
            
            # Write the merged dataset to a new file.
            ds_merged.to_netcdf(new_filepath)
            
            # Replace the group with the new combined filename.
            updated_files_info.append(new_filename)

            # remove individual files that were combined into one
            for filename in file_group:
                os.remove(path_out + filename)
    
    return updated_files_info





def get_urls_for_each_model_from_json(path_in,experiment_id,variable,frequency):
    """
    reads a json file with all the urls for the data corresponding to each
    model or 'source_id'
    """
    filename_in = 'file_urls_esgf_' + project + '_' + experiment_id + '_' + frequency + '_' + variable + '.json'
    
    with open(path_in[experiment_id]/filename_in, 'r') as f:
        files_info = json.load(f)

    source_ids = list(files_info.keys())

    return source_ids,files_info


def write_out_filenames_to_json(path_out,experiment_id,variable,project,frequency,files_info_regrid,source_id,write2file):

    # create postprocessed filename json
    path_out_local = path_out[experiment_id] + variable + '/'
    filename_out   = 'postprocessed_' + source_id + '_filenames_esgf_' + project + '_' + experiment_id + '_' + frequency + '_' + variable + '.json'

    Path(path_out_local).mkdir(exist_ok=True)

    # write to json
    with open(path_out_local + filename_out, 'w') as f:
        json.dump(files_info_regrid[source_id], f, indent=2)



# main code
if __name__ == "__main__":

    source_ids, files_info = get_urls_for_each_model_from_json(path_in,experiment_id,variable,frequency)

    index = source_ids.index('EC-Earth3-Veg')
    source_ids = source_ids[index:index+1]

    # define new json list of filenames only for regridded/post-processed output
    files_info_regrid = {model: [entry["filename"] for entry in entries] for model, entries in files_info.items()}    

    # loop over models
    if write2file:
        for source_id in source_ids:
            # loop over individual files per model
            for file_info in files_info[source_id]:

                print(file_info)

                download_esgf_data(file_info,path_in[experiment_id])

                files_info_regrid[source_id] = regrid_esgf_data_and_extract_nino34(file_info,files_info_regrid[source_id],path_in[experiment_id],variable,new_grid,time_range)    

            files_info_regrid[source_id] = combine_files_in_time_by_ensemble_member(files_info_regrid[source_id],path_in[experiment_id])
        
            write_out_filenames_to_json(path_out,experiment_id,variable,project,frequency,files_info_regrid,source_id,write2file)



scmip_path = Path('/datalake/NS9560K/ESGF/CMIP6/ScenarioMIP')
hcmip_path = Path('/datalake/NS9560K/ESGF/CMIP6/CMIP')

ssp370_pattern = '*/*/ssp370/*/Amon/psl/*/latest/*.nc'
histor_pattern = '*/*/historical/*/Amon/psl/*/latest/*.nc'

available_scenario = list(scmip_path.glob(ssp370_pattern))
available_historical = list(hcmip_path.glob(histor_pattern))

scenarios = list(set([scen.stem.split('_')[2] + '_' + scen.stem.split('_')[4] for scen in available_scenario]))
historical = list(set([hist.stem.split('_')[2] + '_' + hist.stem.split('_')[4] for hist in available_historical]))

overlap = [scen for scen in scenarios if scen in historical]