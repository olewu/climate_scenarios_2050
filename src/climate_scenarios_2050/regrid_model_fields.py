# go through all simulations and interpolate them to a common grid (for easier merging as hindcast ensemble):
# get all fields from `/projects/NS9873K/owul/projects/climate_scenarios_2050/data/CMIP6/` 
# regrid to 2 x 2 degree and save in `/projects/NS9873K/owul/projects/climate_scenarios_2050/data/CMIP6_processed/`
# load all ensemble members for a single variable and do the regridding (different interp for different variables)
# save the ensemble for each model in a single zarr store. Makes it easier to add new simulations or variables
# the final hindcasts can then combine variables and models in the same store (one hindcast zarr store for each parameter set)

from climate_scenarios_2050.interpolate import interpolate_grid
from climate_scenarios_2050.config import *
import pandas as pd
from shutil import rmtree
import re
import cftime

def regrid_model_variable(model,variable,forcing,grid,overwrite=False):

    # load all ensemble members for a single model, forcing and variable:
    mod_var_path = database_path/f"{forcing}/{model}/{variable}"
    mod_var_files = sorted(list(mod_var_path.glob("*.nc"))) # make sure to pass a sorted list

    # make sure to sort out files with projections after 2100:
    if forcing in ["ssp245","ssp370"]:
        new_mvf = []
        for f in mod_var_files:
            srch = re.search(r"_(\d{6})-\d{6}$",f.stem)
            strty = int(srch.group(1)[:4])
            if strty < 2020:
                new_mvf.append(f)
        mod_var_files = new_mvf

    if mod_var_files:
        # add ensemble member encoding:
        with open(model_id_file,'rb') as f:
            model_id = pickle.load(f)

        ens_id = []; global_mem_num = []
        for mova_fl in mod_var_files:
            ENS_ID = re.search(r"_(r\d{1,3}i\d{1,2}p\d{1,2}f\d{1,2})_",mova_fl.stem).group(1)
            ens_id.append(ENS_ID)
            global_mem_num.append(model_id[model]+ENS_ID)

        new_fl_name = mod_var_files[0].stem.replace("_"+ens_id[0],"")

        outname = data_paths["processed"].parent/f"CMIP6_processed/{forcing}/{model}/{variable}/{new_fl_name}.zarr"

        if (outname.exists()) and (not overwrite):
            print(f"{outname} already exists.")
            return

        ds_regr = interpolate_grid(mod_var_files,grid_spacing=grid,variable=variable,engine="netcdf4",method=interpolate_method[variable],to_stdgrid=True)

        if not isinstance(ds_regr.indexes['time'],pd.DatetimeIndex):
            # restrict future period to 2100
            if forcing in ["ssp245","ssp370"]:
                cls = type(ds_regr.indexes["time"][0])
                msk = (ds_regr.indexes['time'] >= cls(2015,1,1)) & (ds_regr.indexes['time'] < cls(2101,1,1))

                ds_regr = ds_regr.isel(time=msk)
                
                # if isinstance(ds_regr.time,cftime.DatetimeNoLeap):
                #     ds_regr = ds_regr.sel(time=slice(cftime.DatetimeNoLeap(2015, 1, 1),cftime.DatetimeNoLeap(2101, 1, 1)))
                # elif isinstance(ds_regr.time,cftime.Datetime360Day):
                #     ds_regr = ds_regr.sel(time=slice(cftime.Datetime360Day(2015, 1, 1),cftime.Datetime360Day(2101, 1, 1)))
                # elif isinstance(ds_regr.time,cftime.DatetimeProlepticGregorian):
                #     ds_regr = ds_regr.sel(time=slice(cftime.DatetimeProlepticGregorian(2015, 1, 1),cftime.DatetimeProlepticGregorian(2101, 1, 1)))
                # elif isinstance(ds_regr.time,cftime.DatetimeGregorian):
                #     ds_regr = ds_regr.sel(time=slice(cftime.DatetimeGregorian(2015, 1, 1),cftime.DatetimeGregorian(2101, 1, 1)))
            try:
                ds_regr['time'] = ds_regr.indexes['time'].to_datetimeindex()
            except:
                ds_regr = ds_regr.assign_coords(time=pd.to_datetime(ds_regr.indexes['time'].values))


        # normalize the month labeling:
        normalized_time = pd.to_datetime([t.replace(day=1, hour=0, minute=0, second=0) for t in pd.to_datetime(ds_regr["time"])])

        ds_regr = ds_regr.assign_coords(time=normalized_time).sortby("time")
        
        # restrict to common period (in case it ahsen't been done earlier)
        ds_regr = ds_regr.sel(time=slice(pd.Timestamp(1850, 1, 1),pd.Timestamp(2101, 1, 1)))

        ds_regr = ds_regr.assign_coords({"member":global_mem_num})

        # drop extra coordinates that are unwanted:
        drop_coords = [coo for coo in list(ds_regr.coords) if coo not in ["time","lat","lon","lat_bnds","lon_bnds","member"]]
        ds_regr = ds_regr.drop_vars(drop_coords)


        # re-chunk:
        chunk_scheme = {
            "member":5,
            "time":660,
            "lat":60,
            "lon":60,
        }

        ds_regr = ds_regr.chunk(chunk_scheme)

        # remove the old zarr store if there is one and it's supposed to be overwritten!
        if outname.exists():
            rmtree(outname)

        ds_regr.to_dataset(name=variable).to_zarr(outname, mode="w")
    else:
        print(f"cannot find any simulations for {model} {variable} {forcing}.")
        return

def main(variable,forcing,grid,overwrite=False):
    """
    """
    # iterate trhough all models for a given forcing:
    all_models_for_forcing = sorted([pth.parent.stem for pth in database_path.glob(f"{forcing}/*/{variable}") if pth.parent.stem not in ["OLD","url_repo"]])

    for model in all_models_for_forcing:

        # if (model in ["CNRM-ESM2-1","CNRM-CM6-1-HR","CNRM-CM6-1","CMCC-ESM2","CMCC-CM2-SR5","CESM2-WACCM","CESM2","CAS-ESM2-0","CAMS-CSM1-0","BCC-CSM2-MR","AWI-CM-1-1-MR","ACCESS-ESM1-5","ACCESS-CM2"]):# and (variable == "tas"):
        # if model != "FGOALS-g3":
        #     continue

        print(f"processing {model} {variable} {forcing}")
        try:
            regrid_model_variable(model,variable,forcing,grid,overwrite)
        except Exception as e:
            print(f"\n failed on {model} {variable} {forcing}\nwith {e}")
            

if __name__ == "__main__":
    
    grid = {"lat":2,"lon":2}
    # all the bilinearly regridded fields are relatively fast:
    for variable in ["tas"]: #["psl"]: # ["tas", "rsds", "sfcWind", "psl"]:# ["pr"]: #
        for forcing in ["historical","ssp245","ssp370"]: # ["ssp370"]:# , 
            main(variable,forcing,grid,overwrite=True)