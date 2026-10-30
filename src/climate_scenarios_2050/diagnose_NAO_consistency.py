"""
Find out at which step of the NAO chain (cmip6_nao.py -> split_variability.py -> merge_NAO.py)
the recomputed index differs from the existing merged index
`internal-NAO-variability-HadSLP-ERA5-merged-1980.nc`.

Cheapest checks first:
 A) merge: do the internal NAO files currently on disk reproduce the merged index?
    (they are overwritten every time split_variability.py runs, for every scenario in the loop)
 B) split: which scenario MMEM reproduces the internal NAO files on disk from the full EOF indexes on disk?
 C) EOF: which EOF fitting/centering period reproduces the full ERA5 EOF index on disk?
"""

import numpy as np
import xarray as xr
from climate_scenarios_2050.config import *
from climate_scenarios_2050.cmip6_nao import process_monthly_field, get_EOF_model
from climate_scenarios_2050.utils import ext_from_months
from climate_scenarios_2050.update_merged_NAO import prep_ERA5_mon_clean, load_mmem, split_obs_index, merge_hadslp_era5


def compare(name,new,old):
    new, old = xr.align(new.squeeze(drop=True),old.squeeze(drop=True),join="inner")
    diff = new - old
    print(
        f"{name:<50s} {int(new.year.min())}-{int(new.year.max())} "
        f"corr={float(xr.corr(new,old)):.5f} rmse={float((diff**2).mean()**.5):.2e} "
        f"max|diff|={float(np.abs(diff).max()):.2e} mean diff={float(diff.mean()):+.2e} "
        f"std ratio={float(new.std()/old.std()):.4f}"
    )

def open_da(path,var=None):
    with xr.open_dataset(path) as ds:
        da = ds[var] if var is not None else ds[list(ds.data_vars)[0]]
        return da.load()


if __name__ == "__main__":

    gsp = 1.5
    grid_spacing = {'lat':gsp,'lon':gsp}
    NAO_months = [12,1,2,3]
    mnth_ext = ext_from_months(NAO_months)
    merge_year = 1980
    forcings = ["ssp119","ssp126","ssp245","ssp370","ssp585"]
    # candidate EOF fitting (and centering) periods:
    candidate_periods = [[1950,2015],[1970,2015],[1979,2015],[1950,2014],[1940,2024],[1961,1990],[1981,2010]]
    run_eof_check = True # C) needs the ERA5 SLP fields and takes a while

    era5_eof_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/mslp_EOF")
    era5_split_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
    hadslp_dir = proj_base/"data/HadSLP"
    merged_file = Path("/datalake/NS9873K/DATA/CMIP6/wind_db/internal-NAO-variability-HadSLP-ERA5-merged-1980.nc")

    merged_old = open_da(merged_file)
    era5_full_old = open_da(era5_eof_dir/f"ERA5_psl_EOF_full_index{mnth_ext}_1939-2024_{gsp}deg.nc","scores").sel(mode=1)
    hadslp_full_old = open_da(hadslp_dir/f"HadSLP_psl_EOF_full_index{mnth_ext}_1850-2014_{gsp}deg.nc","scores").sel(mode=1)
    era5_int_disk = open_da(era5_split_dir/"internal-NAO-variability_f-None.nc")
    hadslp_int_disk = open_da(hadslp_dir/"internal-NAO-variability_f-None.nc")

    #---------------A) merge step--------------#
    print("\nA) merged index vs. internal NAO files currently on disk")
    compare("HadSLP part (<1980) vs HadSLP internal file",hadslp_int_disk.sel(year=slice(merge_year-1)),merged_old.sel(year=slice(merge_year-1)))
    compare("merge_NAO.py applied to files on disk",merge_hadslp_era5(hadslp_int_disk,era5_int_disk,merge_year),merged_old)

    #---------------B) split step--------------#
    print("\nB) internal NAO recomputed from the full EOF indexes on disk, by scenario MMEM")
    for obs_name,full_old,int_disk in [("ERA5",era5_full_old,era5_int_disk),("HadSLP",hadslp_full_old,hadslp_int_disk)]:
        for forcing in forcings:
            try:
                mmem = load_mmem(forcing,ref_ds=obs_name)
            except Exception as e:
                print(f"{obs_name} {forcing}: could not load MMEM ({e})")
                continue
            int_new, _ = split_obs_index(full_old,mmem)
            compare(f"{obs_name} internal, {forcing} MMEM vs file on disk",int_new,int_disk)
            if obs_name == "ERA5":
                compare(f"  -> merged with HadSLP file on disk vs merged",merge_hadslp_era5(hadslp_int_disk,int_new,merge_year),merged_old)

    #---------------C) EOF step--------------#
    if run_eof_check:
        print("\nC) full ERA5 EOF index recomputed with different EOF periods vs file on disk")
        era5_files = sorted(Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/mean_sea_level_pressure/").glob("*.nc"))
        field = process_monthly_field(
            files_to_process=era5_files,
            variable="msl",
            interpolate=True,
            interpolate_options={'grid_spacing':grid_spacing,'method':'bilinear'},
            concat_dim="time",
            unit_conversion_factor=1/100,
            start_sel=1850,
            end_sel=int(era5_full_old.year.max())+1,
            months=NAO_months,
            domain=NAO_domain['winter'],
            time_dim="time",
            center=[], # centered below for each candidate period
            return_global=False,
            preprocess=prep_ERA5_mon_clean,
        )
        for crd in ["member","number","expver"]:
            if crd in field.coords:
                field = field.squeeze().drop_vars(crd)
        field = field.compute()

        # the saved EOF model (what update_merged_NAO.py uses), centered over its 1950-2015 period:
        saved_model_file = era5_eof_dir/f"ERA5_psl_EOF{mnth_ext}_1950-2015_{gsp}deg.zarr"
        candidates = [("saved EOF model 1950-2015",[1950,2015],saved_model_file)]
        candidates += [(f"EOF refitted {p[0]}-{p[1]}",p,None) for p in candidate_periods]

        for name,period,saved_model in candidates:
            field_c = field - field.sel(year=slice(*period)).mean("year")
            if saved_model is not None:
                eof_model = get_EOF_model(field_c.sel(year=slice(*period)),detrend_field=True,sample_dims={'time':'year'},saved_model=saved_model,overwrite=False)
            else:
                eof_model = get_EOF_model(field_c.sel(year=slice(*period)),detrend_field=True,sample_dims={'time':'year'},saved_model=None)
            sign = -1 if eof_model.components(normalized=False).sel(mode=1,lat=35.25,lon=-20.25) < 0 else 1
            idx = sign * eof_model.transform(field_c,normalized=False)
            idx = (idx - eof_model.scores().mean("year"))/eof_model.scores().std("year")
            compare(name,idx.sel(mode=1),era5_full_old)
