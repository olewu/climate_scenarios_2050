"""
Update the merged HadSLP/ERA5 internal NAO index
(`internal-NAO-variability-HadSLP-ERA5-merged-1980.nc`, written by `merge_NAO.py`)
once new ERA5 mean sea level pressure data is available.

The merged index is the end of a three-step chain:
 1) `cmip6_nao.py` (ref_obs="ERA5"): DJFM ERA5 SLP projected onto the ERA5 EOF1 fitted over
    1950-2015 -> `ERA5_psl_EOF_full_index_DJFM_<first>-<last>_1.5deg.nc`
 2) `split_variability.py` (obs_ds="ERA5"): forced NAO signal removed by regressing the index on
    the CMIP6 multi-model ensemble mean (MMEM) -> `internal-NAO-variability_f-None.nc`
 3) `merge_NAO.py`: ERA5 internal NAO rescaled to the HadSLP mean/std over the overlap period,
    HadSLP until 1979, ERA5 from 1980 on.

This script redoes the ERA5 part of the chain with all *fitted* quantities frozen
(EOF pattern and its standardization, centering period, forced-signal regression coefficient
and the HadSLP harmonization), so that the existing years are reproduced and only new years
are appended. This keeps the merged index consistent with anything that has already been fitted
or ranked with it. Set `split_fit_period = None` to refit the forced-signal regression with all
available years instead (this changes the whole history slightly, like rerunning steps 2 and 3).

A DJFM season with year label Y (December Y + Jan-Mar Y+1) is only added once all four months exist.
"""

import shutil
import numpy as np
import pandas as pd
import xarray as xr
import xeofs as xe
from climate_scenarios_2050.config import *
from climate_scenarios_2050.cmip6_nao import process_monthly_field
from climate_scenarios_2050.utils import load_model_ensemble_NAO, ext_from_months


def prep_ERA5_mon_clean(ds):
    """
    like `cmip6_nao.prep_ERA5_mon`, but also drop `expver` and `number` which come with
    recent CDS downloads and break the concatenation of old and new files
    """
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time":"time"})
    return ds.drop_vars(["expver","number"],errors="ignore")

def last_complete_season(files,months,preprocess=None):
    """
    find the last season year (year of December, as in `make_season_group`)
    for which all months in `months` are available in `files`
    """
    times = []
    for fl in files:
        with xr.open_dataset(fl) as ds:
            if preprocess is not None:
                ds = preprocess(ds)
            times.append(pd.DatetimeIndex(ds["time"].values))
    times = pd.DatetimeIndex(np.concatenate(times))
    available = set(zip(times.year,times.month))

    for year in range(times.year.max(),times.year.min()-1,-1):
        # months before December belong to the following calendar year:
        if all(((year + int(mon < 12)),mon) in available for mon in months):
            return year
    raise ValueError("no complete season found")

def era5_eof_index(files,eof_model_file,start_year,last_year,normal_period,months,grid_spacing,domain):
    """
    project DJFM ERA5 SLP onto the saved (!) ERA5 EOF model, as in `cmip6_nao.py`
    returns all modes, standardized with the scores over the EOF fitting period
    """
    field = process_monthly_field(
        files_to_process=files,
        variable="msl",
        interpolate=True,
        interpolate_options={'grid_spacing':grid_spacing,'method':'bilinear'},
        concat_dim="time",
        unit_conversion_factor=1/100, # Pa to hPa
        start_sel=start_year,
        end_sel=last_year+1, # cut_dataarray keeps season years up to end_sel - 1
        months=months,
        domain=domain,
        time_dim="time",
        center=normal_period,
        return_global=False,
        preprocess=prep_ERA5_mon_clean,
    )
    for crd in ["member","number","expver"]:
        if crd in field.coords:
            field = field.squeeze().drop_vars(crd)

    # never refit, otherwise the existing years change:
    eof_model = xe.single.EOF.load(eof_model_file)

    # same sign convention as in cmip6_nao.py:
    if eof_model.components(normalized=False).sel(mode=1,lat=35.25,lon=-20.25) < 0:
        pattern_invert_factor = -1
    else:
        pattern_invert_factor = 1

    full_index = pattern_invert_factor * eof_model.transform(field,normalized=False).compute()
    PC_mu = eof_model.scores().mean(dim="year")
    PC_sigma = eof_model.scores().std(dim="year")
    return ((full_index - PC_mu)/PC_sigma).compute()

def load_mmem(forcing,ref_ds="ERA5"):
    """
    NAO multi-model ensemble mean, computed as in `split_variability.split_variability_index`
    """
    ensemble = load_model_ensemble_NAO(None,forcing,ref_ds=ref_ds)["scores"]
    years_full = np.arange(ensemble.year.min(),ensemble.year.max()+1)
    ensemble = ensemble.reindex(year=years_full).interpolate_na(dim="year",method="linear")
    return ensemble.mean("member").compute()

def split_obs_index(index_obs,mmem,fit_years=None):
    """
    split the observed index into forced and internal variability as in
    `split_variability.split_variability_index`, but with means and regression
    coefficient estimated over `fit_years` only (all years if None)
    """
    if fit_years is None:
        fit_years = index_obs.year.values
    fit = index_obs.year.isin(fit_years)

    mmem_obsper = mmem.sel(year=index_obs.year)
    mmem_normed = mmem_obsper - mmem_obsper.sel(year=fit).mean("year")
    obs_mean = index_obs.sel(year=fit).mean("year")
    index_normed = index_obs - obs_mean

    # note: cov uses ddof=1 and var ddof=0 as in split_variability.py (kept to reproduce the existing files)
    beta = xr.cov(index_normed.sel(year=fit),mmem_normed.sel(year=fit))/mmem_normed.sel(year=fit).var()
    forced = beta * mmem_normed

    return index_normed - forced, forced + obs_mean

def merge_hadslp_era5(nao_hadslp,nao_e5,merge_year):
    """
    rescale ERA5 to HadSLP over the overlap period and merge, as in `merge_NAO.py`
    """
    overlap = slice(int(nao_e5.year.min()),int(nao_hadslp.year.max()))
    nao_e5_ref = nao_e5.sel(year=overlap)
    nao_hadslp_ref = nao_hadslp.sel(year=overlap)

    nao_e5_harm = (nao_e5 - nao_e5_ref.mean())*(nao_hadslp_ref.std()/nao_e5_ref.std()) + nao_hadslp_ref.mean()

    return xr.concat([nao_hadslp.sel(year=slice(merge_year-1)),nao_e5_harm.sel(year=slice(merge_year,None))],dim="year")

def backup(path,tag):
    if path.exists():
        backup_path = path.with_name(f"{path.stem}_backup-{tag}{path.suffix}")
        shutil.copy(path,backup_path)
        print(f"backed up {path.name} to {backup_path.name}")


if __name__ == "__main__":

    #---------------SETTINGS--------------#
    # need to match the settings used in cmip6_nao.py/split_variability.py/merge_NAO.py:
    gsp = 1.5
    grid_spacing = {'lat':gsp,'lon':gsp}
    variable = "psl"
    NAO_months = [12,1,2,3]
    mnth_ext = ext_from_months(NAO_months)
    start_year = 1850
    normal_period = [1950,2015] # EOF fitting and centering period
    domain = NAO_domain['winter']
    merge_year = 1980

    # scenario whose NAO MMEM was used to remove the forced signal from ERA5 in split_variability.py
    # (the ERA5 split file does not carry the scenario in its name, so check this!):
    forcing = "ssp245"

    # years used to fit the forced-signal regression (frozen to the years of the existing file);
    # set to None to refit with all available years
    split_fit_period = [1939,2024]

    # maximum tolerated difference to the existing merged index in the overlapping years
    tolerance = 1e-6

    # also overwrite the ERA5 internal/forced NAO files used by e.g. linear_model_wsp_NAO.py:
    update_era5_split_files = False

    #---------------PATHS--------------#
    era5_files = sorted(Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/mean_sea_level_pressure/").glob("*.nc"))
    eof_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/mslp_EOF")
    eof_model_file = eof_dir/f"ERA5_{variable}_EOF{mnth_ext}_{normal_period[0]}-{normal_period[1]}_{gsp}deg.zarr"
    era5_split_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
    hadslp_internal_file = proj_base/"data/HadSLP/internal-NAO-variability_f-None.nc"
    merged_file = Path("/datalake/NS9873K/DATA/CMIP6/wind_db/internal-NAO-variability-HadSLP-ERA5-merged-1980.nc")

    #---------------1) ERA5 EOF NAO--------------#
    last_year = last_complete_season(era5_files,NAO_months,preprocess=prep_ERA5_mon_clean)
    print(f"last complete DJFM season in ERA5: {last_year} (Dec {last_year} - Mar {last_year+1})")

    full_index = era5_eof_index(era5_files,eof_model_file,start_year,last_year,normal_period,NAO_months,grid_spacing,domain)
    ix_first = int(full_index.year.min())
    ix_last = int(full_index.year.max())
    full_index_file = eof_dir/f"ERA5_{variable}_EOF_full_index{mnth_ext}_{ix_first}-{ix_last}_{gsp}deg.nc"
    backup(full_index_file,"previous")
    full_index.to_netcdf(full_index_file,mode="w")
    nao_e5_full = full_index.sel(mode=1).squeeze()

    #---------------2) remove forced signal--------------#
    mmem = load_mmem(forcing,ref_ds="ERA5")
    fit_years = None if split_fit_period is None else np.arange(split_fit_period[0],split_fit_period[-1]+1)
    nao_e5_int, nao_e5_forced = split_obs_index(nao_e5_full,mmem,fit_years)

    if update_era5_split_files:
        for name,da in zip(["internal","forced"],[nao_e5_int,nao_e5_forced]):
            split_file = era5_split_dir/f"{name}-NAO-variability_f-None.nc"
            backup(split_file,ix_last)
            da.to_netcdf(split_file,mode="w")

    #---------------3) harmonize and merge--------------#
    with xr.open_dataarray(hadslp_internal_file) as da:
        nao_hadslp = da.load()
    merged_nao = merge_hadslp_era5(nao_hadslp,nao_e5_int,merge_year)

    # check that the existing years are reproduced:
    with xr.open_dataarray(merged_file) as da:
        merged_old = da.load()
    common_years = np.intersect1d(merged_old.year,merged_nao.year)
    max_diff = float(np.abs(merged_old.sel(year=common_years).values - merged_nao.sel(year=common_years).values).max())
    new_years = np.setdiff1d(merged_nao.year,merged_old.year)
    print(f"max. difference to the existing merged index over {common_years.min()}-{common_years.max()}: {max_diff:.2e}")
    print(f"new years: {new_years}")

    if (split_fit_period is not None) and (max_diff > tolerance):
        raise RuntimeError(
            "the existing merged index is not reproduced; check `forcing`, `split_fit_period` "
            "and whether the ERA5 data in the fitting periods has changed. Nothing has been written."
        )

    backup(merged_file,int(merged_old.year.max()))
    merged_nao.to_netcdf(merged_file,mode="w")
    print(f"wrote {merged_file} ({int(merged_nao.year.min())}-{int(merged_nao.year.max())})")
