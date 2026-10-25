
import xarray as xr
import pandas as pd
import numpy as np
import re
import cartopy.crs as ccrs
import xesmf as xe
from scipy import stats
import proplot as pplt
import pickle
from pathlib import Path
from climate_scenarios_2050.utils import make_season_group, make_season_group_multi
from climate_scenarios_2050.config import proj_base, data_paths, model_id_file
from climate_scenarios_2050.generate_NAO_hindcast import load_model_ensemble_NAO
from climate_scenarios_2050.subsampling_list import Smith_subsampling
from climate_scenarios_2050.filters import butter_lowpass_filter
from climpred import HindcastEnsemble
from dask.distributed import Client, LocalCluster, get_client

client = Client()
fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')


def subsample(ensemble,subsampling,NAO_ref="HadSLP"):

    ensemble = ensemble.copy()

    base_path_search = proj_base/'data/CMIP6_processed'
    
    av_models_id = np.unique(ensemble.m_id)

    with open(model_id_file,'rb') as f:
        model_id = pickle.load(f)
    av_models = [mod for mod,id in model_id.items() if id in av_models_id]

    if 'varsplit' in subsampling:
        obs_split = xr.open_dataarray(proj_base/f'data/{NAO_ref}/{subsampling}_{NAO_ref}_psl_EOF_full_index_DJFM_1850-2014_1.5deg.nc').sel(mode=1)
        # subselect the models based on the amount of low-frequency NAO variance:
        model_selection = []
        for model in av_models:
            vsplit = xr.open_dataarray(base_path_search/f'historical/{model}/EOF_model/{subsampling}_{model}_psl_EOF_{NAO_ref}_index_DJFM_1850-2014_1.5deg.nc').sel(mode=1)
            # if the ratio in the observations is more than 10% larger than in the simulations, drop the model
            # print('{:} obs: {:1.2f} sim: {:1.2f}'.format(model,obs_split.values, vsplit.values))
            criterion = (obs_split - vsplit) < .1
            if criterion:
                model_selection.append(model)
    elif 'NAO_AR' in subsampling:
        # load the simulation list:
        model_sel_file = proj_base/'data/subsampling'/f'{subsampling}_subsampling.pkl'
        with open(model_sel_file, "rb") as fp:
            model_selection = pickle.load(fp)
    elif "Smith" in subsampling:
        model_selection = [amo for amo in av_models if amo in Smith_subsampling[subsampling]]
    else:
        print('{:} subsampling not implemented yet'.format(subsampling))
        return

    model_id_selection = [model_id[model] for model in model_selection]

    return ensemble.sel(member=ensemble.m_id.isel(init_year=0).isin(model_id_selection).values)

def group_field_by_season(ds:xr.Dataset,time_dim:str='time',months_in_season:list=[12, 1, 2, 3]) -> xr.Dataset:
    """
    
    """
    time = ds[f'{time_dim}'].compute()

    # rewrite the years in the time array if the first month in the season comes from the previous year:
    prev_year_months = [mo for mo in months_in_season if mo > months_in_season[-1]]
    if prev_year_months:
        # make an array with the verification year (note that the year tag belongs to the first month in months_in_season
        season_year = xr.where(
            time.dt.month.isin(prev_year_months),
            time.dt.year,
            time.dt.year - 1
        )
    else:
        season_year = time.dt.year

    # mask those values that are not in the requested season:
    seas_mask = time.dt.month.isin(months_in_season)
    
    # use the fact that the months are in the exact same order for each lead time! This will not be valid for other hindcast structures!!
    season_grouper = season_year.where(seas_mask).isel(init_year=0).compute() - season_year.isel(init_year=0).init_year

    return ds.groupby(season_grouper).mean().rename({"group":"lead_time"}).sel(lead_time=slice(0,None))

def align_e5(ds):
    if "valid_time" in list(ds.coords):
        return ds.rename({"valid_time":"time"}).drop_vars(["number","expver"])
    elif "date" in list(ds.coords):
        return ds.assign_coords(time=pd.Timestamp(str(ds.date.values))).expand_dims("time").drop_vars(["number","date","expver"])
    else:
        return ds

fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

roll = 41

hc_dir = Path("/datalake/NS9873K/DATA/CMIP6_hindcast/")
vf_path = Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1")

hc_param_set = {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,}

hc_dir_params = hc_dir/"hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_{vf_win_len}yvf_roll{roll}y".format(**hc_param_set,roll=roll)
forcing = hc_param_set["scen"]

filter_ext = "f-None"

# processing: how to filter/aggregate/average the data
proc_type = 'running_mean'
Nyr = 8

# lead_times: what lead times to verify
lead_times = np.arange(0,36,1)
lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

linestyle = {5:'solid',15:'dashed',25:'dotted',35:'solid'}

variable = "tas" # "pr" # "tas" # "sfcWind" #
var_sh = "t2m" # "tp" # "t2m" # "si10" #
variable_long = "2m_temperature" # "total_precipitation" # "2m_temperature" # "10m_wind_speed" #
int_meth = "bilinear" # "conservative" # "bilinear" #


#----------------------------------------------------------------------------------#
# Load verification data
#----------------------------------------------------------------------------------#
ref_files = sorted(list(vf_path.glob(f"{variable_long}/*.nc")))
ref_field = xr.open_mfdataset(ref_files,preprocess=align_e5,combine="nested",concat_dim="time").rename({"latitude":"lat","longitude":"lon"}).load()
if var_sh == "tp":
    ref_field[var_sh] = ref_field[var_sh]*1e3

# average to multi-winter data
season = make_season_group(ref_field)
ref_field_av = ref_field.groupby(season).mean().drop_sel(year=-1).rolling(year=Nyr,center=True).mean().dropna("year")
# ref_field_av = ref_field.groupby("time.year").mean().rolling(year=Nyr,center=True).mean().dropna("year")


#----------------------------------------------------------------------------------#
# Load CMIP6 "hindcast" fields
#----------------------------------------------------------------------------------#
hc_filename = sorted(list(hc_dir_params.glob(f"hc_{variable}_*.zarr")))
hc_ens_field = xr.open_mfdataset(hc_filename,engine="zarr")
hc_ens_field = hc_ens_field.chunk({"lead_time":-1,"member":100})

# hc_ens_field_grouped = hc_ens_field.groupby("lead_time").mean().chunk({"lead_time":-1})
hc_ens_field_grouped = group_field_by_season(hc_ens_field).chunk({"lead_time":-1})
hc_ens_field_av = hc_ens_field_grouped.rolling(lead_time=Nyr,center=True).mean().chunk({"member":-1})
# manually drop the first Nyr-1 years since they're NaN:
hc_ens_field_av = hc_ens_field_av.isel(lead_time=slice(Nyr//2,None))

# convert field to correct units (mainly important for precipitation):
if variable == "pr":
    hc_ens_field_av[variable] = hc_ens_field_av[variable] * 86400

hc_ens_field_av = hc_ens_field_av.chunk({"lat":"auto","lon":"auto","member":-1,"lead_time":1,"init_year":-1})
# generate the "valid time" variable to align with observations:
# hc_ens_field_av = hc_ens_field_av.assign_coords(valid_time = hc_ens_field.init_year + hc_ens_field_av.lead_time)

# re-grid validation data to match hindcasts:
to_grid = xr.Dataset({"lat": hc_ens_field_av.lat,"lon": hc_ens_field_av.lon,})
regridder = xe.Regridder(ref_field_av, to_grid, method=int_meth, ignore_degenerate=True, periodic=True)

ref_field_av_rgr = regridder(ref_field_av)

# load MMEM forced signal to subtract from the verification data:
mmem = xr.open_dataarray(proj_base/f"data/CMIP6/forced/mmem-{variable}_hist-{forcing}_{filter_ext}.nc")
mmem_ann = mmem.groupby("time.year").mean()

mmem_normed_obsper = mmem_ann.sel(year=ref_field_av_rgr.year) - mmem_ann.sel(year=ref_field_av_rgr.year).mean("year")
ref_field_normed = ref_field_av_rgr - ref_field_av_rgr.mean("year")

# compute the forced signal in the observations by regressing NAO onto the MMM
forced_signal_obs = xr.cov(ref_field_normed[var_sh],mmem_normed_obsper,dim="year")/mmem_normed_obsper.var() * mmem_normed_obsper

# remove the forced signal from observations:
ref_field_int = ref_field_normed - forced_signal_obs

# Compute model means
# hc_ens_field_av_mmean = hc_ens_field_av.sel(init_year=slice(1980,2010)).groupby("m_id").mean(["member","init_year"]).compute()

# # Subtract model means → anomalies per member
# hc_ens_field_av_an = hc_ens_field_av.groupby("m_id") - hc_ens_field_av_mmean
# hc_ens_field_av_an = hc_ens_field_av_an.chunk({"lat":"auto","lon":"auto","member":-1,"lead_time":1,"init_year":-1})

# ref_field_av_rgr_an = ref_field_av_rgr - ref_field_av_rgr.sel(year=slice(1980,2010)).mean("year")


hc_ens_field_av_an = hc_ens_field_av.assign_coords(time = hc_ens_field_av.init_year + hc_ens_field_av.lead_time)
hc_ens_field_av_an_natl = hc_ens_field_av_an[variable].sel(lat=[55,71],lon=[-11,25]).mean(["lat","lon"]).compute()
ref_field_av_rgr_natl = ref_field_int[var_sh].sel(lat=[55,71],lon=[-11,25]).mean(["lat","lon"])

quants = hc_ens_field_av_an_natl.quantile([.05,.95],dim="member")
# LT = 7
fig,axs = pplt.subplots(figwidth=5,figheight=4,suptitle="Prediction N.Atl. {:}".format(" ".join(variable_long.split('_'))))
for LT in [5,15,25]:
    ens_mean_lt = hc_ens_field_av_an_natl.mean(["member"]).sel(lead_time=LT)
    sel_iy = ens_mean_lt.swap_dims({"init_year":"time"}).init_year
    sel_iy = sel_iy.sel(time=ref_field_av_rgr_natl.year)
    axs.fill_between(hc_ens_field_av_an_natl.time.sel(lead_time=LT,init_year=sel_iy),quants.sel(quantile=0.05,lead_time=LT,init_year=sel_iy),quants.sel(quantile=0.95,lead_time=LT,init_year=sel_iy),alpha=.2)
    ens_mean_lt.sel(init_year=sel_iy).plot(ax=axs,x="time")

ref_field_av_rgr_natl.plot(color="k",ax=axs)
fig.savefig(fpath/f"field_eval/natl_{variable_long}.png",dpi=300,bbox_inches="tight")


# wrap into a climpred dataset:
hindcasts = HindcastEnsemble(hc_ens_field_av.where(hc_ens_field_av.init_year >= ref_field_av_rgr.year.min(), drop=True).rename({"init_year":"init","lead_time":"lead"}))
hindcasts = hindcasts.add_observations(ref_field_int.rename({"year":"time",var_sh:variable}))

# evaluate
acc = hindcasts.verify(metric="acc", comparison="e2o", dim="init", alignment="same_verif").load()
# rps = hindcasts.verify(metric="rps", comparison="m2o", dim="member", alignment="same_verif").load()

# plot global skill:
for LT in [5,10,15,20,25,30]:
    fig,axs = pplt.subplots(figwidth=5,figheight=4,suptitle="ACC {:} (LT {:}Y)".format(" ".join(variable_long.split('_')),LT),proj='robin')
    cf = axs.pcolormesh(acc.lon, acc.lat, acc.sel(lead=LT)[variable], extend='both',transform=ccrs.PlateCarree())
    fig.colorbar(cf, loc='b', label=variable, extendsize='1.7em')
    axs.format(latlines=30, lonlines=60,coast=True)
    fig.savefig(fpath/f"field_eval/acc_{variable_long}_LY{LT}.png",dpi=300,bbox_inches="tight")


# rmse = hindcasts.verify(metric="rmse", comparison="e2o", dim="init", alignment="same_verif").load()
# for LT in [5,15,25,35]:
#     fig,axs = pplt.subplots(figwidth=5,figheight=4,suptitle="RMSE {:} (LT {:}Y)".format(" ".join(variable_long.split('_')),LT),proj='robin')
#     cf = axs.pcolormesh(rmse.lon, rmse.lat, rmse.sel(lead=LT)[variable], vmax=1, extend='max',transform=ccrs.PlateCarree())
#     fig.colorbar(cf, loc='b', label=variable, extendsize='1.7em')
#     axs.format(latlines=30, lonlines=60,coast=True)
# fig.savefig(fpath/f"field_eval/rmse_{variable_long}_LY{LT}.png",dpi=300,bbox_inches="tight")
