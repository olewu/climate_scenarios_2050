# Compute NAO index from all available CMIP6 runs

import xarray as xr
import xeofs as xe
import numpy as np
import pandas as pd
from scipy import signal

import proplot as pplt

import xeofs as xe

from climate_scenarios_2050.config import *
from climate_scenarios_2050.utils import load_verification, find_coordnames, make_season_group, make_season_group_multi, lko_mean, rm_tree
# from joblib import Parallel, delayed

def compute_ref_nao(
        ref_type:str,
        period:list,
        ref_period:list=[1961,2015],
        N_run:int=1,
        var:str='mean_sea_level_pressure',
        NAO_months:list=[12,1,2,3],
    ):
    """
    each NAO value is a N_run-year running average of the previous N_run year (including the tagged year)
    """
    
    ds = load_verification(var,period,ref_type)

    ds_coords = find_coordnames(ds)

    # compute SLP averages in centers of action (absolute monthly data):
    northern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_box['north']['latitude'],ds_coords['lon_dim']:NAO_box['north']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()
    
    southern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_box['south']['latitude'],ds_coords['lon_dim']:NAO_box['south']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()
    # since the areas are relatively small, area weighting has a negligible effect

    if NAO_months:
        # filter to winter averages (DJFM):
        season = make_season_group(ds,months_in_season=NAO_months)
        northern_seas = northern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
        southern_seas = southern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
    else:
        northern_seas = northern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
        southern_seas = southern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*ref_period))

    # compute annual winter NAO index
    NAO_raw = southern_seas - northern_seas
    # normalize to 0 mean:
    # NAO_anom = NAO_raw - NAO_raw.mean()

    # calculate anomalies wrt a leave-k-year-out mean of the NAO for fairer 
    # comparison to a forecast situation where year under consideration is unavailble:
    NAO_lko_anom = NAO_raw - lko_mean(NAO_raw[var_name_map['long_to_cf'][var]],N_run)
    # NAO_lko_anom = NAO_raw - NAO_raw[var_name_map['long_to_cf'][var]].mean(('year'))

    # compute rolling average (time tag refers to last year in the window)
    # NAO_anom_Ny_run = NAO_anom.rolling(year=N_run).mean().dropna('year')
    NAO_lko_anom_Ny_run = NAO_lko_anom.rolling(year=N_run).mean().dropna('year')
    
    return NAO_lko_anom_Ny_run

# compute reference NAO index based on two area averages ("point-wise")
def NAO_pt_wise(
        ref_type:str,
        period:list,
        normal_period:list=None,
        normalization_leave_out:int=0, # only works if 
        N_run:int=1,
        var:str='mean_sea_level_pressure',
        NAO_months:list=[12,1,2,3],
        NAO_lims:dict=NAO_box,
    ):

    #-------LOAD-------#
    # load reference data for given period:
    ds = load_verification(var,period,ref_type)
    # derive coordinate names:
    ds_coords = find_coordnames(ds)

    #-------AREA AVERAGE-------#
    # compute area averages:
    northern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_lims['north']['latitude'],ds_coords['lon_dim']:NAO_lims['north']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()
    
    southern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_lims['south']['latitude'],ds_coords['lon_dim']:NAO_lims['south']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()

    #-------SEASON SUBSELECT-------#
    # subselect to specified season:
    if NAO_months:
        # filter to winter averages (DJFM):
        season = make_season_group(ds,months_in_season=NAO_months)
        northern_seas = northern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*period))
        southern_seas = southern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*period))
    else:
        northern_seas = northern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*period))
        southern_seas = southern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*period))


    #-------COMPUTE NAO INDEX-------#
    NAO_index = southern_seas - northern_seas

    #-------NORMALIZE-------#
    # normalize based on specified normal period:
    if normal_period is not None:
        # normalize over normal period:
        NAO_index = NAO_index - NAO_index.sel(year=slice(*normal_period)).mean('year')
    else:
        NAO_index = NAO_index - lko_mean(NAO_index.sel(year=slice(*normal_period))[var_name_map['long_to_cf'][var]],normalization_leave_out)

    #-------RUNNING AVERAGE-------#
    # compute running average:
    NAO_index = NAO_index.rolling(year=N_run).mean().dropna('year')

    return NAO_index


def cut_dataarray(
        input_dataarray,
        period,
        domain=NAO_domain['winter'],
        months:list=[12,1,2,3],
        time_dim="time",
        member_dim="member",
        return_global=True
    ):
    """
    Cut the dataarray to the desired domain and months
    """

    da = regularize_grid(input_dataarray)

    da_reg = da.cf.sel(
        {
            "lat":domain['latitude'],
            "lon":domain['longitude']
        }
    ) # 3.5min (50mem)

    #-------SEASON SUBSELECT-------#
    # subselect to specified season:
    if isinstance(input_dataarray.coords[time_dim].to_index(), pd.MultiIndex):
        if months:
            # filter to winter averages (DJFM):
            season = make_season_group_multi(da,months_in_season=months)
            da_reg_seas = da_reg.groupby(season).mean(time_dim).persist()
            da_reg_seas = da_reg_seas.unstack().sel(year=slice(period[0],period[-1]-1))
            da_seas = da.groupby(season).mean(time_dim).persist()
            da = da_seas.unstack().sel(year=slice(period[0],period[-1]-1))
    else:
        if months:
            # filter to winter averages (DJFM):
            season = make_season_group(da,months_in_season=months)
            da_reg_seas = da_reg.groupby(season).mean(time_dim).sel(year=slice(period[0],period[-1]-1))
            da = da.groupby(season).mean(time_dim).sel(year=slice(period[0],period[-1]-1)).chunk({'year':-1,'lat':-1,'lon':-1})
        else:
            da_reg_seas = da_reg.groupby(time_dim+'.year').mean(time_dim).sel(year=slice(*period))
            da = da.groupby(time_dim+'.year').mean(time_dim).sel(year=slice(*period)).chunk({'year':-1,'lat':-1,'lon':-1})

    if return_global:
        return da_reg_seas, da
    else:
        return da_reg_seas

# EOF-based definition of the NAO index:
def NAO_eof_based(
        input_dataarray:xr.DataArray,
        period:list,
        normal_period:list=None,
        # normalization_leave_out:int=0, # only works if 
        # N_run:int=1,
        months:list=[12,1,2,3],
        domain=NAO_domain['winter'],
        eof_mode:int=1,
        global_corr=False,
        time_dim="time",
        member_dim="member",
        saved_model=None,
        overwrite=False,
        # rotation:bool=True
    ):
    
    if (saved_model is not None) and (not isinstance(saved_model,Path)):
        saved_model = Path(saved_model)

    if isinstance(input_dataarray.coords[time_dim].to_index(), pd.MultiIndex):
        sample_dims = (member_dim,'year')
    else:
        sample_dims = ('year',)

    da_reg_seas, da = cut_dataarray(input_dataarray,period=period,domain=domain,months=months,time_dim=time_dim,member_dim=member_dim,return_global=True)

    # detrend:

    # # restrict to North Atlantic region:
    # da_reg = da.cf.sel(
    #     {
    #         "lat":domain['latitude'],
    #         "lon":domain['longitude']
    #     }
    # ) # 3.5min (50mem)

    # #-------SEASON SUBSELECT-------#
    # # subselect to specified season:
    # if isinstance(input_dataarray.coords[time_dim].to_index(), pd.MultiIndex):
    #     sample_dims = (member_dim,'year')
    #     if months:
    #         # filter to winter averages (DJFM):
    #         season = make_season_group_multi(da,months_in_season=months)
    #         da_reg_seas = da_reg.groupby(season).mean(time_dim).persist()
    #         da_reg_seas = da_reg_seas.unstack().sel(year=slice(period[0],period[-1]-1))
    #         da_seas = da.groupby(season).mean(time_dim).persist()
    #         da = da_seas.unstack().sel(year=slice(period[0],period[-1]-1))
    # else:
    #     sample_dims = ('year',)
    #     if months:
    #         # filter to winter averages (DJFM):
    #         season = make_season_group(da,months_in_season=months)
    #         da_reg_seas = da_reg.groupby(season).mean(time_dim).sel(year=slice(period[0],period[-1]-1))
    #         da = da.groupby(season).mean(time_dim).sel(year=slice(period[0],period[-1]-1)).chunk({'year':-1,'lat':-1,'lon':-1})
    #     else:
    #         da_reg_seas = da_reg.groupby(time_dim+'.year').mean(time_dim).sel(year=slice(*period))
    #         da = da.groupby(time_dim+'.year').mean(time_dim).sel(year=slice(*period)).chunk({'year':-1,'lat':-1,'lon':-1})

    # subselect to specified normalization period:
    if normal_period is not None:
        da_reg_norm = da_reg_seas.sel(year=slice(*normal_period))
    else:
        da_reg_norm = da_reg_seas.copy()
    
    #-------EOF-------#
    # fit the EOF model
    if (saved_model is not None) and (saved_model.exists()) and (not overwrite):
        print('loading existing EOF model')
        model = xe.single.EOF.load(saved_model)
    else:
        model = xe.single.EOF(use_coslat=True,n_modes=10)
        # fit the model
        model.fit(da_reg_norm,dim=sample_dims) # 45 s (40 mem)

        if (saved_model is not None) and (overwrite or (not saved_model.exists())):
            print('saving EOF model')
            if saved_model.exists():
                rm_tree(saved_model)
            model.save(saved_model)

    # EOF pattern in units of the data:
    model_components = model.components(normalized=False)

    full_index = model.transform(da_reg_seas,normalized=True)

    # get full dataset correlation pattern:
    if global_corr:
        print('computing global correlation')
        corr_full = xr.corr(full_index.sel(mode=eof_mode),da,dim=sample_dims).compute()
        return full_index.sel(mode=eof_mode).compute(), model_components.sel(mode=eof_mode), model.explained_variance_ratio().sel(mode=eof_mode), model, corr_full
    else:
        return full_index.sel(mode=eof_mode), model_components.sel(mode=eof_mode), model.explained_variance_ratio().sel(mode=eof_mode), model


# plot the pattern for testing:
# p = model_components.sel(mode=eof_mode).plot(transform=ccrs.PlateCarree(),subplot_kws={'projection':ccrs.LambertConformal(central_longitude=-20, central_latitude=45)}); p.axes.coastlines()

# function to regularize the grid to go from -180 to 180 and latitudes in ascending order
def regularize_grid(ds,input_lon_range=[0,360]):

    ds_regular = ds.copy()

    # find out how coordinates are organized:
    ds_coords = find_coordnames(ds)

    if (ds_regular[ds_coords['lon_dim']].values[0] == 0) or (input_lon_range == [0,360]):
        ds_regular.coords[ds_coords['lon_dim']] = (ds_regular.coords[ds_coords['lon_dim']] + 180) % 360 - 180
        ds_regular = ds_regular.sortby(ds_regular[ds_coords['lon_dim']])
    else:
        print('going from [-180,180] to [0,360] not implemented yet')
        pass
        
    # flip latitudes if necessary:
    if ds_regular[ds_coords['lat_dim']].values[0] > ds_regular[ds_coords['lat_dim']].values[-1]:
        ds_regular = ds_regular.isel({ds_coords['lat_dim']:slice(None,None,-1)})

    return ds_regular


if __name__=='__main__':
        
    pplt.rc['cartopy.autoextent'] = True

    sty = 1850
    eny = 2015

    dataset = '20CR'

    if dataset == '20CR':
        ds_ra = xr.open_dataset('/projects/NS9873K/owul/projects/climate_scenarios_2050/data/20CRV3/prmsl.mon.mean.nc')

        # Aggregate in time and choose variable:
        ds_ra_ann = ds_ra.prmsl.loc[(ds_ra.time.dt.year >= sty)&(ds_ra.time.dt.year <= eny)].groupby('time.year').mean()/100 # 
        # ds_ra_ann = ds_ra_ann.rolling(year=8).mean().dropna('year')
        # ds_ra_ann = ds_ra.prmsl.loc[(ds_ra.time.dt.month == 1) | (ds_ra.time.dt.month == 2)].groupby('time.year').mean()/100 # 

        # Subset to analysis domain:
        da_eof = ds_ra_ann.sel(lat=slice(20,89))

    elif dataset == 'HadSLP':
        ds_ra = xr.open_dataset('/projects/NS9873K/owul/projects/climate_scenarios_2050/data/HadSLP/slp.mnmean.real.nc')

        # Aggregate in time and choose variable:
        ds_ra_ann = ds_ra.slp.loc[(ds_ra.time.dt.year >= sty)&(ds_ra.time.dt.year <= eny)].groupby('time.year').mean()/100 # 
        # ds_ra_ann = ds_ra_ann.rolling(year=8).mean().dropna('year')
        # ds_ra_ann = ds_ra.prmsl.loc[(ds_ra.time.dt.month == 1) | (ds_ra.time.dt.month == 2)].groupby('time.year').mean()/100 # 

        # Subset to analysis domain:
        da_eof = ds_ra_ann.sel(lat=slice(89,20))
        new_lon = np.concatenate([da_eof.lon.values,[360]])
        da_eof = xr.concat([da_eof, da_eof.sel(lon=0)], dim="lon")
        da_eof = da_eof.assign_coords(lon=new_lon)

    test_detrend = xr.apply_ufunc(signal.detrend,da_eof,kwargs=dict(axis=0),
                                input_core_dims=[["year"]],output_core_dims=[["year"]],
                                output_dtypes=[da_eof.dtype],
                                dask='parallelized',vectorize=True)
    # test_detrend = test_detrend + da_eof.mean(dim='year')

    eof_model = xe.single.EOF(use_coslat=True,n_modes=10)

    eof_model.fit(da_eof,dim='year')

    # EOF pattern in units of the data:
    model_components = eof_model.components(normalized=False)
    exp_var = eof_model.explained_variance_ratio()
    pc = eof_model.transform(da_eof,normalized=True)

    mode=1
    plt_field = model_components.sel(mode=mode)
    evar = np.round(exp_var.sel(mode=mode).values*100)

    fig = pplt.figure(refwidth=3)
    axs = fig.subplots(nrows=1, proj='npstere',proj_kw={'central_longitude':0,})
    axs.format(
        suptitle='{3:} NH EOF{4:} ({0:} - {1:}); {2:}%'.format(sty,eny,evar,dataset,mode),
        coast=True, latlines=30, lonlines=60,
        leftlabels=(f''),
        leftlabelweight='normal',
    )
    # kw = {'levels': pplt.arange(-1, 1, .2)}
    pc0 = axs[0].contourf(plt_field.lon,plt_field.lat,plt_field,cmap='Div',cmap_kw={'cut':0})#,**kw)
    fig.colorbar(pc0, loc='r', span=1, label=f'[1]', extendsize='1.7em')

    pc.sel(mode=mode).plot()