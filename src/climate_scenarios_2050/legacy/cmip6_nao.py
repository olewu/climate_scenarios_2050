from climate_scenarios_2050.interpolate import interpolate_grid
from climate_scenarios_2050.compute_NAO import NAO_eof_based, cut_dataarray
from climate_scenarios_2050.config import *
from climate_scenarios_2050.spectral_analysis import multitaper
from climate_scenarios_2050.plot_tools import plot_loadings_and_spectrum
from climate_scenarios_2050.utils import ext_from_months, rm_tree, preprocess_spatial_coords, detrend_preprocessor
from climate_scenarios_2050.filters import butter_bandpass_filter, butter_lowpass_filter, lowfreq_variance_ratio
import xarray as xr
from scipy.signal import detrend
import xeofs as xe
import pandas as pd
import numpy as np
import re
from datetime import date
from functools import partial


def process_monthly_field(
        files_to_process,
        variable,
        interpolate=False,
        interpolate_options={},
        engine='netcdf4',
        concat_dim='member',
        member_coord=None,
        start_sel=None,
        end_sel=None,
        domain={'longitude':slice(None),'latitude':slice(None)},
        months=[],
        time_dim="time",
        center=[], # put in a centering/normalization period [from, to]
        return_global=True,
        unit_conversion_factor=1,
        preprocess=None,
    ):
    """
    
    """ 

    # Interpolate if requested:
    if interpolate:
        if not interpolate_options or 'grid_spacing' not in interpolate_options:
            raise ValueError('interpolate_options must contain a grid_spacing key')
        else:
            if 'method' not in interpolate_options:
                # set bilinear as default method:
                interpolate_options['method'] = 'bilinear'
            
            field = interpolate_grid(
                files_to_process,
                interpolate_options['grid_spacing'],
                variable,
                preprocess=preprocess,
                engine=engine,
                method=interpolate_options['method'],
                concat_dim=concat_dim,
            )

    else:
        ds = xr.open_mfdataset(files_to_process,concat_dim=concat_dim,preprocess=preprocess,combine='nested',engine=engine,parallel=False)
        ds = ds.chunk({concat_dim: 20,'time': 100,'lat': -1,'lon':-1})
        if 'lat' not in ds.coords:
            ds = ds.cf.rename({"Y": "lat"})
        if 'lon' not in ds.coords:
            ds = ds.cf.rename({"X": "lon"})

        field = ds[variable]

    if (concat_dim in field.dims) & (concat_dim not in field.coords):
        if member_coord is not None:
            field = field.assign_coords({concat_dim: member_coord})
        else:
            field = field.assign_coords({concat_dim:range(1,field.sizes[concat_dim]+1)})  

    # Cut the field to the desired domain and season (data will obtain a yearly time coordinate 'year'):
    processed_field,processed_field_global = cut_dataarray(
        field,
        period=[start_sel,end_sel],
        domain=domain,
        months=months,
        time_dim=time_dim,
        return_global=True,
    )

    if center:
        processed_field = processed_field - processed_field.sel(year=slice(*center)).mean(dim=('year',))
        if return_global:
            processed_field_global = processed_field_global - processed_field_global.sel(year=slice(*center)).mean(dim=('year',))
    if return_global:
        return processed_field * unit_conversion_factor, processed_field_global * unit_conversion_factor
    else:
        return processed_field * unit_conversion_factor

def get_EOF_model(
        field,
        detrend_field=True,
        sample_dims={'time':'year','member':'member'},
        saved_model=None,
        overwrite=False
    ):
    """
    """

    if detrend_field:
        chunks = {sample_dims['time']:-1,'lat':20,'lon':20}
        if len(sample_dims) > 1:
            chunks[sample_dims['member']] = 20
            field = field.chunk(chunks)

        field = xr.apply_ufunc(detrend,field,
            input_core_dims=[[sample_dims['time']]],output_core_dims=[[sample_dims['time']]],
            dask='allowed',vectorize=True,
        )

    if (saved_model is not None) and (not isinstance(saved_model,Path)):
        saved_model = Path(saved_model)

    sample_dims_list = tuple([val for _,val in sample_dims.items()])
    
    #-------EOF-------#
    # fit the EOF model
    if (saved_model is not None) and (saved_model.exists()) and (not overwrite):
        print('loading existing EOF model')
        model = xe.single.EOF.load(saved_model)
    else:
        model = xe.single.EOF(use_coslat=True,n_modes=10)
        # fit the model
        model.fit(field,dim=sample_dims_list) # 45 s (40 mem)

        if (saved_model is not None) and (overwrite or (not saved_model.exists())):
            print('saving EOF model')
            if saved_model.exists():
                rm_tree(saved_model)
            model.save(saved_model)

    return model

def prep_ERA5_mon(ds):
    if "valid_time" in ds.dims:
        ds_cp = ds.copy()
        ds_cp = ds_cp.rename({"valid_time":"time"})
        return ds_cp
    else:
        return ds

if __name__ == '__main__':

    make_plots = True

    recompute_EOFs = True

    fpath = base_path/'www/scenarios_2050/figures/model_NAO/'
    fpath.mkdir(exist_ok=True)

    # define target grid
    gsp = 1.5
    grid_spacing = {'lat':gsp,'lon':gsp}
    
    variable = 'psl'
    
    NW = 3  # Time-bandwidth product (higher = better spectral concentration)
    K = 2 * NW - 1  # Number of tapers (Slepian sequences)

    # define cut-offs for bandpass filter:
    long_per = 12
    low_freq = 1/long_per
    short_per = 7
    high_freq = 1/short_per

    splt_per = 20
    splt_freq = 1/splt_per

    mode = 1
    NAO_months = [12,1,2,3]
    mnth_ext = ext_from_months(NAO_months)

    start_sel = date(1850,1,1)
    end_sel = date(2025,12,31)
    
    # define normal period over which to compute EOFs:
    stnrm = 1950; ednrm = 2015
    normal_period = [stnrm,ednrm]

    proj_start_sel = date(2015,1,1)
    proj_end_sel = date(2099,12,31)

    # T_ax_expected = pd.period_range(start_sel,end_sel,freq='M')
    sty = start_sel.year
    eny = end_sel.year
    period = [sty,eny]
    # partially evaluate preprocessor:
    # prepro_partial = partial(detrend_preprocessor,T_ax_expected=T_ax_expected,start_year=sty,end_year=eny)
    domain = NAO_domain['winter']
    search_dir = proj_base/f'data/CMIP6/historical/'
    model_dir_list = sorted(list(search_dir.glob('*')))


    ref_obs = 'HadSLP'
    
    #---------------------------------------#
    # Derive EOF patterns from observations
    #---------------------------------------#
    
    if ref_obs == 'HadSLP':
        obs_file = proj_base/'data/HadSLP/slp.mnmean.real.nc'
        obs_var = 'slp'
        unit_conversion_factor = 1 # is in hPa
        ref_ext = '_HadSLP'
        concat_dim = "member" # need to set this to "member"
        EOF_model_path = obs_file.parent
        preprocess = None
    elif ref_obs == '20CRV3':
        obs_file = proj_base/'data/20CRV3/prmsl.mon.mean.nc'
        obs_var = 'prmsl'
        unit_conversion_factor = 1/100 # Pa to hPa
        ref_ext = '_20CRV3'
        concat_dim = "member" # need to set this to "member"
        preprocess = None
        EOF_model_path = obs_file.parent
    elif ref_obs == 'ERA5':
        obs_file = sorted(list(Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/mean_sea_level_pressure/").glob("*.nc")))
        obs_var = 'msl'
        unit_conversion_factor = 1/100 # Pa to hPa
        ref_ext = '_ERA5'
        preprocess = prep_ERA5_mon
        concat_dim = "time"
        EOF_model_path = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/mslp_EOF")
    
    processed_obs_field, processed_obs_field_global = process_monthly_field(
        files_to_process=obs_file,
        variable=obs_var,
        interpolate=True,
        interpolate_options={'grid_spacing':grid_spacing,'method':'bilinear'},
        concat_dim=concat_dim,
        unit_conversion_factor=unit_conversion_factor,
        start_sel=sty,
        end_sel=eny,
        months=NAO_months,
        domain=domain,
        time_dim="time",
        center=normal_period,
        preprocess=preprocess,
    )

    if "member" in processed_obs_field.dims:
        processed_obs_field = processed_obs_field.squeeze().drop('member')
        processed_obs_field_global = processed_obs_field_global.squeeze().drop('member')

    if "number" in processed_obs_field.coords:
        processed_obs_field = processed_obs_field.drop('number')
        processed_obs_field_global = processed_obs_field_global.squeeze().drop('number')

    if "expver" in processed_obs_field.coords:
        processed_obs_field = processed_obs_field.drop('expver')
        processed_obs_field_global = processed_obs_field_global.squeeze().drop('expver')


    # subselect to specified normalization period:
    if normal_period is not None:
        processed_obs_field_norm = processed_obs_field.sel(year=slice(*normal_period))
    else:
        processed_obs_field_norm = processed_obs_field

    EOF_model_obs = get_EOF_model(processed_obs_field_norm,detrend_field=True,sample_dims={'time':'year'},saved_model=EOF_model_path/f'{ref_obs}_{variable}_EOF{mnth_ext}_{stnrm}-{ednrm}_{gsp}deg.zarr',overwrite=recompute_EOFs)

    # flip the (arbitrary) sign to match the expected NAO pattern (positive NAO = low pressure over Iceland and high pressure over Azores):
    if EOF_model_obs.components(normalized=False).sel(mode=1,lat=35.25,lon=-20.25) < 0:
        obs_pattern_invert_factor = -1
    else:
        obs_pattern_invert_factor = 1

    full_index = obs_pattern_invert_factor * EOF_model_obs.transform(processed_obs_field,normalized=False).compute()
    # center and standardize the index to the norm period
    PC_obs_mu = EOF_model_obs.scores().mean(dim="year")
    PC_obs_sigma = EOF_model_obs.scores().std(dim="year")
    full_index = ((full_index - PC_obs_mu)/PC_obs_sigma).compute()
    ix_first = full_index.year.min().values
    ix_last = full_index.year.max().values
    full_index.to_netcdf(EOF_model_path/f'{ref_obs}_{variable}_EOF_full_index{mnth_ext}_{ix_first}-{ix_last}_{gsp}deg.nc')

    # global correlation pattern:
    processed_obs_field_global_detrended = xr.apply_ufunc(
        detrend,processed_obs_field_global,
        input_core_dims=[['year']],output_core_dims=[['year']],
        dask='allowed',vectorize=True,
    )
    global_corr = xr.corr(full_index.sel(mode=1),processed_obs_field_global_detrended,dim='year').compute()
    global_regr = global_corr * processed_obs_field_global_detrended.std(dim="year")

    # filter:
    index_filt = xr.apply_ufunc(butter_bandpass_filter,full_index,low_freq,high_freq,1)
    decadal_variance_ratio = index_filt.sel(mode=1).var('year')/full_index.sel(mode=1).var('year')
    freqs, psd_mt = multitaper(full_index.sel(mode=1),NW,K)
    title_ext = '({}%)'.format(np.round(EOF_model_obs.explained_variance_ratio().sel(mode=1).values*100))
    if make_plots:
        fig = plot_loadings_and_spectrum(full_index.sel(mode=1),global_corr,[ix_first,ix_last],norm_period=normal_period,freqs=freqs,psd=psd_mt,K=K,roll=8,model_name=ref_obs,extra_title=title_ext,nmode=mode,domain=domain,print_dec_var=decadal_variance_ratio.mean(),ylims=[.05,5],xlims=[0.01,0.5])
        fname = f'{ref_obs}_{variable}_EOF{mode}{mnth_ext}_{ix_first}-{ix_last}_{gsp}deg_{K}taper.png'
        fig.savefig(fpath/fname,dpi=300,bbox_inches='tight')

    var_ratio_obs = xr.apply_ufunc(
        lowfreq_variance_ratio,full_index,splt_freq,
        input_core_dims=(['year'],[]),output_core_dims=[[]],output_dtypes=[float],
        vectorize=True
    )

    var_ratio_obs.to_netcdf(EOF_model_path/f'varsplit_{splt_per}y_{ref_obs}_{variable}_EOF_full_index{mnth_ext}_{ix_first}-{ix_last}_{gsp}deg.nc')

    #------------------------------------------------------------------#
    # Derive model (historical) EOF patterns and project onto observed
    #------------------------------------------------------------------#
    failed_models_hist = []
    failed_models_scen = []
    for mod_dir in model_dir_list:
        all_sims = sorted(list(mod_dir.glob(f'{variable}/*.nc')))
        if all_sims:
            model = all_sims[0].stem.split('_')[2]
            fname = f'{model}_{variable}_EOF{mode}{mnth_ext}_{sty}-2014_{gsp}deg_{K}taper.png'
            # if (fpath/fname).exists():
            #     print(f'File {fname} already exists, skipping')
            #     continue
            print(model,len(all_sims))

            # extract model run indicator:
            member_id = [re.search(r'r\d{1,3}i\d{1,2}p\d{1,2}f\d{1,2}',sim.stem).group() for sim in all_sims]

            try:
                # pre-process the model data (interpolate, cut to domain, aggregate to desired seasonal aggragation):
                processed_model_field_hist, processed_model_field_global_hist = process_monthly_field(
                    files_to_process=all_sims,
                    variable=variable,
                    interpolate=True,
                    interpolate_options={'grid_spacing':grid_spacing,'method':'bilinear'},
                    unit_conversion_factor=1/100,
                    start_sel=sty,
                    end_sel=eny,
                    months=NAO_months,
                    domain=domain,
                    concat_dim='member',
                    member_coord=member_id,
                    time_dim="time",
                    center=False, # don't normalize here but afterwards, since we need access to the historical mean over normal_period
                    preprocess=preprocess_spatial_coords,
                )

                # subselect data over the normalization period:
                if normal_period is not None:
                    processed_model_field_hist_norm = processed_model_field_hist.sel(year=slice(*normal_period))
                else:
                    processed_model_field_hist_norm = processed_model_field_hist

                # find all members that have no-Nan data over the normal period
                # this is particularly important for EC-Earth where many runs only start in 1970
                valid_members = (
                    processed_model_field_hist_norm.notnull()
                    .all(dim=("year", "lat", "lon"))
                )

                processed_model_field_hist_norm = processed_model_field_hist_norm.sel(member=valid_members)

                # compute historical normal mean:
                hist_center = processed_model_field_hist_norm.mean(dim=(['year','member']),skipna=True)
                # full field --> anomalies wrt to normal period mean:
                processed_model_field_hist = (processed_model_field_hist - hist_center).compute()

                # E3SM-1-0  member 'r1i2p2f1' has two missing years (2014, 2015)
                if model == 'E3SM-1-0':
                    # processed_model_field_hist = processed_model_field_hist.sel(member=~processed_model_field_hist.member.isin(["r1i2p2f1"]))
                    member_mean = processed_model_field_hist.mean("member",skipna=True)
                    processed_model_field_hist = processed_model_field_hist.fillna(member_mean)
                    
                    # processed_model_field_global_hist = processed_model_field_global_hist.sel(member=~processed_model_field_global_hist.member.isin(["r1i2p2f1"]))
                    member_mean_gl = processed_model_field_global_hist.mean("member",skipna=True)
                    processed_model_field_global_hist = processed_model_field_global_hist.fillna(member_mean_gl)
                    

                processed_model_field_global_hist = processed_model_field_global_hist - processed_model_field_global_hist.sel(year=slice(*normal_period)).mean(dim=(['year','member']))

                # derive model's own EOF pattern:
                EOF_model_name = Path(str(mod_dir).replace('CMIP6','CMIP6_processed'))/'EOF_model'/f'{model}_{variable}_EOF{mnth_ext}_{stnrm}-{ednrm}_{gsp}deg.zarr'
                EOF_model_name.parent.mkdir(exist_ok=True)
                try:
                    EOF_model_sim = get_EOF_model(processed_model_field_hist_norm,detrend_field=True,saved_model=EOF_model_name,overwrite=recompute_EOFs)
                except:
                    EOF_model_sim = get_EOF_model(processed_model_field_hist_norm.sel(year=slice(1970,normal_period[-1])),detrend_field=True,saved_model=EOF_model_name,overwrite=recompute_EOFs)

                # flip the (arbitrary) sign to match the expected NAO pattern (positive NAO = low pressure over Iceland and high pressure over Azores):
                if EOF_model_sim.components(normalized=False).sel(mode=1,lat=35.25,lon=-20.25) < 0:
                    mod_pattern_invert_factor = -1
                else:
                    mod_pattern_invert_factor = 1

                # full model index:
                try: # this will most likely fail when the number of ensemble members used for EOF model is different than full ensemble mmeber list (e.g. EC-Earth3)
                    # if not (EOF_model_name.parent/f'{model}_{variable}_EOF_index{mnth_ext}_{sty}-2014_{gsp}deg.nc').exists():
                    try:
                        full_index_model = mod_pattern_invert_factor * EOF_model_sim.transform(processed_model_field_hist,normalized=False)
                    except:
                        full_index_model = [mod_pattern_invert_factor * EOF_model_sim.transform(processed_model_field_hist.sel(member=mem).drop("member"),normalized=False) for mem in processed_model_field_hist.member]
                        full_index_model = xr.concat(full_index_model,dim='member').assign_coords(coords={'member':processed_model_field_hist.member})
                    # full_index_model1 = processed_model_field_hist.dot(EOF_model_sim.components(normalized=True),dims=['lat', 'lon'])
                    # center and standardize over reference period:
                    PC_mod_mu = EOF_model_sim.scores().mean(dim=["member","year"])
                    PC_mod_sigma = EOF_model_sim.scores().std(dim=["member","year"])
                    full_index_model = (full_index_model - PC_mod_mu)/PC_mod_sigma
                    full_index_model.to_netcdf(EOF_model_name.parent/f'{model}_{variable}_EOF_index{mnth_ext}_{sty}-2014_{gsp}deg.nc')
                
                    # global correlation pattern:
                    processed_model_field_global_hist_detrended = xr.apply_ufunc(
                        detrend,processed_model_field_global_hist,
                        input_core_dims=[['year']],output_core_dims=[['year']],
                        dask='allowed',vectorize=True,
                    )

                    global_corr = xr.corr(full_index_model.sel(mode=1),processed_model_field_global_hist_detrended,dim=('year','member')).drop('mode')
                    global_corr.to_netcdf(EOF_model_name.parent/f'{model}_{variable}_EOF1_global_corr{mnth_ext}_{sty}-2014_{gsp}deg.nc')

                except:
                    print(f'failed to project all {model} historical ensemble members onto EOF pattern')
                    pass
                

                # project the historical field onto the OBSERVED EOF pattern:
                full_hyb_index_sim = [obs_pattern_invert_factor * EOF_model_obs.transform(processed_model_field_hist.sel(member=mem).drop('member'),normalized=False) for mem in processed_model_field_hist.member]
                full_hyb_index_sim = xr.concat(full_hyb_index_sim,dim='member').assign_coords(coords={'member':processed_model_field_hist.member})
                # center and standardize over reference period:
                full_hyb_index_sim = (full_hyb_index_sim - PC_obs_mu)/PC_obs_sigma
                full_hyb_index_sim.to_netcdf(EOF_model_name.parent/f'{model}_{variable}_EOF_{ref_obs}_index{mnth_ext}_{sty}-2014_{gsp}deg.nc')

                # filter index time series:
                index = full_hyb_index_sim.sel(mode=1)
                try:
                    index_filt = xr.apply_ufunc(
                        butter_bandpass_filter,index,low_freq,high_freq,1,
                        vectorize=True, dask = 'allowed'
                    )
                except:
                    index_filt = xr.apply_ufunc(
                        butter_bandpass_filter,index,low_freq,high_freq,1,
                        vectorize=False, dask = 'allowed'
                    )
                
                decadal_variance_ratio = (index_filt.var('year')/index.var('year')).mean('member').compute()

                # spectral analysis:
                PSD = []
                for mem in index.member:
                    freqs, psd_mt = multitaper(index.sel(member=mem),NW,K)
                    PSD.append(psd_mt)
                PSD = np.array(PSD)

                               
                var_ratio_sim = xr.apply_ufunc(
                    lowfreq_variance_ratio,full_hyb_index_sim,splt_freq,
                    input_core_dims=(['year'],[]),output_core_dims=[[]],output_dtypes=[float],
                    vectorize=True, dask='parallelized',
                ).mean('member').compute()

                var_ratio_sim.to_netcdf(EOF_model_name.parent/f'varsplit_{splt_per}y_{model}_{variable}_EOF_{ref_obs}_index{mnth_ext}_{sty}-{eny}_{gsp}deg.nc')
                
                # plot:
                if make_plots == True:
                    title_ext = ' ({}%) {} mems'.format(np.round(EOF_model_sim.explained_variance_ratio().sel(mode=1).values*100),len(index.member))
                    fig = plot_loadings_and_spectrum(index,global_corr,period,norm_period=normal_period,freqs=freqs,psd=PSD,K=K,roll=8,model_name=model,extra_title=title_ext,domain=domain,nmode=mode,print_dec_var=decadal_variance_ratio.mean(),ylims=[.05,5],xlims=[0.01,0.5])
                    fig.savefig(fpath/fname,dpi=300,bbox_inches='tight')

            except Exception as e:
                print('Processing failed with {:}'.format(e))
                failed_models_hist.append(model)
                continue
        
            # load the corresponding projections and project the centered fields onto the EOF pattern:
            for scenario in ['ssp119','ssp126','ssp585']: #['ssp245','ssp370']:
                try:
                    ssp_path = proj_base/f'data/CMIP6/{scenario}/{model}/{variable}'
                    all_proj = sorted(list(ssp_path.glob('*2015*.nc')))
                    # could also skip the following and just compute the index for all available simulations:
                    # sort out members that were not considerer/available for historical period
                    # all_proj = [proj for proj in all_proj if re.search(r'r\d{1,3}i\d{1,2}p\d{1,2}f\d{1,2}',proj.stem).group() in member_id]

                    if not all_proj:
                        print(f'no {scenario} projections available for {model}')
                        continue

                    member_id_proj = [re.search(r'r\d{1,3}i\d{1,2}p\d{1,2}f\d{1,2}',proj.stem).group() for proj in all_proj]

                    processed_model_field_proj = process_monthly_field(
                        files_to_process=all_proj,
                        variable=variable,
                        interpolate=True,
                        interpolate_options={'grid_spacing':grid_spacing,'method':'bilinear'},
                        unit_conversion_factor=1/100,
                        start_sel=proj_start_sel.year,
                        end_sel=proj_end_sel.year,
                        months=NAO_months,
                        domain=domain,
                        concat_dim='member',
                        member_coord=member_id_proj,
                        time_dim="time",
                        return_global=False,
                        preprocess=preprocess_spatial_coords,
                    )

                    processed_model_field_proj_anom = (processed_model_field_proj - hist_center).compute()

                    save_proj_path = Path(str(EOF_model_name.parent).replace('historical',scenario))
                    save_proj_path.mkdir(exist_ok=True, parents=True)
                    
                    # project the normalized slp time series onto the NAO pattern derived from historical run:
                    # if not (save_proj_path/f'{model}_{variable}_EOF_full_index{mnth_ext}_{sty}-2014_{gsp}deg.nc').exists():
                    try:
                        full_index_proj = mod_pattern_invert_factor * EOF_model_sim.transform(processed_model_field_proj_anom,normalized=False).compute()
                        full_index_proj = (full_index_proj - PC_mod_mu)/PC_mod_sigma
                        full_index_proj.to_netcdf(save_proj_path/f'{model}_{variable}_EOF_full_index{mnth_ext}_{sty}-2014_{gsp}deg.nc')
                    except:
                        print(f' {scenario} failed to project onto {model} EOF pattern')
                        pass

                    # project the normalized slp time series onto the NAO pattern derived from reference/observations:
                    full_hyb_index_proj = [obs_pattern_invert_factor * EOF_model_obs.transform(processed_model_field_proj_anom.sel(member=mem).drop('member'),normalized=False) for mem in processed_model_field_proj.member]
                    full_hyb_index_proj = xr.concat(full_hyb_index_proj,dim='member').assign_coords(coords={'member':processed_model_field_proj.member})
                    full_hyb_index_proj = (full_hyb_index_proj - PC_obs_mu)/PC_obs_sigma
                    full_hyb_index_proj.to_netcdf(save_proj_path/f'{model}_{variable}_EOF_{ref_obs}_full_index{mnth_ext}_{sty}-{eny}_{gsp}deg.nc')

                except Exception as e:
                    print('Projection of future ({:}) field failed with {:}'.format(scenario,e))
                    failed_models_scen.append(f'{scenario}: {model}')
                    continue
            
    print('failed historical:')
    print(failed_models_hist)

    print('failed projections:')
    print(failed_models_scen)


# import shutil
# for mo in np.arange(1,4):
#     MO = str(mo).zfill(2)
#     FILE = Path(f"/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/mean_sea_level_pressure/mean_sea_level_pressure_2026_{MO}.nc")
#     FILE_tmp = Path(f"/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/mean_sea_level_pressure/mean_sea_level_pressure_2026_{MO}_.nc")
#     with xr.open_dataset(FILE) as ds:
#         ds = ds.drop(["expver","number"])
#     ds.to_netcdf(FILE_tmp)

#     FILE.unlink()
#     shutil.copyfile(FILE_tmp, FILE)
#     FILE_tmp.unlink()