import pandas as pd
import xarray as xr
from climate_scenarios_2050.config import proj_base, base_path
from climate_scenarios_2050.cmip6_nao import butter_lowpass_filter
import proplot as pplt

fpath = base_path/'www/scenarios_2050/figures/NAO_wind_power'



nao_djfm = pd.read_csv('/projects/NS9873K/owul/projects/climate_scenarios_2050/data/climate_indices/NAO/NAO_djfm_Hurrell_station_based_1864-2023.csv')

nao_djfm = nao_djfm[nao_djfm.year > 1959].reset_index(drop=True)

nao_Nyr_run = nao_djfm.groupby('year').mean()['index_ts'].rolling(8,center=True).mean()

filt = 'rm'
highcut = 1/7

nao_hadslp = xr.open_dataset(proj_base/f'data/HadSLP/HadSLP_psl_EOF_full_index_DJFM_1850-2014_1.5deg.nc').sel(mode=1).squeeze().scores
if filt == 'lp':
    nao_hadslp_lp = xr.apply_ufunc(butter_lowpass_filter, nao_hadslp, highcut, 1)
elif filt == 'rm':
    nao_hadslp_lp = nao_hadslp.rolling(year=10, center=True).mean().dropna('year')

prod_files = sorted(list((proj_base/'data/power_production/hourly/').glob('*_ERA5_100m.csv')))

for file in prod_files:

    station = file.stem.split('_')[0]

    df = pd.read_csv(file,skiprows=3,delimiter=';')
    df.columns = df.columns.str.strip("'")

    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

    prod_ts_hourly = df.set_index('time')['Prod (MWh)'].to_xarray()
    ann_prod = prod_ts_hourly.groupby('time.year').sum()
    if filt == 'lp':
        ann_prod_lp = xr.apply_ufunc(butter_lowpass_filter, ann_prod, highcut, 1)
    elif filt == 'rm':
        ann_prod_lp = ann_prod.rolling(year=10,center=True).mean()

    # other low-pass filtering? [df.month.isin([12,1,2,3])]
    df_Nyr = df.groupby('year')['Prod (MWh)'].sum()/1000
    df_Nyr_run = df_Nyr.rolling(10,center=True).mean()


    fig = pplt.figure(refwidth=4)
    ax = fig.subplot(xlim=(1965,2024),
                    yformatter='sci',ylabel='Annual Production',
                    title=f'{station} annual production vs NAO')
    ax.plot(ann_prod.year,ann_prod,label='Production',color='C0',ls='dashed',alpha=.5,lw=.75)
    ax.plot(ann_prod_lp.year,ann_prod_lp,label='Production low-pass',color='C0',ls='solid',lw=2)
    ax.format()
    axs = ax.twinx(ylabel='NAO index')
    axs.plot(nao_hadslp.year,nao_hadslp,color='k',ls='dashed',label='NAO',alpha=.5,lw=.75)
    axs.plot(nao_hadslp_lp.year,nao_hadslp_lp,color='k',ls='solid',label='NAO low-pass',lw=2)

    ax.legend(loc=2)
    axs.legend(loc=3)

    fname = f'NAO_wind_power_{filt}_{station}.png'
    fig.savefig(fpath/fname)

    print(xr.corr(nao_hadslp,ann_prod))
    print(xr.corr(nao_hadslp_lp,ann_prod_lp))
    
    corr = pd.merge(df_Nyr_run,nao_Nyr_run,on='year').corr()

    print(station)
    print(corr)
    print('')
