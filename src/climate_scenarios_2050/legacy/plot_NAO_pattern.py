from climate_scenarios_2050.compute_NAO import NAO_eof_based, load_verification
from climate_scenarios_2050.config import NAO_domain, var_name_map, AO_domain
from climate_scenarios_2050.utils import make_season_group
import proplot as pplt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import xarray as xr
import numpy as np


pplt.rc['cartopy.autoextent'] = True

period = [1960,2023]
vf_type = 'ERA5'
normal_period = [1990,2021] # normal reference period for defining NAO
NAO_var = 'mean_sea_level_pressure'

NAO_index, NAO_pattern, NAO_exvar = NAO_eof_based(
    vf_type,
    period,
    normal_period=normal_period,
    var=NAO_var,
    months=[1,2,3,4,5,6,7,8,9,10,11,12],
    domain=NAO_domain['winter'],
    eof_mode=1,
)

NAO_filt = NAO_index.rolling(year=8).mean().dropna('year')

# Compute NAO terciles:
NAO_terciles_raw = NAO_index.quantile([1/3,2/3])

gs = pplt.GridSpec(ncols=1, nrows=3)

fig = pplt.figure(refwidth=4)
ax0 = fig.subplot(gs[0:2,0], proj='npstere',proj_kw={'central_longitude':-20, }) # 'lcc','central_latitude':45,
pc0 = ax0.contourf(NAO_pattern.longitude,NAO_pattern.latitude,NAO_pattern)
ax0.format(
    title='NAO pattern ({0} - {1})'.format(normal_period[0],normal_period[-1]), coast=True, latlines=30, lonlines=60,
)
ax0.colorbar(pc0, loc='r', label=f'hPa', extendsize='1.7em')

ax1 = fig.subplot(gs[2,0])
NAO_index.plot(ax=ax1,label='raw',color='k')
NAO_filt.plot(ax=ax1,label='8-yr rolling mean',color='C1')
ax1.format(
    title=f'NAO index', ylabel='standardized anomalies [1]'
)
ax1.legend()
col = PatchCollection([
    Rectangle((y-.5, ax1.get_ylim()[0]), 1, 1,alpha=.1)
    for y in range(period[0], period[-1] + 1)
])

cmap = ListedColormap([
    'blue', 'white','red'
])
# set data, colormap and color limits
col.set_array(np.digitize(NAO_index,[*NAO_terciles_raw.values]))
col.set_cmap(cmap)
col.set_clim(0,2)
col.set_alpha(.15)
ax1.add_collection(col)
# ax1.axhspan(NAO_terciles_raw.isel(quantile=0).values, ax1.get_ylim()[0], color='blue', alpha=0.1)
# ax1.axhspan(NAO_terciles_raw.isel(quantile=0).values, NAO_terciles_raw.isel(quantile=1).values, color='grey', alpha=0.1)
# ax1.axhspan(ax1.get_ylim()[-1], NAO_terciles_raw.isel(quantile=1).values, color='red', alpha=0.1)

fig.format(suptitle=f'{vf_type} NAO')
# fig.savefig('/projects/NS9873K/www/decadal/NAO/{0}_NAO_eof_{1}-{2}.png'.format(vf_type,*normal_period),dpi=300)


#------PROJECTIONS ONTO NAO INDEX-------#
# project other variables onto NAO index to see related variability
# also do this for the 8-year filtered time series

for key,var in var_name_map['long_to_cf'].items():
    full_field = load_verification(key,period,vf_type)
    season = make_season_group(full_field,months_in_season=[12,1,2,3])
    full_field = full_field.groupby(season).mean('time').sel(year=slice(period[0],period[-1]-1))[var].sel(latitude=slice(90,10))
    full_field_filt = full_field.compute().rolling(year=8).mean().dropna('year')

    full_field_corr = xr.corr(NAO_index,full_field,dim='year').compute()
    full_field_corr_filt = xr.corr(NAO_filt,full_field_filt,dim='year').compute()

    fig = pplt.figure(refwidth=3)
    axs = fig.subplots(nrows=1, proj='npstere',proj_kw={'central_longitude':0,})
    axs.format(
        suptitle=f'NAO-{var} correlation',
        coast=True, latlines=30, lonlines=60,
        leftlabels=(f''),
        leftlabelweight='normal',
    )
    kw = {'levels': pplt.arange(-1, 1, .2)}
    pc0 = axs[0].contourf(full_field_corr.longitude,full_field_corr.latitude,full_field_corr,cmap='Div',cmap_kw={'cut':-.05},**kw)
    fig.colorbar(pc0, loc='r', span=1, label=f'[1]', extendsize='1.7em')

    fig.savefig('/projects/NS9873K/www/decadal/NAO/{0}_NAO_eof_{1}-{2}_correlation_{3}.png'.format(vf_type,*normal_period,key),dpi=300)


    fig = pplt.figure(refwidth=3)
    axs = fig.subplots(nrows=1, proj='npstere',proj_kw={'central_longitude':0,})
    axs.format(
        suptitle=f'NAO-{var} correlation',
        coast=True, latlines=30, lonlines=60,
        leftlabels=(f''),
        leftlabelweight='normal',
    )
    pc0 = axs[0].contourf(full_field_corr_filt.longitude,full_field_corr_filt.latitude,full_field_corr_filt,cmap='Div',cmap_kw={'cut':-.05},**kw)
    fig.colorbar(pc0, loc='r', span=1, label=f'[1]', extendsize='1.7em')

    fig.savefig('/projects/NS9873K/www/decadal/NAO/{0}_NAO_8yr_eof_{1}-{2}_correlation_{3}.png'.format(vf_type,*normal_period,key),dpi=300)
    