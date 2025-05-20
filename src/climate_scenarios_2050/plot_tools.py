from climate_scenarios_2050 import config

from pathlib import Path

from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import proplot as pplt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import FuncFormatter

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyproj import transform

def base10_notation(x,pos):
    if x == 0:
        return "0"
    exponent = np.log10(x)
    if exponent.is_integer():
        return f"$10^{{{int(exponent)}}}$"
    else:
        return f"${x:.2g}$"

def make_station_plot(station_dict=None,outpath=None,figfilename='map_stations'):
    
    if outpath is None:
        outpath = config.proj_base
    elif isinstance(outpath,str):
        outpath = Path(outpath)
    
    if station_dict is None:
        station_dict = config.stations
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EuroPP())
    ax.set_extent([-20, 20, 42, 70], crs=ccrs.PlateCarree())

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth.
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    # Add our states feature.
    ax.add_feature(states_provinces, edgecolor='gray')
    # Add land feature, overriding the default negative zorder so it shows
    # above the background image.
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')
    ax.gridlines(color='grey')
    
    lons,lats,names = [],[],[]
    for st,prop in station_dict.items():
        lons.append(prop['lon'])
        lats.append(prop['lat'])
        names.append(prop['name'])
    
    hora = ['right','center','left','left','left','right','left']
    vera = ['center','top','center','center','center','center','center']
    colors = plt.cm.tab10.colors[:len(lons)]
    offslo = [-2,0,1.7,1.5,1.7,-1.7,1.7]
    offsla = [0,-.8,0,0,0,0,0]
    ax.scatter(lons,lats,transform=ccrs.PlateCarree(),c=colors,zorder=100,s=13)
    for lo,la,na,ha,va,co,of,ofla in zip(lons,lats,names,hora,vera,colors,offslo,offsla):
        ax.annotate(na, (lo+of,la+ofla), transform=ccrs.PlateCarree(), bbox=dict(facecolor='grey', alpha=0.9, edgecolor=co), fontsize=10, ha=ha,va=va)

    # Add a text annotation for the license information to the
    # the bottom right corner.
    text = AnchoredText('\u00A9 {}; license: {}'''.format(SOURCE, LICENSE),loc='lower right', prop={'size': 7}, frameon=True)
    ax.add_artist(text)

    plt.savefig(outpath/f'{figfilename}.png',dpi=300,bbox_inches='tight')


def interpolate_box_edges(lons, lats, n_points=100):
    """Interpolate between corners to create smooth box edges."""
    coords = []
    for i in range(len(lons) - 1):
        lon_interp = np.linspace(lons[i], lons[i+1], n_points)
        lat_interp = np.linspace(lats[i], lats[i+1], n_points)
        coords.append(np.column_stack([lon_interp, lat_interp]))
    return np.vstack(coords)


def invert(x):
    # 1/x with special treatment of x == 0
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x

def plot_loadings_and_spectrum(index,pattern,period,model_name,freqs,psd,K,roll=8,extra_title='',nmode=1,units='1',variable_name='SLP',domain=None,print_dec_var=None,ylims=None,xlims=None):
    # index_filt = index.rolling(year=roll).mean().dropna('year')

    try:
        lag1_autocorr = np.corrcoef(index.values[:,1:].flatten(),index.values[:,:-1].flatten())[0,1]
    except:
        lag1_autocorr = np.corrcoef(index.values[1:],index.values[:-1])[0,1]
    time_series_var = index.values.var()
    AR1_spec = time_series_var/(1 + lag1_autocorr**2 - 2*lag1_autocorr*np.cos(2*np.pi*freqs))

    gs = pplt.GridSpec(ncols=2, nrows=3)

    fig = pplt.figure(refwidth=4)
    ax0 = fig.subplot(gs[0:2,0:2], proj='npstere',proj_kw={'central_longitude':-20, }) # 'lcc','central_latitude':45,
    pc0 = ax0.contourf(pattern.lon,pattern.lat,pattern,levels=np.arange(-1,1.01,.2))
    ax0.format(
        title='{3} EOF{2} pattern ({0} - {1})'.format(period[0],period[-1],nmode,variable_name), coast=True, latlines=30, lonlines=60,
    )
    ax0.colorbar(pc0, loc='r', label=f'{units}', extendsize='1.7em')
    if domain is not None:
        box_coords = np.array([
            [domain['longitude'].start, domain['latitude'].start],
            [domain['longitude'].start,  domain['latitude'].stop],
            [domain['longitude'].stop,  domain['latitude'].stop],
            [domain['longitude'].stop, domain['latitude'].start],
            [domain['longitude'].start, domain['latitude'].start]  # Close the box
        ])

        box_coords_smooth = interpolate_box_edges(box_coords[:, 0], box_coords[:, 1],n_points=100)

        ax0.plot(box_coords_smooth[:, 0], box_coords_smooth[:, 1], transform=ccrs.PlateCarree(), color='red', linewidth=2)
    # ax1 = fig.subplot(gs[2,0:2])
    # index.plot(ax=ax1,label='raw',color='k')
    # index_filt.plot(ax=ax1,label=f'{roll}-yr rolling mean',color='C1')
    # ax1.format(
    #     title=f'PC{nmode}', ylabel='standardized anomalies [1]'
    # )
    # ax1.legend()
    
    # plot spectrum:
    ax2 = fig.subplot(gs[2,0:2])
    
    if (len(psd.shape) > 1) & (psd.shape[0] > 1):
        psd_full = psd
        psd = psd.mean(0)
    else:
        psd = psd.squeeze()
        psd_full = None

    
    ax2.plot(freqs[1:],psd[1:],label=f'Multitaper ({K})',color='C0')
    # ax2.plot(freqs[1:],np.exp(logline_spec),label=r'$s^{%.2f}$' % np.round(m,2),ls='dotted',color='C0')
    ax2.plot(freqs[1:],AR1_spec[1:],label='AR(1) process',color='C1')
    
    if psd_full is not None:
        low,high = np.percentile(psd_full,[10,90],axis=0)
        ax2.fill_between(x=freqs,y1=low,y2=high,color='C0',alpha=.2)
    
    # ax2.plot(freqs[1:],threshold[1:],label='Multitaper AR(1) Empirical Significance Threshold',ls='dashed',color='C1')
    # markerline, stemlines, baseline = ax2.stem(freq[len(freq)//2+1:],sp[len(freq)//2+1:].real,linefmt='grey',markerfmt='o')
    # stemlines.set_linestyle('dashed')
    # markerline.set_label('Fourier Spectrum')
    # markerline.set_markerfacecolor('none')
    # markerline.set_markeredgecolor('grey')
    # markerline.set_markersize(5)

    # ax_min = min(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:])
    # ax_max = max(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:],*sp[len(freq)//2:].real)
    # ax2.set_ylim([ax_min - ax_min/2,ax_max+ax_max*2])
    freq_ticks = np.array([.01,.02,.05,.1,.2])  # Common log-scale ticks

    ax2.format(
        title='Multitaper Spectrum',
        ylabel='Spectral Density', xlabel='Frequency (year$^{-1}$)',
        yscale='log',xscale='log',
        xticks=freq_ticks
    )


    if xlims is not None:
        ax2.set_xlim(xlims)


    # ax_top = ax2.dualx(
    #     'inverse', locator='log', label='period (years)'
    # )
    ax_top = ax2.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax2.get_xlim())  # Keep same limits
    
    ax_top.set_xticks(freq_ticks)  # Use the same tick positions
    ax_top.set_xlabel('Period (years)')

    ax2.yaxis.set_major_formatter(FuncFormatter(base10_notation))

    if ylims is not None:
        ax2.set_ylim(ylims)

    # Create a secondary x-axis on top
    # ax_top = ax2.secondary_xaxis(location='top',functions=(invert,invert))
    
    # Compute period values (T = 1 / f)
    period_ticks = 1 / freq_ticks
    ax_top.set_xticklabels([f'{t:.1f}' for t in period_ticks])  # Format as period values
    
    yl,yu = ax2.get_ylim()
    ax2.fill_between(x=[1/12,1/7],y1=yu,y2=yl,color='lightgreen',alpha=.3)

    if print_dec_var is not None:
        ax2.annotate('{:2.1f}%'.format(print_dec_var*100), (1/7,.05), xycoords = ('data','axes fraction'), bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgreen'), fontsize=10, va='bottom')

    ax2.legend(loc='ll',ncols=2)

    fig.format(suptitle=f'{model_name} EOF{nmode}{extra_title}')

    return fig

# def plot_spectrum():
#     fig = pplt.figure(refwidth=5,refheight=2.5,suptitle='{:} spectral density ({:} - {:})'.format(title,t[0],t[-1]))
#     ax = fig.add_subplot()
#     # ax.loglog(f[1:-1],Pxx[1:-1],label='Boxcar taper',lw=.5)
#     # ax.loglog(f[:-1],Pxx_flattop[:-1],label='Flattop taper',lw=.5)
#     # ax.loglog(f[:-1],Pxx_hann[:-1],label='Bartlett taper',lw=.5)

#     ax.format(
#         ylabel='Spectral Density', xlabel='frequency (year$^{-1}$)',
#         yscale='log',xscale='log'
#     )
#     ax.plot(freqs[1:],psd_mt[1:],label=f'Multitaper ({K})',color='C0')
#     ax.plot(freqs[1:],np.exp(logline_spec),label=r'$s^{%.2f}$' % np.round(m,2),ls='dotted',color='C0')
#     ax.plot(freqs[1:],AR1_spec[1:],label='AR(1) process',color='C1')
#     ax.plot(freqs[1:],threshold[1:],label='Multitaper AR(1) Empirical Significance Threshold',ls='dashed',color='C1')
#     markerline, stemlines, baseline = ax.stem(freq[len(freq)//2+1:],sp[len(freq)//2+1:].real,linefmt='grey',markerfmt='o')
#     stemlines.set_linestyle('dashed')
#     markerline.set_label('Fourier Spectrum')
#     markerline.set_markerfacecolor('none')
#     markerline.set_markeredgecolor('grey')
#     markerline.set_markersize(5)

#     # ax_min = min(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:])
#     # ax_max = max(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:],*sp[len(freq)//2:].real)
#     # ax.set_ylim([ax_min - ax_min/2,ax_max+ax_max*2])

#     freq_ticks = np.array([.005,.01,.02,.05,.1,.2])  # Common log-scale ticks
#     ax.set_xticks(freq_ticks)

#     # Create a secondary x-axis on top
#     ax_top = ax.twiny()
#     ax_top.set_xscale('log')
#     ax_top.set_xlim(ax.get_xlim())  # Keep same limits

#     # Compute period values (T = 1 / f)
#     period_ticks = 1 / freq_ticks
#     ax_top.set_xticks(freq_ticks)  # Use the same tick positions
#     ax_top.set_xticklabels([f'{t:.1f}' for t in period_ticks])  # Format as period values
#     ax_top.set_xlabel('Period (years)')

#     # dx, dy = -15, 0
#     # offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
#     # for label in axd.xaxis.get_majorticklabels():
#     #     label.set_transform(label.get_transform() - offset)

#     fig.legend(loc='b',ncols=2)

#     fig.savefig('/projects/NS9873K/www/decadal/NAO/{:}_{:}-{:}_spectral_density_multitaper_{:}.png'.format(idx_name,t[0],t[-1],K),dpi=300)


if __name__ == '__main__':
    outpath = Path('/projects/NS9873K/www/scenarios_2050/figures')
    make_station_plot(outpath=outpath)