import proplot as pplt
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy import signal
from climate_scenarios_2050.utils import load_climate_index

def multitaper(time_series,NW,K):

    tapers = signal.windows.dpss(len(time_series), NW, K)  # Compute DPSS tapers
    
    # Compute multitaper power spectral density
    psd_mt = np.zeros(len(time_series)//2)
    freqs = fft.fftfreq(len(time_series), 1)[:len(time_series)//2]  # Positive frequencies

    for taper in tapers:
        x_tapered = time_series * taper  # Apply taper to the signal
        X_tapered = fft.fft(x_tapered)[:len(time_series)//2]  # Compute FFT and keep positive freqs
        psd_mt += np.abs(X_tapered) ** 2  # Sum power spectra

    psd_mt /= K  # Average over tapers

    return freqs, psd_mt

if __name__ == '__main__':

    #--------------Spectral analysis of observed climate index-------------#
    INDEX_NAME = 'NAO'
    agg = 'monthly'
    itype = 'station'
    start_year = 1850
    end_year = 2015

    # load the index time series:
    INDEX,time_dim,idx_name = load_climate_index(INDEX_NAME,agg,itype,filter=False,start=start_year,end=end_year,filter_type='10yrLP')

    # generate the title based on the input
    if INDEX_NAME == 'AMO':
        title = f'HadISST1 AMO (Trenberth) {agg}'
    elif INDEX_NAME == 'NAO':
        title = f'NAO (Hurrell) {agg}'

    # process (remove mean, detrend)

    # extract time vector and data vector as numpy:
    t = INDEX[time_dim].to_numpy()

    time_series_raw = INDEX.index_ts.to_numpy()
    # normalize:
    time_series = time_series_raw - time_series_raw.mean()
    print(t[-1])

    # 
    fig = pplt.figure(refwidth=4,refheight=3)
    ax = fig.add_subplot()
    ax.plot(t,time_series)
    ax.set_title(INDEX_NAME)
    N = 8
    ax.plot(t[N//2:-N//2+1],np.convolve(time_series,[1/N]*N,mode='valid'))


    sp = np.abs(fft.fftshift(fft.fft(time_series)))
    freq = fft.fftshift(fft.fftfreq(t.shape[0]))

    fig = pplt.figure()
    ax = fig.add_subplot()
    ax.loglog(freq[len(freq)//2+1:],sp[len(freq)//2+1:])

    f,Pxx = signal.periodogram(time_series,scaling='density')
    f,Pxx_flattop = signal.periodogram(time_series,window='flattop',scaling='density')
    f,Pxx_hann = signal.periodogram(time_series,window='bartlett',scaling='density')

    # Define multitaper parameters
    NW = 3  # Time-bandwidth product (higher = better spectral concentration)
    K = 2 * NW - 1  # Number of tapers (Slepian sequences)

    freqs, psd_mt = multitaper(time_series,NW,K)

    # linear fit to multitaper esitmate:
    m,b = np.polyfit(np.log(freqs[1:]),np.log(psd_mt[1:]),1)
    logline_spec = np.polyval([m,b],np.log(freqs[1:]))

    # Spectrum of AR(1) process with alpha1 estimated from NAO time series
    lag1_autocorr = np.corrcoef(time_series[1:],time_series[:-1])[0,1]
    time_series_var = time_series.var()
    AR1_spec = time_series_var/(1 + lag1_autocorr**2 - 2*lag1_autocorr*np.cos(2*np.pi*freqs))
    # simple empirical significance threshold for multitapers:
    threshold = AR1_spec * (1 + 2 * np.sqrt(2 / K))

    fig = pplt.figure(refwidth=5,refheight=2.5,suptitle='{:} spectral density ({:} - {:})'.format(title,t[0],t[-1]))
    ax = fig.add_subplot()
    # ax.loglog(f[1:-1],Pxx[1:-1],label='Boxcar taper',lw=.5)
    # ax.loglog(f[:-1],Pxx_flattop[:-1],label='Flattop taper',lw=.5)
    # ax.loglog(f[:-1],Pxx_hann[:-1],label='Bartlett taper',lw=.5)

    ax.format(
        ylabel='Spectral Density', xlabel='frequency (year$^{-1}$)',
        yscale='log',xscale='log'
    )
    ax.plot(freqs[1:],psd_mt[1:],label=f'Multitaper ({K})',color='C0')
    ax.plot(freqs[1:],np.exp(logline_spec),label=r'$s^{%.2f}$' % np.round(m,2),ls='dotted',color='C0')
    ax.plot(freqs[1:],AR1_spec[1:],label='AR(1) process',color='C1')
    ax.plot(freqs[1:],threshold[1:],label='Multitaper AR(1) Empirical Significance Threshold',ls='dashed',color='C1')
    markerline, stemlines, baseline = ax.stem(freq[len(freq)//2+1:],sp[len(freq)//2+1:].real,linefmt='grey',markerfmt='o')
    stemlines.set_linestyle('dashed')
    markerline.set_label('Fourier Spectrum')
    markerline.set_markerfacecolor('none')
    markerline.set_markeredgecolor('grey')
    markerline.set_markersize(5)

    # ax_min = min(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:])
    # ax_max = max(*psd_mt[1:],*np.exp(logline_spec),*AR1_spec[1:],*threshold[1:],*sp[len(freq)//2:].real)
    # ax.set_ylim([ax_min - ax_min/2,ax_max+ax_max*2])

    freq_ticks = np.array([.005,.01,.02,.05,.1,.2])  # Common log-scale ticks
    ax.set_xticks(freq_ticks)

    # Create a secondary x-axis on top
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax.get_xlim())  # Keep same limits

    # Compute period values (T = 1 / f)
    period_ticks = 1 / freq_ticks
    ax_top.set_xticks(freq_ticks)  # Use the same tick positions
    ax_top.set_xticklabels([f'{t:.1f}' for t in period_ticks])  # Format as period values
    ax_top.set_xlabel('Period (years)')

    # dx, dy = -15, 0
    # offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
    # for label in axd.xaxis.get_majorticklabels():
    #     label.set_transform(label.get_transform() - offset)

    fig.legend(loc='b',ncols=2)

    fig.savefig('/projects/NS9873K/www/decadal/NAO/{:}_{:}-{:}_spectral_density_multitaper_{:}.png'.format(idx_name,t[0],t[-1],K),dpi=300)


    #-------------Load Model NAO time series-----------#
    # Compute a spectrum for each ensemble member (same physics)
    # and average the resulting spectra for an estimate of the model's
    # ability to produce the peaks (if we want to call 
    # them that given the observations...)


