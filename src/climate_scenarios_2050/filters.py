from scipy import signal
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfiltfilt(sos, data)
        return y

def butter_lowpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

def butter_lowpass_filter(data, cutoff, fs, order=4):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


def lowfreq_variance_ratio(ts,cutoff,fs=1):
    """
    Compute the ratio of low-frequency variability (f < cutoff) 
    to the total variance of a time series ts
    """
    f, Pxx = signal.welch(ts, fs=fs)

    # Total power (variance)
    total_var = np.trapz(Pxx, f)

    # Low-frequency variance
    low_freq_mask = f <= cutoff
    low_var = np.trapz(Pxx[low_freq_mask], f[low_freq_mask])
    
    # Low-frequency variance ratio:
    low_ratio = low_var / total_var

    return low_ratio