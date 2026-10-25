from scipy import signal
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, nancut=False, order=4):
    nan_indices = np.where(np.isnan(data))[0]
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if nancut and (len(nan_indices) != 0):
        # Check if all NaNs are consecutive at the beginning or end
        if (np.all(np.diff(nan_indices) == 1) and (nan_indices[-1] + 1 == len(data) or nan_indices[0] == 0)):
            y = data.copy()
            # Remove NaNs, process, and write back
            non_nan_indices = ~np.isnan(data)
            processed = signal.sosfiltfilt(sos, data[non_nan_indices])
            y[non_nan_indices] = processed
            return y
        else:
            # Return all NaNs if isolated NaNs are present
            return np.full_like(data, np.nan)
    else:
        return signal.sosfiltfilt(sos, data)

def butter_lowpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

def butter_lowpass_filter(data, cutoff, fs, nancut=False, order=4, pad=None):
    nan_indices = np.where(np.isnan(data))[0]
    sos = butter_lowpass(cutoff, fs, order=order)
    if pad is not None:
        if isinstance(pad,int) & pad>0:
            data = np.r_[data[pad:0:-1], data, data[-2:-pad-2:-1]]
            low_in = pad
            upp_in = -pad
        else:
            print("pad has to be of type `int` (> 0) or None")
            return
    else:
        low_in = 0
        upp_in = None
    if nancut and (len(nan_indices) != 0):
        # Check if all NaNs are consecutive at the beginning or end
        if (np.all(np.diff(nan_indices) == 1) and (nan_indices[-1] + 1 == len(data) or nan_indices[0] == 0)):
            y = data.copy()
            # Remove NaNs, process, and write back
            non_nan_indices = ~np.isnan(data)
            processed = signal.sosfiltfilt(sos, data[non_nan_indices])
            y[non_nan_indices] = processed
            return y[low_in:upp_in]
        else:
            # Return all NaNs if isolated NaNs are present
            return np.full_like(data, np.nan)[low_in:upp_in]
    else:
        return signal.sosfiltfilt(sos, data)[low_in:upp_in]


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