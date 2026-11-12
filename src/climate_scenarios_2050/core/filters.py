"""
Butterworth filters for 1D series (zero-phase). Leading/trailing NaNs are kept, series with NaNs
in between return all NaN.

Replaces the old `filters.py`: `pad` now works for all positive integers (`isinstance(pad,int) & pad>0`
rejected even values) and is applied after the NaNs are cut off (the NaN mask did not match the padded series).
"""

import numpy as np
from scipy import signal


def _apply(data, sos, pad=None):
    data = np.asarray(data, dtype=float)
    out = np.full_like(data, np.nan)
    valid = np.flatnonzero(~np.isnan(data))
    if valid.size == 0:
        return out
    first, last = valid[0], valid[-1]
    if valid.size != last - first + 1:
        return out
    core = data[first : last + 1]
    if pad is not None:
        if not (isinstance(pad, (int, np.integer)) and pad > 0):
            raise ValueError("pad has to be a positive integer or None")
        core = np.r_[core[pad:0:-1], core, core[-2 : -pad - 2 : -1]]
        out[first : last + 1] = signal.sosfiltfilt(sos, core)[pad:-pad]
    else:
        out[first : last + 1] = signal.sosfiltfilt(sos, core)
    return out


def lowpass(data, cutoff_period, order=4, pad=None, fs=1):
    """keep periods longer than `cutoff_period` (in time steps / fs)"""
    sos = signal.butter(order, 1 / cutoff_period, fs=fs, btype="low", output="sos")
    return _apply(data, sos, pad)


def bandpass(data, short_period, long_period, order=4, pad=None, fs=1):
    """keep periods between `short_period` and `long_period`"""
    sos = signal.butter(order, [1 / long_period, 1 / short_period], fs=fs, btype="band", output="sos")
    return _apply(data, sos, pad)
