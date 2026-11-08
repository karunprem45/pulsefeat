"""Signal conditioning for raw PPG.

PPG carries the cardiac pulse (roughly 0.5 to 3 Hz, i.e. 30 to 180 bpm, plus harmonics),
on top of slow baseline wander from breathing and movement and high-frequency noise.
A zero-phase band-pass keeps the pulse shape intact, because phase shifts would move
the timing features (rise time, notch position) that the model relies on.
"""
import numpy as np
from scipy import signal


def bandpass(x, fs, low=0.5, high=8.0, order=4):
    """Zero-phase Butterworth band-pass (filtfilt), applied as second-order sections."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def zscore(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def resample_to(x, fs_in, fs_out):
    """Polyphase resampling to a common rate, so features are comparable across datasets."""
    if fs_in == fs_out:
        return np.asarray(x, float)
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return signal.resample_poly(x, int(fs_out) // g, int(fs_in) // g)
