"""Beat detection: find each systolic peak and the foot (onset) before it."""
import numpy as np
from scipy import signal


def dominant_period(x, fs, min_hr=40.0, max_hr=180.0):
    """Beat period (samples) from the autocorrelation peak, plus its strength (0 to 1).

    A clean PPG repeats every beat, so its autocorrelation has a strong peak at one
    beat period. Noise doesn't. This also gives a spacing for peak detection, so the
    reflected wave in a slow heartbeat isn't mistaken for a second beat.
    """
    x = np.asarray(x, float) - np.mean(x)
    ac = signal.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    lo, hi = int(fs * 60.0 / max_hr), int(fs * 60.0 / min_hr)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return None, 0.0
    peaks, _ = signal.find_peaks(ac[lo:hi + 1])
    if len(peaks) == 0:
        return None, 0.0
    best = peaks[np.argmax(ac[lo + peaks])] + lo
    if ac[best] <= 0:
        return None, 0.0
    # prefer the shortest lag that is nearly as strong (avoids picking 2 beats)
    strong = [p + lo for p in peaks if ac[p + lo] >= 0.8 * ac[best]]
    lag = min(strong)
    return int(lag), float(ac[lag])


def detect_beats(x, fs, prominence=0.3):
    """Return (feet, peaks) sample indices for a band-passed, z-scored PPG.

    Peaks must be at least 60% of a beat apart. The foot of each beat is the minimum
    between the previous peak and this one.
    """
    period, _ = dominant_period(x, fs)
    min_dist = int(0.6 * period) if period else int(fs * 60.0 / 180.0)
    peaks, _ = signal.find_peaks(x, distance=max(min_dist, 1), prominence=prominence)
    feet, kept = [], []
    for i in range(1, len(peaks)):
        lo, hi = peaks[i - 1], peaks[i]
        feet.append(lo + int(np.argmin(x[lo:hi])))
        kept.append(hi)
    return np.array(feet, int), np.array(kept, int)


def heart_rate(peaks, fs):
    """Mean heart rate in bpm from peak-to-peak intervals."""
    if len(peaks) < 2:
        return float("nan")
    return float(60.0 * fs / np.mean(np.diff(peaks)))
