"""Signal quality checks. Bad windows are dropped before any feature is computed.

Each check targets a real failure mode of optical sensors:
- flatline: sensor off the skin or saturated
- clipping: amplifier at its rail, which flattens the systolic peaks
- skewness: clean PPG is clearly skewed (sharp upstroke, slow decay); motion noise is
  closer to symmetric, so low skewness flags a corrupted window
- periodicity: clean PPG repeats every beat; noise does not
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats

from .beats import dominant_period


@dataclass
class Quality:
    ok: bool
    reason: str
    skewness: float
    clipped_frac: float
    periodicity: float = 0.0


def assess(raw, filtered, fs=125, min_skew=0.2, max_clip=0.02, min_period_corr=0.5, min_std=1e-6):
    raw = np.asarray(raw, float)
    if raw.std() < min_std:
        return Quality(False, "flatline", 0.0, 0.0)
    lo, hi = raw.min(), raw.max()
    tol = 0.001 * (hi - lo)
    clipped = np.mean((raw >= hi - tol) | (raw <= lo + tol))
    # a few samples always sit at the extremes; long runs at the rail mean clipping
    skew = float(stats.skew(filtered))
    if clipped > max_clip:
        return Quality(False, "clipping", skew, float(clipped))
    if skew < min_skew:
        return Quality(False, "low_skewness", skew, float(clipped))
    _, corr = dominant_period(filtered, fs)
    if corr < min_period_corr:
        return Quality(False, "not_periodic", skew, float(clipped), corr)
    return Quality(True, "ok", skew, float(clipped), corr)
