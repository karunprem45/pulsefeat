"""Per-beat waveform features, chosen for physiological reasons, then summarized per window.

Why these features (the mechanism, not just the model):
- amplitude: pulsatile blood volume change under the sensor; drops with vasoconstriction
- rise time (foot to peak): faster systolic upstroke with stronger ejection and stiffer arteries
- pulse width at half height: how long the vessel stays distended in each beat
- reflection timing: part of the pulse reflects back from the periphery. Stiffer arteries
  and higher pressure make the reflection come back sooner, so the late wave moves earlier
- b/a ratio of the second derivative (the "APG"): a classic marker of arterial stiffness
- heart rate and beat-to-beat variability: autonomic state, which shifts pressure
"""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz

FEATURES = ["hr_bpm", "hrv_sdnn_ms", "amp", "rise_time_s", "width50_s",
            "beat_dur_s", "area_norm", "late_wave_t_s", "apg_b_over_a"]


def _beat_features(x, fs, foot, peak, next_foot):
    seg = x[foot:next_foot]
    n = len(seg)
    if n < 5:
        return None
    base = x[foot]
    amp = x[peak] - base
    if amp <= 0:
        return None
    s = (seg - base) / amp                      # 0 at the foot, 1 at the peak
    above = np.where(s >= 0.5)[0]
    width50 = (above[-1] - above[0]) / fs if len(above) else np.nan

    p = peak - foot
    # late (reflected) wave: largest local bump or shoulder after the peak, seen in the
    # first derivative as the point where the downslope flattens most
    d1 = np.gradient(s)
    late = p + int(np.argmax(d1[p:])) if p < n - 1 else p
    d2 = np.gradient(d1)
    a_idx = int(np.argmax(d2[:max(p, 1)]))
    b_idx = a_idx + int(np.argmin(d2[a_idx:max(p, a_idx + 1)]))
    a, b = d2[a_idx], d2[b_idx]
    return {
        "amp": amp,
        "rise_time_s": p / fs,
        "width50_s": width50,
        "beat_dur_s": n / fs,
        "area_norm": float(_trapz(s) / n),
        "late_wave_t_s": (late - p) / fs,
        "apg_b_over_a": b / a if a > 0 else np.nan,
    }


def window_features(x, fs, feet, peaks):
    """Median of per-beat features across the window, plus HR and HRV."""
    rows = []
    for i in range(len(feet) - 1):
        f = _beat_features(x, fs, feet[i], peaks[i], feet[i + 1])
        if f is not None:
            rows.append(f)
    if len(rows) < 3:
        return None
    out = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0]}
    ibi = np.diff(peaks) / fs
    out["hr_bpm"] = float(60.0 / np.mean(ibi))
    out["hrv_sdnn_ms"] = float(np.std(ibi) * 1000.0)
    return {k: out[k] for k in FEATURES}
