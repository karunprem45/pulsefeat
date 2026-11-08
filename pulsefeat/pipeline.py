"""Record -> fixed windows -> filter -> quality check -> beats -> features (+ labels)."""
import numpy as np
from scipy import signal

from .filters import bandpass, zscore, resample_to
from .quality import assess
from .beats import detect_beats
from .features import window_features, FEATURES

TARGET_FS = 125


def abp_labels(abp, fs):
    """Systolic and diastolic pressure for a window: median of beat maxima and minima."""
    peaks, _ = signal.find_peaks(abp, distance=int(fs * 60 / 180))
    troughs, _ = signal.find_peaks(-abp, distance=int(fs * 60 / 180))
    if len(peaks) < 3 or len(troughs) < 3:
        return None
    sbp, dbp = float(np.median(abp[peaks])), float(np.median(abp[troughs]))
    if not (60 <= sbp <= 220 and 30 <= dbp <= 130 and sbp > dbp):
        return None       # implausible: catheter flush or artifact
    return sbp, dbp


def process_record(rec, window_s=8.0):
    """Return (rows, rejects). Each row has subject, features and labels."""
    fs = rec["fs"]
    ppg = resample_to(rec["ppg"], fs, TARGET_FS)
    abp = resample_to(rec["abp"], fs, TARGET_FS) if "abp" in rec else None
    fs = TARGET_FS
    n = int(window_s * fs)
    rows, rejects = [], {}
    for start in range(0, len(ppg) - n + 1, n):
        raw = ppg[start:start + n]
        filt = zscore(bandpass(raw, fs))
        q = assess(raw, filt, fs)
        if not q.ok:
            rejects[q.reason] = rejects.get(q.reason, 0) + 1
            continue
        feet, peaks = detect_beats(filt, fs)
        feats = window_features(filt, fs, feet, peaks)
        if feats is None or any(np.isnan(v) for v in feats.values()):
            rejects["too_few_beats"] = rejects.get("too_few_beats", 0) + 1
            continue
        if abp is not None:
            lab = abp_labels(abp[start:start + n], fs)
            if lab is None:
                rejects["bad_abp"] = rejects.get("bad_abp", 0) + 1
                continue
            sbp, dbp = lab
        else:
            sbp, dbp = rec["sbp"], rec["dbp"]
        rows.append({"subject": rec["subject"], **feats, "sbp": sbp, "dbp": dbp})
    return rows, rejects


def build_table(records, window_s=8.0):
    import pandas as pd
    rows, rejects = [], {}
    for rec in records:
        r, rej = process_record(rec, window_s)
        rows.extend(r)
        for k, v in rej.items():
            rejects[k] = rejects.get(k, 0) + v
    return pd.DataFrame(rows, columns=["subject", *FEATURES, "sbp", "dbp"]), rejects
