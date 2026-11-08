"""Data sources: a synthetic PPG generator for tests, and a loader for real data.

Real data: the UCI "Cuff-Less Blood Pressure Estimation" dataset (Kachuee et al.), built
from MIMIC-II. Each record has PPG, arterial blood pressure (ABP) and ECG at 125 Hz.
ABP gives the ground-truth systolic and diastolic pressure for every window.

Synthetic data is ONLY for testing the code. Never report results from it.
"""
import numpy as np

UCI_FS = 125


def synthetic_record(rng, seconds=60, fs=125, sbp=120.0, hr=70.0, noise=0.05,
                     motion=False, subject_offset=0.0):
    """One PPG record whose shape depends on systolic pressure (sbp).

    Each beat = systolic Gaussian + reflected Gaussian. Higher pressure makes the
    reflected wave arrive earlier and stronger, and shortens the upstroke.
    """
    t = np.arange(int(seconds * fs)) / fs
    x = np.zeros_like(t)
    beat = 60.0 / hr
    k = (sbp - 120.0) / 40.0 + subject_offset         # about -0.75 .. +1.0
    sys_t = 0.15 * beat * (1 - 0.15 * k)
    refl_t = sys_t + 0.30 * beat * (1 - 0.25 * k)
    refl_a = 0.45 * (1 + 0.35 * k)
    start = 0.0
    while start < seconds:
        jitter = rng.normal(0, 0.02 * beat)
        x += np.exp(-((t - start - sys_t) ** 2) / (2 * (0.06 * beat) ** 2))
        x += refl_a * np.exp(-((t - start - refl_t) ** 2) / (2 * (0.09 * beat) ** 2))
        start += beat + jitter
    x += 0.3 * np.sin(2 * np.pi * 0.25 * t)            # breathing baseline wander
    x += rng.normal(0, noise, len(t))
    if motion:
        x += rng.normal(0, 1.5, len(t)).cumsum() * 0.05  # motion artifact
    return x


def synthetic_dataset(n_subjects=30, records_per_subject=4, seed=0, seconds=60, fs=125):
    """Return a list of dicts: subject, ppg, fs, sbp, dbp (one constant label per record)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_subjects):
        offset = rng.normal(0, 0.1)
        for _ in range(records_per_subject):
            sbp = rng.uniform(95, 165)
            hr = rng.uniform(55, 100)
            out.append({"subject": s, "fs": fs, "sbp": sbp, "dbp": 0.6 * sbp + rng.normal(0, 4),
                        "ppg": synthetic_record(rng, seconds, fs, sbp, hr,
                                                motion=rng.random() < 0.1,
                                                subject_offset=offset)})
    return out


def load_uci_part(path, max_records=None):
    """Load one Part_N.mat file. Returns a list of dicts with ppg and abp arrays.

    The files are MATLAB cell arrays of 3 x N matrices (rows: PPG, ABP, ECG). Older
    MATLAB files load with scipy; v7.3 files need h5py.
    """
    import os
    name = os.path.splitext(os.path.basename(path))[0]
    records = []
    try:
        from scipy.io import loadmat
        cells = loadmat(path)[name]
        mats = [cells[0, i] for i in range(cells.shape[1])]
    except NotImplementedError:
        import h5py
        f = h5py.File(path, "r")
        refs = f[name]
        mats = [np.array(f[refs[i, 0]]).T for i in range(refs.shape[0])]
    for i, m in enumerate(mats):
        if max_records is not None and i >= max_records:
            break
        m = np.asarray(m, float)
        if m.shape[0] != 3:
            m = m.T
        records.append({"subject": f"{name}_{i}", "fs": UCI_FS, "ppg": m[0], "abp": m[1]})
    return records
