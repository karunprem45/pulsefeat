import numpy as np
from pulsefeat.data import synthetic_record, synthetic_dataset
from pulsefeat.filters import bandpass, zscore
from pulsefeat.beats import detect_beats
from pulsefeat.features import window_features, FEATURES
from pulsefeat.pipeline import build_table, abp_labels
from pulsefeat.model import evaluate

FS = 125


def _feats(sbp, seed=0):
    rng = np.random.default_rng(seed)
    x = zscore(bandpass(synthetic_record(rng, seconds=8, sbp=sbp, hr=70, noise=0.01), FS))
    feet, peaks = detect_beats(x, FS)
    return window_features(x, FS, feet, peaks)


def test_all_features_present_and_finite():
    f = _feats(120)
    assert list(f) == FEATURES
    assert all(np.isfinite(v) for v in f.values())


def test_reflected_wave_moves_earlier_at_higher_pressure():
    # the generator's physics: higher pressure -> earlier reflection
    assert _feats(160)["late_wave_t_s"] < _feats(100)["late_wave_t_s"]


def test_abp_labels_and_rejects_implausible():
    t = np.arange(8 * FS) / FS
    abp = 95 + 25 * np.sin(2 * np.pi * 1.2 * t)
    sbp, dbp = abp_labels(abp, FS)
    assert abs(sbp - 120) < 1 and abs(dbp - 70) < 1
    assert abp_labels(np.full(8 * FS, 300.0) + np.sin(2 * np.pi * 1.2 * t), FS) is None


def test_models_beat_baseline_with_subject_wise_cv():
    df, _ = build_table(synthetic_dataset(n_subjects=15, records_per_subject=3, seconds=40))
    assert df["subject"].nunique() == 15
    res = evaluate(df, n_splits=5)
    assert res["gradient_boosting"][0] < res["baseline_mean"][0]
    assert res["ridge"][0] < res["baseline_mean"][0]
