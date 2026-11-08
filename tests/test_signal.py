import numpy as np
from scipy import signal
from pulsefeat.filters import bandpass, zscore, resample_to
from pulsefeat.beats import detect_beats, heart_rate
from pulsefeat.quality import assess
from pulsefeat.data import synthetic_record

FS = 125


def test_bandpass_removes_baseline_and_keeps_pulse():
    t = np.arange(20 * FS) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)
    y = bandpass(pulse + drift, FS)
    mid = slice(2 * FS, -2 * FS)        # ignore filter edges
    assert np.corrcoef(y[mid], pulse[mid])[0, 1] > 0.99


def test_bandpass_is_zero_phase():
    t = np.arange(20 * FS) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    y = bandpass(pulse, FS)
    lag = np.argmax(signal.correlate(y, pulse, mode="full")) - (len(t) - 1)
    assert abs(lag) <= 1


def test_resample_length():
    x = np.random.default_rng(0).normal(size=1000)
    assert len(resample_to(x, 1000, 125)) == 125


def test_heart_rate_detected_within_3_bpm():
    rng = np.random.default_rng(1)
    for hr in [55, 75, 100]:
        x = zscore(bandpass(synthetic_record(rng, seconds=30, hr=hr, noise=0.02), FS))
        feet, peaks = detect_beats(x, FS)
        assert abs(heart_rate(peaks, FS) - hr) < 3, hr
        assert np.all(feet < peaks)


def test_quality_rejects_flatline_and_clipping_and_noise():
    rng = np.random.default_rng(2)
    good = synthetic_record(rng, seconds=8)
    assert assess(good, zscore(bandpass(good, FS))).ok
    flat = np.ones(8 * FS)
    assert assess(flat, flat).reason == "flatline"
    clipped = np.clip(good, None, np.percentile(good, 70))
    assert assess(clipped, zscore(bandpass(clipped, FS))).reason == "clipping"
    noise = rng.normal(size=8 * FS)
    assert not assess(noise, zscore(bandpass(noise, FS))).ok
