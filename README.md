# PulseFeat

Estimating blood pressure from PPG (the optical pulse signal from a finger or wrist sensor). It cleans raw PPG, rejects bad windows, finds each heartbeat, extracts waveform features chosen for physiological reasons, and tests models with subject-wise cross-validation.

Blood pressure is used as the target because good public data exists for it. The same pipeline shape (condition the signal, check quality, segment beats, extract mechanistic features, validate per subject) is what pressure estimation from PPG needs in general.

## Data

Real data: the UCI **Cuff-Less Blood Pressure Estimation** dataset (Kachuee et al., built from MIMIC-II). Each record has PPG, arterial blood pressure (ABP) and ECG at 125 Hz. The ABP gives the true systolic and diastolic pressure for every window. Download it from the UCI Machine Learning Repository and put the `Part_N.mat` files in `data/`.

Synthetic data (`pulsefeat/data.py`) exists only so the tests and code can run without the download. **Never report numbers from synthetic data.**

## Pipeline

| Step | File | What it does |
|---|---|---|
| Condition | `filters.py` | Zero-phase 0.5 to 8 Hz Butterworth band-pass (removes breathing drift and high-frequency noise without shifting the waveform in time), z-score, resample to 125 Hz |
| Quality | `quality.py` | Rejects 8 s windows that are flat, clipped, not skewed like a real pulse, or not periodic |
| Beats | `beats.py` | Beat period from autocorrelation, then systolic peaks spaced at least 60% of a beat apart, and the foot before each peak |
| Features | `features.py` | Per beat: amplitude, rise time, width at half height, beat duration, normalized area, timing of the reflected wave, b/a ratio of the second derivative. Per window: medians, plus heart rate and HRV |
| Labels | `pipeline.py` | Systolic and diastolic pressure from the ABP in the same window; drops physiologically impossible values |
| Models | `model.py` | Mean baseline, ridge regression and gradient boosting, all scored with GroupKFold so no person is in both training and test |

### Why these features

- **Rise time**: the upstroke gets faster with stronger ejection and stiffer arteries.
- **Reflected wave timing**: part of each pulse bounces back from the periphery. Stiffer arteries and higher pressure bring it back sooner.
- **b/a ratio of the second derivative**: a known marker of arterial stiffness.
- **Amplitude and width**: how much the vessel fills, and for how long, which changes with vascular tone.
- **Heart rate and HRV**: autonomic state, which moves pressure.

### Why subject-wise CV

Windows from the same person look alike. A random split puts the same person in training and testing, and the error looks much better than it would on a new patient. GroupKFold prevents that.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -v                                               # tests (synthetic signals)
python -m scripts.run_pipeline --source synthetic       # code check only
python -m scripts.run_pipeline --source uci --files data/Part_1.mat --max-records 300
python -m scripts.plot_window --file data/Part_1.mat --record 0
```

Start with `--max-records 300` so the first run is quick, then run the full file.

## Things to check on real data

- **Rejection rate.** If most windows are rejected, look at which reason and plot a few. The thresholds in `quality.py` were set on synthetic data.
- **Beat detection.** Plot several windows with `plot_window`. Feet and peaks should sit on every beat.
- **The UCI files have no patient IDs.** Each record is used as its own group. Some records may come from the same patient, so the cross-validation can still be a bit optimistic. Say this if asked.
- **Compare to the baseline.** A model is only useful if it clearly beats predicting the mean.

## Limits

- Blood pressure, not intracranial pressure. There is no public PPG + ICP dataset.
- Hand-built features plus classical models. No deep learning yet.
- ICU data from MIMIC-II, which is not the same as wearable data.
