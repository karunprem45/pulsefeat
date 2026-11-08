"""Plot one window: raw signal, filtered signal with detected feet and peaks.

Saves results/window.png. Uses a synthetic record unless --file is given.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsefeat.filters import bandpass, zscore
from pulsefeat.beats import detect_beats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="a UCI Part_N.mat file")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--start-s", type=float, default=0.0)
    args = ap.parse_args()
    fs = 125
    if args.file:
        from pulsefeat.data import load_uci_part
        raw_all = load_uci_part(args.file, args.record + 1)[args.record]["ppg"]
    else:
        from pulsefeat.data import synthetic_record
        raw_all = synthetic_record(np.random.default_rng(0), seconds=20)
    a = int(args.start_s * fs)
    raw = raw_all[a:a + 8 * fs]
    x = zscore(bandpass(raw, fs))
    feet, peaks = detect_beats(x, fs)
    t = np.arange(len(x)) / fs

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(t, raw, lw=1)
    ax[0].set_ylabel("raw PPG")
    ax[1].plot(t, x, lw=1)
    ax[1].plot(t[peaks], x[peaks], "v", label="systolic peak")
    ax[1].plot(t[feet], x[feet], "^", label="foot")
    ax[1].set_ylabel("filtered (z-score)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/window.png", dpi=120)
    print("saved results/window.png")


if __name__ == "__main__":
    main()
