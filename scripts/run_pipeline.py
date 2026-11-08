"""Build the feature table and evaluate models.

  python -m scripts.run_pipeline --source synthetic        # code check only, never report
  python -m scripts.run_pipeline --source uci --files data/Part_1.mat --max-records 300
"""
import argparse
import os
import numpy as np

from pulsefeat.pipeline import build_table
from pulsefeat.model import evaluate, ridge_coefficients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "uci"], default="synthetic")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--window-s", type=float, default=8.0)
    args = ap.parse_args()

    if args.source == "synthetic":
        from pulsefeat.data import synthetic_dataset
        records = synthetic_dataset()
        print("SYNTHETIC DATA: this only checks the code. Do not report these numbers.\n")
    else:
        from pulsefeat.data import load_uci_part
        records = []
        for f in args.files:
            records += load_uci_part(f, args.max_records)
        print(f"loaded {len(records)} records from {len(args.files)} file(s)")

    df, rejects = build_table(records, args.window_s)
    total = len(df) + sum(rejects.values())
    print(f"windows kept: {len(df)} of {total}")
    print("rejected by reason:", rejects)
    print(f"subjects/records: {df['subject'].nunique()}")

    os.makedirs("results", exist_ok=True)
    out = f"results/features_{args.source}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}\n")

    for target in ["sbp", "dbp"]:
        print(f"== {target.upper()}: mean absolute error (mmHg), subject-wise 5-fold ==")
        for name, (mae, sd) in evaluate(df, target).items():
            print(f"  {name:18s} {mae:6.2f} +/- {sd:.2f}")
    print("\n== standardized ridge coefficients for SBP (sign = direction) ==")
    coefs = ridge_coefficients(df, "sbp")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:16s} {v:+.2f}")


if __name__ == "__main__":
    main()
