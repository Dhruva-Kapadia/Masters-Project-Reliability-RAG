#!/usr/bin/env python3
"""Concatenate all per-config CSVs for one run into a single summary table.

Usage:
    uv run python scripts/aggregate_results.py --run_dir runs3
    # or, to point directly at a csv directory:
    uv run python scripts/aggregate_results.py --output_dir runs3/output --out runs3/output/summary.csv
"""
import argparse
import glob
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=None, help='runsN directory; sets output_dir=<run_dir>/output and out=<run_dir>/output/summary.csv unless overridden')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.run_dir is not None:
        args.output_dir = args.output_dir or os.path.join(args.run_dir, 'output')
        args.out = args.out or os.path.join(args.run_dir, 'output', 'summary.csv')
    else:
        args.output_dir = args.output_dir or 'output'
        args.out = args.out or 'output/summary.csv'

    files = sorted(glob.glob(os.path.join(args.output_dir, '*.csv')))
    files = [f for f in files if os.path.basename(f) != os.path.basename(args.out)]

    if not files:
        print(f"No CSVs found in {args.output_dir}")
        return

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(['dataset_name', 'defense_method', 'attack_method', 'model_name'])
    summary.to_csv(args.out, index=False)

    print(f"Aggregated {len(files)} run(s) -> {args.out}")
    cols = ['dataset_name', 'defense_method', 'attack_method', 'acc', 'asr', 'dataset_size']
    print(summary[cols].to_string(index=False))


if __name__ == '__main__':
    main()