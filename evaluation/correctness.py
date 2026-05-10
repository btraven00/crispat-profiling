"""Stage: evaluation.correctness.

Assert that parallel runs produce assignments identical to the baseline run,
for every (method, n_jobs) combination. Given a fixed seed and the same
inputs, the per-gRNA fits must yield bit-identical results regardless of
n_jobs — that is the contract the upstream patch is making.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd


def load_assignments(run_dir: Path, method: str) -> pd.DataFrame:
    f = run_dir / method / "assignments.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Results root (e.g. results/)")
    args = ap.parse_args()

    root = Path(args.results)
    baseline_dir = root / "baseline"

    failures = []
    checked = 0

    for method in ("gauss", "poisson_gauss"):
        ref = load_assignments(baseline_dir, method)
        if ref is None:
            print(f"correctness: SKIP {method} (no baseline assignments)")
            continue

        candidates = sorted(list(root.glob("parallel_*")) + list(root.glob("hybrid_*")))
        for run_dir in candidates:
            df = load_assignments(run_dir, method)
            if df is None:
                continue
            checked += 1
            if not ref.equals(df):
                failures.append((method, run_dir.name))
                print(f"correctness: FAIL {method} {run_dir.name}")
            else:
                print(f"correctness: OK   {method} {run_dir.name}")

    print(f"\ncorrectness: {checked} comparisons, {len(failures)} failures")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
