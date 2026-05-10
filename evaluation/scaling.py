"""Stage: evaluation.scaling.

Parse obkit-events.jsonl from each run, extract the duration of the
fit_gRNAs / ga_gauss / ga_poisson_gauss spans, and emit a tidy CSV with
columns: variant, method, n_jobs, phase, wall_seconds, speedup, efficiency.

Speedup is computed against the baseline run for each method.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PHASES_OF_INTEREST = ("ga_gauss", "ga_poisson_gauss", "fit_gRNAs", "load_h5ad", "write_outputs")


def parse_events(jsonl: Path) -> pd.DataFrame:
    rows = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(rec)
    return pd.DataFrame(rows)


def span_durations(events: pd.DataFrame) -> dict:
    out = {}
    if events.empty:
        return out
    events = events.copy()
    events["ts"] = pd.to_datetime(events["ts"], format="ISO8601", utc=True)
    for ev_name, group in events.groupby("event"):
        starts = group[group["phase"] == "start"].sort_values("ts")
        ends = group[group["phase"] == "end"].sort_values("ts")
        if len(starts) and len(ends):
            dt = (ends["ts"].iloc[-1] - starts["ts"].iloc[0]).total_seconds()
            out[ev_name] = dt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="Output CSV (default: <results>/scaling.csv)")
    args = ap.parse_args()

    root = Path(args.results)
    out_csv = Path(args.out) if args.out else root / "scaling.csv"

    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        ev_file = run_dir / "obkit-events.jsonl"
        if not ev_file.exists():
            continue
        events = parse_events(ev_file)
        if events.empty:
            continue

        # variant + n_jobs from the 'run' start event attrs
        meta = events[(events["event"] == "run") & (events["phase"] == "start")]
        variant = meta["attrs"].iloc[0].get("variant") if len(meta) else run_dir.name
        n_jobs = meta["attrs"].iloc[0].get("n_jobs", 1) if len(meta) else 1

        durations = span_durations(events)
        for phase, secs in durations.items():
            if phase not in PHASES_OF_INTEREST:
                continue
            method = phase if phase.startswith("ga_") else None
            rows.append({
                "run": run_dir.name,
                "variant": variant,
                "n_jobs": n_jobs,
                "method": method,
                "phase": phase,
                "wall_seconds": secs,
            })

    df = pd.DataFrame(rows)

    # Speedup vs baseline (n_jobs=1, variant=baseline) per method
    if not df.empty:
        baseline = df[(df["variant"] == "baseline") & (df["phase"].isin(("ga_gauss", "ga_poisson_gauss")))]
        baseline_map = dict(zip(baseline["phase"], baseline["wall_seconds"]))
        df["speedup"] = df.apply(
            lambda r: baseline_map.get(r["phase"], float("nan")) / r["wall_seconds"]
            if r["phase"] in baseline_map and r["wall_seconds"] > 0 else float("nan"),
            axis=1,
        )
        df["efficiency"] = df["speedup"] / df["n_jobs"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"scaling: {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
