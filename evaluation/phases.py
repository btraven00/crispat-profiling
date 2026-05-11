"""Stage: evaluation.phases — aggregator for the E1b instrumented sweep.

Walks <results_root>/parallel_n*/, parses parent obkit-events.jsonl and the
per-worker obkit-events.worker.<pid>.jsonl files written by the
parallel-gauss-instr branch, and emits two tidy CSVs:

  - phases_parent.csv:  n_jobs, event, duration_s
        One row per parent-side phase span (load_h5ad, pool_setup, fit_loop,
        assemble, write_csv, combine_batches, plus the outer ga_gauss /
        ga_poisson_gauss / run spans emitted by methods/parallel/run.py).

  - phases_workers.csv: n_jobs, pid, event, gRNA, duration_s
        One row per worker event span. event is "fit_one" with the gRNA in
        the gRNA column, or "worker_init" with gRNA=NaN. duration_s is the
        end-start delta in seconds.

Both CSVs are written into <results_root>/.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _pair_spans(events):
    """Walk events in arrival order; emit (event, gRNA, start_ts, end_ts).

    obkit emits one record per (event, phase). Within a single file the
    process is single-threaded, so a 'start' is always followed by its
    matching 'end' before another start for the same (event, gRNA) key.
    Pair via a stack keyed by (event, gRNA).
    """
    stacks = {}
    out = []
    for rec in events:
        ev = rec.get("event")
        ph = rec.get("phase")
        attrs = rec.get("attrs") or {}
        gRNA = attrs.get("gRNA")
        key = (ev, gRNA)
        ts = pd.to_datetime(rec["ts"], format="ISO8601", utc=True)
        if ph == "start":
            stacks.setdefault(key, []).append(ts)
        elif ph == "end":
            stack = stacks.get(key) or []
            if stack:
                start = stack.pop()
                out.append((ev, gRNA, start, ts))
            else:
                # end without a start — worker_init is end-only by design.
                out.append((ev, gRNA, pd.NaT, ts))
    return out


def _read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


_NJOBS_RE = re.compile(r"parallel_n(\d+)$")


def aggregate(root: Path):
    parent_rows, worker_rows = [], []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _NJOBS_RE.match(run_dir.name)
        if not m:
            continue
        n_jobs = int(m.group(1))

        parent_log = run_dir / "obkit-events.jsonl"
        if parent_log.exists():
            spans = _pair_spans(_read_jsonl(parent_log))
            for ev, gRNA, t_start, t_end in spans:
                if pd.isna(t_start):
                    continue
                parent_rows.append({
                    "n_jobs": n_jobs,
                    "event": ev,
                    "duration_s": (t_end - t_start).total_seconds(),
                })

        for worker_log in sorted(run_dir.glob("obkit-events.worker.*.jsonl")):
            pid_m = re.search(r"worker\.(\d+)\.jsonl$", worker_log.name)
            pid = int(pid_m.group(1)) if pid_m else -1
            recs = _read_jsonl(worker_log)
            if not recs:
                continue
            t_first = pd.to_datetime(recs[0]["ts"], format="ISO8601", utc=True)
            t_last = pd.to_datetime(recs[-1]["ts"], format="ISO8601", utc=True)
            spans = _pair_spans(recs)
            for ev, gRNA, t_start, t_end in spans:
                if pd.isna(t_start):
                    # worker_init is end-only; record duration as NaN.
                    worker_rows.append({
                        "n_jobs": n_jobs, "pid": pid,
                        "event": ev, "gRNA": gRNA,
                        "duration_s": float("nan"),
                        "t_first_event": t_first, "t_last_event": t_last,
                    })
                    continue
                worker_rows.append({
                    "n_jobs": n_jobs, "pid": pid,
                    "event": ev, "gRNA": gRNA,
                    "duration_s": (t_end - t_start).total_seconds(),
                    "t_first_event": t_first, "t_last_event": t_last,
                })

    parent = pd.DataFrame(parent_rows)
    workers = pd.DataFrame(worker_rows)
    return parent, workers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Root containing parallel_n*/ subdirs (e.g. results-e1-instr).")
    args = ap.parse_args()
    root = Path(args.results)

    parent, workers = aggregate(root)

    out_parent = root / "phases_parent.csv"
    out_workers = root / "phases_workers.csv"
    parent.to_csv(out_parent, index=False)
    workers.to_csv(out_workers, index=False)
    print(f"phases: wrote {out_parent}  ({len(parent)} rows)")
    print(f"phases: wrote {out_workers} ({len(workers)} rows)")

    if not parent.empty:
        print("\nParent phase totals (s) per n_jobs:")
        pivot = parent.groupby(["n_jobs", "event"])["duration_s"].sum().unstack(fill_value=0)
        print(pivot.round(3).to_string())

    if not workers.empty:
        fit = workers[workers["event"] == "fit_one"].dropna(subset=["duration_s"])
        if not fit.empty:
            print("\nfit_one duration distribution (s):")
            summary = fit.groupby("n_jobs")["duration_s"].agg(
                count="count", min="min",
                p05=lambda s: s.quantile(0.05),
                median="median",
                p95=lambda s: s.quantile(0.95),
                max="max", mean="mean",
            )
            print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
