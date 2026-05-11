"""Stage: plots — E1b phase-resolved figures.

Reads phases_parent.csv and phases_workers.csv produced by
evaluation/phases.py and emits:

  1. fit_one_distribution.png
       Per n_jobs, distribution of per-gRNA fit times across all workers.
       Median (point), p5–p95 ribbon, min/max whiskers.

  2. parent_phases.png
       Stacked bars: parent-side phase wall time per n_jobs. Visualises
       the empirical "serial floor" (load_h5ad + pool_setup + assemble +
       write_csv) that USL α should be attributable to.

  3. worker_active_time.png
       Per n_jobs, distribution of per-worker total active time
       (sum of fit_one durations). The max of this distribution is the
       straggler bound on fit_loop wall — i.e. the β-attributable tail.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_linerange, geom_errorbar, geom_col,
    geom_text,
    scale_x_log10, scale_y_log10, scale_x_continuous,
    labs, theme, theme_minimal, element_text, ggsave, position_stack,
)


def _quantiles_per_njobs(df, col):
    g = df.groupby("n_jobs")[col]
    return pd.DataFrame({
        "n_jobs": g.median().index,
        "min": g.min().values,
        "p05": g.quantile(0.05).values,
        "median": g.median().values,
        "p95": g.quantile(0.95).values,
        "max": g.max().values,
        "count": g.count().values,
    })


def plot_fit_one_distribution(workers: pd.DataFrame, out: Path):
    fit = workers[workers["event"] == "fit_one"].dropna(subset=["duration_s"])
    if fit.empty:
        print("plots: no fit_one rows")
        return
    q = _quantiles_per_njobs(fit, "duration_s")
    p = (
        ggplot(q, aes(x="n_jobs"))
        + geom_linerange(aes(ymin="min", ymax="max"), color="grey", size=0.6)
        + geom_linerange(aes(ymin="p05", ymax="p95"), color="steelblue", size=2.0, alpha=0.8)
        + geom_point(aes(y="median"), size=2.8, color="black")
        + scale_x_log10()
        + labs(
            x="n_jobs", y="per-gRNA fit_one duration (s)",
            title="fit_one duration distribution per n_jobs",
            subtitle="Black point = median.  Blue bar = p5–p95.  Grey whisker = min–max.",
        )
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"))
    )
    ggsave(p, str(out), dpi=150, width=7, height=4.5, units="in")
    print(f"plots: wrote {out}")


def plot_parent_phases(parent: pd.DataFrame, out: Path):
    # Restrict to inner phases (skip outer run/method spans that bracket
    # everything else — they'd dominate the stack).
    inner = ("load_h5ad", "pool_setup", "fit_loop",
             "assemble", "write_csv", "combine_batches")
    df = parent[parent["event"].isin(inner)].copy()
    if df.empty:
        print("plots: no parent inner-phase rows")
        return
    # If a phase fires per-batch (gauss has batches), sum within (n_jobs, event).
    agg = df.groupby(["n_jobs", "event"], as_index=False)["duration_s"].sum()
    # Preserve a stable phase order top-to-bottom.
    agg["event"] = pd.Categorical(agg["event"], categories=list(inner), ordered=True)
    p = (
        ggplot(agg, aes(x="factor(n_jobs)", y="duration_s", fill="event"))
        + geom_col(position="stack")
        + labs(
            x="n_jobs", y="wall time (s)",
            title="Parent-side phase breakdown per n_jobs",
            subtitle="fit_loop = waiting on workers (parallelisable).  "
                     "Others = serial floor candidate.",
        )
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"))
    )
    ggsave(p, str(out), dpi=150, width=8, height=4.5, units="in")
    print(f"plots: wrote {out}")


def plot_worker_active_time(workers: pd.DataFrame, out: Path):
    fit = workers[workers["event"] == "fit_one"].dropna(subset=["duration_s"])
    if fit.empty:
        print("plots: no fit_one rows for worker-active-time")
        return
    per_worker = (
        fit.groupby(["n_jobs", "pid"])["duration_s"].sum()
        .reset_index().rename(columns={"duration_s": "active_s"})
    )
    q = _quantiles_per_njobs(per_worker.rename(columns={"active_s": "duration_s"}),
                             "duration_s")
    p = (
        ggplot(q, aes(x="n_jobs"))
        + geom_linerange(aes(ymin="min", ymax="max"), color="grey", size=0.6)
        + geom_linerange(aes(ymin="p05", ymax="p95"), color="darkorange", size=2.0, alpha=0.8)
        + geom_point(aes(y="median"), size=2.8, color="black")
        + scale_x_log10()
        + labs(
            x="n_jobs", y="per-worker total active time (s)",
            title="Worker active-time distribution per n_jobs",
            subtitle="Max of this = straggler bound on fit_loop wall (β tail).  "
                     "Median = typical worker load.",
        )
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"))
    )
    ggsave(p, str(out), dpi=150, width=7, height=4.5, units="in")
    print(f"plots: wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Root containing phases_*.csv (e.g. results-e1-instr).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_csv = root / "phases_parent.csv"
    workers_csv = root / "phases_workers.csv"
    if not parent_csv.exists() or not workers_csv.exists():
        raise SystemExit(
            f"phase CSVs missing in {root} — run evaluation/phases.py first"
        )

    def _safe_read(p):
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    parent = _safe_read(parent_csv)
    workers = _safe_read(workers_csv)
    if parent.empty and workers.empty:
        print("plots: both phase CSVs are empty — nothing to plot")
        return

    plot_fit_one_distribution(workers, out_dir / "fit_one_distribution.png")
    plot_parent_phases(parent, out_dir / "parent_phases.png")
    plot_worker_active_time(workers, out_dir / "worker_active_time.png")


if __name__ == "__main__":
    main()
