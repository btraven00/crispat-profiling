"""E2-specific plots: hybrid (n_jobs × omp_threads) sweep + memory cost.

Outputs:
    results/plots/e2_hybrid.png   speedup vs n_jobs, hybrid vs single-axis.
    results/plots/e2_memory.png   peak aggregated RSS vs n_jobs.

Together they tell the trade story: process parallelism wins on speed
(e2_hybrid) but each added process costs you memory (e2_memory). The
memory panel justifies the BPCells/scx-via-picklerick note in BUGS.md.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_text, geom_abline,
    scale_x_log10, scale_y_log10, scale_color_brewer,
    labs, theme, theme_minimal, element_text, ggsave,
)


HYBRID_RE = re.compile(r"hybrid_n(\d+)_omp(\d+)")
PARALLEL_RE = re.compile(r"parallel_n(\d+)")


def build_e2_table(scaling_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scaling_csv)
    df = df[df["phase"] == "ga_gauss"].copy()

    rows = []
    for _, r in df.iterrows():
        run = r["run"]
        if run == "baseline":
            rows.append({**r.to_dict(), "config": "single-axis (omp=1)", "omp": 1})
            continue
        m = HYBRID_RE.match(run)
        if m:
            n, omp = int(m.group(1)), int(m.group(2))
            rows.append({**r.to_dict(), "config": "hybrid (8-core budget)", "omp": omp})
            continue
        m = PARALLEL_RE.match(run)
        if m:
            rows.append({**r.to_dict(), "config": "single-axis (omp=1)", "omp": 1})
    return pd.DataFrame(rows)


def peak_aggregated_rss_mb(denet_jsonl: Path) -> float | None:
    """Peak aggregated RSS in MB across the run."""
    if not denet_jsonl.exists():
        return None
    peak = 0.0
    with denet_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            block = rec.get("aggregated") or rec.get("parent")
            if not block:
                continue
            rss_kb = block.get("mem_rss_kb") or 0
            if rss_kb > peak:
                peak = rss_kb
    return peak / 1024.0 if peak else None


def plot_memory(root: Path, out_dir: Path):
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = HYBRID_RE.match(run_dir.name)
        if m:
            n, omp = int(m.group(1)), int(m.group(2))
            config = "hybrid (8-core budget)"
        else:
            m = PARALLEL_RE.match(run_dir.name)
            if m:
                n, omp = int(m.group(1)), 1
                config = "single-axis (omp=1)"
            elif run_dir.name == "baseline":
                n, omp = 1, 1
                config = "single-axis (omp=1)"
            else:
                continue
        rss = peak_aggregated_rss_mb(run_dir / "denet.jsonl")
        if rss is not None:
            rows.append({"n_jobs": n, "omp": omp, "config": config,
                         "peak_rss_mb": rss,
                         "label": f"({n},{omp})"})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    p = (
        ggplot(df, aes(x="n_jobs", y="peak_rss_mb", color="config"))
        + geom_line(size=1)
        + geom_point(size=3)
        + geom_text(aes(label="label"), nudge_y=0.04, size=8, show_legend=False)
        + scale_x_log10(breaks=[1, 2, 4, 8, 16])
        + scale_y_log10()
        + labs(
            x="n_jobs (process count)",
            y="peak aggregated RSS (MB)",
            title="E2 — Memory cost of process parallelism",
            subtitle="RSS sums parent + all worker processes (denet's aggregated block).",
            color="",
        )
        + theme_minimal()
        + theme(legend_position="top", plot_subtitle=element_text(size=9, color="#555"))
    )
    out_path = out_dir / "e2_memory.png"
    ggsave(p, str(out_path), dpi=150, width=7, height=5, units="in")
    print(f"e2: wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_e2_table(root / "scaling.csv")
    if df.empty:
        print("e2: no data")
        return

    df["label"] = df.apply(lambda r: f"({int(r['n_jobs'])},{int(r['omp'])})", axis=1)

    p = (
        ggplot(df, aes(x="n_jobs", y="speedup", color="config"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="grey")
        + geom_line(size=1)
        + geom_point(size=3)
        + geom_text(aes(label="label"), nudge_y=0.08, size=8, show_legend=False)
        + scale_x_log10(breaks=[1, 2, 4, 8, 16])
        + scale_y_log10()
        + labs(
            x="n_jobs (process count)",
            y="speedup vs serial baseline",
            title="E2 — Process parallelism wins on crispat ga_gauss",
            subtitle="Labels: (n_jobs, omp_threads).  Dashed line = ideal linear scaling.",
            color="",
        )
        + theme_minimal()
        + theme(legend_position="top", plot_subtitle=element_text(size=9, color="#555"))
    )
    out_path = out_dir / "e2_hybrid.png"
    ggsave(p, str(out_path), dpi=150, width=7, height=5, units="in")
    print(f"e2: wrote {out_path}")

    plot_memory(root, out_dir)


if __name__ == "__main__":
    main()
