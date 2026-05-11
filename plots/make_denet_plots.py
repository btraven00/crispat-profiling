"""Stage: plots — denet eBPF figures.

Reads results-e1-instr/parallel_n*/denet.jsonl and produces three views
designed to test the hypothesis that the per-fit_one slowdown at high
n_jobs is contention-driven:

  1. ebpf_offcpu.png
       Off-CPU time (total + per-event mean) vs n_jobs. Off-CPU time is
       the wall time threads spent blocked — on a lock, page-fault, I/O
       wait, etc. If fit_one slowdown is real contention, this curve
       rises with n_jobs.

  2. ebpf_syscall_categories.png
       Stacked bar of syscall-category share (memory / file_io / process
       / ipc / ...) per n_jobs. A "memory" spike with mmap as the top
       syscall = NUMA / page-fault pressure. A "process" spike =
       fork/clone cost from pool spawn.

  3. ebpf_syscall_timeline_n64.png
       Single-run timeline (n=64): syscall rate, CPU%, and obkit phase
       spans on a shared time axis. Shows which phase generates which
       syscall mix.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_col, geom_segment, geom_text,
    facet_grid, scale_x_log10, scale_x_continuous, scale_y_continuous,
    labs, theme, theme_minimal, element_text, ggsave, position_stack,
)


_NJOBS_RE = re.compile(r"parallel_n(\d+)$")


def _read_denet_full(path: Path):
    """Return list of dicts with ts_ms, cpu, rss_mb, syscalls_by_category,
    syscall_rate_per_sec, offcpu_total_ns, offcpu_avg_ns, offcpu_events."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ts_ms" not in rec:
                continue
            agg = rec.get("aggregated") or {}
            ebpf = agg.get("ebpf") or {}
            sc = ebpf.get("syscalls") or {}
            an = sc.get("analysis") or {}
            oc = ebpf.get("offcpu") or {}
            rows.append({
                "ts_ms": rec["ts_ms"],
                "cpu": agg.get("cpu_usage"),
                "rss_mb": (agg.get("mem_rss_kb") or 0) / 1024.0,
                "syscalls_total": sc.get("total") or 0,
                "by_category": sc.get("by_category") or {},
                "syscall_rate": an.get("syscall_rate_per_sec") or 0.0,
                "io_intensity": an.get("io_intensity") or 0.0,
                "memory_intensity": an.get("memory_intensity") or 0.0,
                "offcpu_total_ns": oc.get("total_time_ns") or 0,
                "offcpu_avg_ns": oc.get("avg_time_ns") or 0,
                "offcpu_max_ns": oc.get("max_time_ns") or 0,
                "offcpu_events": oc.get("total_events") or 0,
            })
    return rows


def _read_denet_mem(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ts_ms" not in rec:
                continue
            agg = rec.get("aggregated") or {}
            rows.append({
                "ts_ms": rec["ts_ms"],
                "rss_mb": (agg.get("mem_rss_kb") or 0) / 1024.0,
                "vms_mb": (agg.get("mem_vms_kb") or 0) / 1024.0,
                "pf_cached": agg.get("page_faults_cached") or 0,
                "pf_disk": agg.get("page_faults_disk") or 0,
                "procs": agg.get("process_count") or 0,
            })
    return pd.DataFrame(rows)


def _per_run_summary(run_dir: Path):
    """One row per run. Cumulative fields take the last sample; rates take
    the mean of in-progress samples."""
    samples = _read_denet_full(run_dir / "denet.jsonl")
    if not samples:
        return None
    last = samples[-1]
    df = pd.DataFrame(samples)
    mem = _read_denet_mem(run_dir / "denet.jsonl")
    out = {
        "syscall_rate_mean": df["syscall_rate"].mean(),
        "syscall_rate_max":  df["syscall_rate"].max(),
        "io_intensity_mean": df["io_intensity"].mean(),
        "memory_intensity_mean": df["memory_intensity"].mean(),
        "syscalls_total": last["syscalls_total"],
        "offcpu_total_s": last["offcpu_total_ns"] / 1e9,
        "offcpu_avg_ms": last["offcpu_avg_ns"] / 1e6,
        "offcpu_max_ms": last["offcpu_max_ns"] / 1e6,
        "offcpu_events": last["offcpu_events"],
        "wall_s": (df["ts_ms"].max() - df["ts_ms"].min()) / 1000.0,
        "n_samples": len(df),
    }
    by_cat = last["by_category"] or {}
    for k, v in by_cat.items():
        out[f"sc_{k}"] = v
    if not mem.empty:
        out["peak_rss_mb"] = float(mem["rss_mb"].max())
        out["peak_vms_mb"] = float(mem["vms_mb"].max())
        out["peak_procs"] = int(mem["procs"].max())
        out["pf_cached_max"] = int(mem["pf_cached"].max())
        out["pf_disk_max"] = int(mem["pf_disk"].max())
    return out


def aggregate(root: Path):
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _NJOBS_RE.match(run_dir.name)
        if not m:
            continue
        s = _per_run_summary(run_dir)
        if s is None:
            continue
        s["n_jobs"] = int(m.group(1))
        s["run_dir"] = str(run_dir)
        rows.append(s)
    return pd.DataFrame(rows).sort_values("n_jobs").reset_index(drop=True)


def plot_offcpu(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    # Two facets sharing x=n_jobs: total off-CPU time, and per-event avg/max.
    a = df[["n_jobs", "offcpu_total_s"]].copy()
    a["metric"] = "total off-CPU (s)"
    a = a.rename(columns={"offcpu_total_s": "value"})
    b = df[["n_jobs", "offcpu_avg_ms"]].copy()
    b["metric"] = "avg off-CPU per event (ms)"
    b = b.rename(columns={"offcpu_avg_ms": "value"})
    c = df[["n_jobs", "offcpu_max_ms"]].copy()
    c["metric"] = "max off-CPU per event (ms)"
    c = c.rename(columns={"offcpu_max_ms": "value"})
    long = pd.concat([a, b, c], ignore_index=True)

    p = (
        ggplot(long, aes(x="n_jobs", y="value"))
        + geom_line() + geom_point(size=2.5)
        + facet_grid("metric ~ .", scales="free_y")
        + scale_x_log10()
        + labs(x="n_jobs", y="",
               title="Off-CPU time vs n_jobs",
               subtitle="Off-CPU = time blocked on lock / page-fault / I/O. "
                        "Rising = contention.")
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"),
                strip_text_y=element_text(size=8, weight="bold"))
    )
    ggsave(p, str(out), dpi=150, width=7, height=6, units="in")
    print(f"plots: wrote {out}")


def plot_memory(df: pd.DataFrame, out_rss: Path, out_pf: Path):
    if df.empty or "peak_rss_mb" not in df.columns:
        return
    # Peak RSS + per-worker derived line.
    plot_df = df[["n_jobs", "peak_rss_mb"]].copy()
    # fork-COW model: rss ≈ parent + n * per_worker. Show per-worker derived
    # cost as a second series for context.
    parent_rss = float(plot_df.loc[plot_df["n_jobs"] == 1, "peak_rss_mb"].iloc[0]) \
        if (plot_df["n_jobs"] == 1).any() else 0.0
    plot_df["per_worker_mb"] = (plot_df["peak_rss_mb"] - parent_rss) / plot_df["n_jobs"].clip(lower=1)

    a = plot_df[["n_jobs", "peak_rss_mb"]].rename(columns={"peak_rss_mb": "value"})
    a["metric"] = "peak RSS (MB)"
    b = plot_df[["n_jobs", "per_worker_mb"]].rename(columns={"per_worker_mb": "value"})
    b["metric"] = "(peak RSS − parent) / n_jobs (MB/worker)"
    long = pd.concat([a, b], ignore_index=True)

    p = (
        ggplot(long, aes(x="n_jobs", y="value"))
        + geom_line() + geom_point(size=2.5)
        + facet_grid("metric ~ .", scales="free_y")
        + scale_x_log10()
        + labs(x="n_jobs", y="",
               title="Memory footprint vs n_jobs",
               subtitle="Linear RSS growth = fork-COW per-worker cost. "
                        "Flat per-worker line means no leak / no super-linear blow-up.")
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"),
                strip_text_y=element_text(size=8, weight="bold"))
    )
    ggsave(p, str(out_rss), dpi=150, width=7, height=5, units="in")
    print(f"plots: wrote {out_rss}")

    # Page faults: cached (minor, fork-COW driven) vs disk (hard, paging).
    pf = df[["n_jobs", "pf_cached_max", "pf_disk_max"]].copy()
    pf_long = pf.melt(id_vars="n_jobs", var_name="kind", value_name="count")
    pf_long["kind"] = pf_long["kind"].map({
        "pf_cached_max": "minor (cached, fork-COW)",
        "pf_disk_max": "major (disk, real memory pressure)",
    })

    p2 = (
        ggplot(pf_long, aes(x="n_jobs", y="count", color="kind"))
        + geom_line() + geom_point(size=2.5)
        + scale_x_log10()
        + labs(x="n_jobs", y="page faults (cumulative)",
               title="Page faults vs n_jobs",
               subtitle="Minor faults plateau = fork-COW dominates.  "
                        "Major faults ~0 = no real memory pressure.")
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"))
    )
    ggsave(p2, str(out_pf), dpi=150, width=7, height=4.5, units="in")
    print(f"plots: wrote {out_pf}")


def plot_syscall_categories(df: pd.DataFrame, out: Path):
    cat_cols = [c for c in df.columns if c.startswith("sc_")]
    if not cat_cols:
        return
    melted = df.melt(
        id_vars=["n_jobs"], value_vars=cat_cols,
        var_name="category", value_name="count",
    )
    melted["category"] = melted["category"].str.replace("sc_", "", regex=False)
    melted = melted[melted["count"] > 0]
    if melted.empty:
        return

    p = (
        ggplot(melted, aes(x="factor(n_jobs)", y="count", fill="category"))
        + geom_col(position="stack")
        + labs(x="n_jobs", y="total syscalls (cumulative)",
               title="Syscall category breakdown per n_jobs",
               subtitle="memory ~ mmap / page faults.  "
                        "process ~ fork / clone (pool spawn).  "
                        "file_io ~ read / write / openat.")
        + theme_minimal()
        + theme(plot_subtitle=element_text(size=8, color="#666"))
    )
    ggsave(p, str(out), dpi=150, width=8, height=4.5, units="in")
    print(f"plots: wrote {out}")


def _read_obkit(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    return df


def plot_syscall_timeline(root: Path, n_jobs: int, out: Path):
    run_dir = root / f"parallel_n{n_jobs}"
    samples = _read_denet_full(run_dir / "denet.jsonl")
    if not samples:
        print(f"plots: no denet samples for n={n_jobs}")
        return
    s = pd.DataFrame(samples)
    s["t"] = (s["ts_ms"] - s["ts_ms"].min()) / 1000.0

    # Phase spans from the parent obkit log.
    ev_path = run_dir / "obkit-events.jsonl"
    spans_df = pd.DataFrame()
    if ev_path.exists():
        events = _read_obkit(ev_path)
        if not events.empty:
            t0_ms = events["ts"].min().timestamp() * 1000.0
            inner = ("load_h5ad", "pool_setup", "fit_loop",
                     "assemble", "write_csv", "combine_batches")
            spans = []
            for ev, group in events[events["event"].isin(inner)].groupby("event"):
                starts = group[group["phase"] == "start"].sort_values("ts")["ts"].tolist()
                ends = group[group["phase"] == "end"].sort_values("ts")["ts"].tolist()
                for st, en in zip(starts, ends):
                    spans.append({
                        "event": ev,
                        "t_start": (st.timestamp() * 1000.0 - t0_ms) / 1000.0,
                        "t_end":   (en.timestamp() * 1000.0 - t0_ms) / 1000.0,
                    })
            spans_df = pd.DataFrame(spans)

    # Three stacked panels: phases / syscall rate / CPU%.
    panels = []
    if not spans_df.empty:
        event_order = list(dict.fromkeys(spans_df["event"]))
        event_to_y = {ev: i for i, ev in enumerate(event_order)}
        ph = spans_df.copy()
        ph["facet"] = "1_phases"
        ph["y"] = ph["event"].map(event_to_y)
        panels.append(ph)

    rate = s[["t", "syscall_rate"]].rename(columns={"syscall_rate": "y"}).copy()
    rate["facet"] = "2_syscall_rate"
    cpu = s[["t", "cpu"]].rename(columns={"cpu": "y"}).copy()
    cpu["facet"] = "3_CPU%"

    p = ggplot()
    if panels:
        ph = panels[0]
        p = (
            p
            + geom_segment(ph, aes(x="t_start", xend="t_end", y="y", yend="y", color="event"), size=6)
            + geom_text(ph, aes(x="t_start", y="y", label="event"),
                        ha="left", va="bottom", size=8, nudge_y=0.15)
        )
    p = (
        p
        + geom_line(rate, aes(x="t", y="y"), color="darkred")
        + geom_line(cpu,  aes(x="t", y="y"), color="steelblue")
        + facet_grid("facet ~ .", scales="free_y")
        + scale_x_continuous(name="time (s)")
        + scale_y_continuous(name="")
        + labs(title=f"Syscall rate + CPU% over time — n_jobs={n_jobs}")
        + theme_minimal()
        + theme(legend_position="none",
                strip_text_y=element_text(size=9, weight="bold"))
    )
    ggsave(p, str(out), dpi=150, width=10, height=6, units="in")
    print(f"plots: wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeline-n", type=int, default=64,
                    help="n_jobs to use for the syscall/CPU timeline plot.")
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate(root)
    if df.empty:
        print(f"denet-plots: no denet.jsonl rows under {root}")
        return

    summary_csv = root / "denet_ebpf_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"denet-plots: wrote {summary_csv}  ({len(df)} runs)")

    plot_offcpu(df, out_dir / "ebpf_offcpu.png")
    plot_memory(df, out_dir / "memory_footprint.png", out_dir / "page_faults.png")
    plot_syscall_categories(df, out_dir / "ebpf_syscall_categories.png")
    plot_syscall_timeline(root, args.timeline_n,
                          out_dir / f"ebpf_syscall_timeline_n{args.timeline_n}.png")


if __name__ == "__main__":
    main()
