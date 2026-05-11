"""Stage: plots — E1c pinned vs unpinned fit_one comparison.

Reads results-e1c/n32_unpinned/obkit-events.worker.*.jsonl and
results-e1c/n32_pinned/obkit-events.worker.*.jsonl, pairs fit_one
start/end events per worker, and produces:

  fit_one_pinning_compare.png — distribution overlay (median + p5–p95 +
  min/max) per condition, side by side.

  summary printed: count, min, p5, median, mean, p95, max per condition;
  and per-condition run wall (from parent obkit-events.jsonl).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_jitter, geom_point, geom_linerange, geom_violin,
    labs, theme, theme_minimal, element_text, ggsave, position_dodge,
    scale_y_continuous,
)


def _fit_one_durations(run_dir: Path) -> pd.DataFrame:
    """Return DataFrame of (pid, gRNA, duration_s) for every fit_one span
    across all worker JSONLs in run_dir."""
    rows = []
    for wp in sorted(run_dir.glob("obkit-events.worker.*.jsonl")):
        pid = int(wp.name.split(".")[-2])
        stack_by_key = {}
        with wp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "fit_one":
                    continue
                attrs = rec.get("attrs") or {}
                key = attrs.get("gRNA")
                ts = pd.to_datetime(rec["ts"], format="ISO8601", utc=True)
                if rec["phase"] == "start":
                    stack_by_key.setdefault(key, []).append(ts)
                elif rec["phase"] == "end":
                    stk = stack_by_key.get(key) or []
                    if stk:
                        start = stk.pop()
                        rows.append({"pid": pid, "gRNA": key,
                                     "duration_s": (ts - start).total_seconds()})
    return pd.DataFrame(rows)


def _run_wall_s(run_dir: Path) -> float | None:
    """Read parent obkit-events.jsonl and return (last end - first start) in s."""
    p = run_dir / "obkit-events.jsonl"
    if not p.exists():
        return None
    ts_list = []
    with p.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_list.append(pd.to_datetime(rec["ts"], format="ISO8601", utc=True))
    if not ts_list:
        return None
    return (max(ts_list) - min(ts_list)).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results-e1c")
    ap.add_argument("--unpinned-dir", default="n32_unpinned")
    ap.add_argument("--pinned-dir",   default="n32_pinned")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = [
        ("unpinned", root / args.unpinned_dir),
        ("pinned",   root / args.pinned_dir),
    ]

    dfs = []
    summary_rows = []
    for tag, d in conds:
        if not d.exists():
            print(f"E1c: missing run dir {d}")
            continue
        df = _fit_one_durations(d)
        if df.empty:
            print(f"E1c: no fit_one spans found in {d}")
            continue
        df["condition"] = tag
        dfs.append(df)
        wall = _run_wall_s(d)
        s = df["duration_s"]
        summary_rows.append({
            "condition": tag,
            "run_wall_s": wall,
            "count": len(s),
            "min": s.min(),
            "p05": s.quantile(0.05),
            "median": s.median(),
            "mean": s.mean(),
            "p95": s.quantile(0.95),
            "max": s.max(),
        })

    if not dfs:
        print("E1c: no data, nothing to plot")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["condition"] = pd.Categorical(
        all_df["condition"], categories=["unpinned", "pinned"], ordered=True,
    )

    # Quantile annotations per condition.
    qs = (all_df.groupby("condition")["duration_s"]
          .agg(min="min",
               p05=lambda s: s.quantile(0.05),
               median="median",
               p95=lambda s: s.quantile(0.95),
               max="max")
          .reset_index())

    p = (
        ggplot()
        + geom_violin(all_df, aes(x="condition", y="duration_s", fill="condition"),
                      alpha=0.35, width=0.8, scale="area")
        + geom_jitter(all_df, aes(x="condition", y="duration_s", color="condition"),
                      width=0.12, height=0, alpha=0.4, size=1.3)
        + geom_linerange(qs, aes(x="condition", ymin="min", ymax="max"),
                         color="#666666", size=0.8)
        + geom_linerange(qs, aes(x="condition", ymin="p05", ymax="p95"),
                         color="black", size=2.2)
        + geom_point(qs, aes(x="condition", y="median"),
                     color="black", fill="white", size=3.5, stroke=1.2)
        + labs(x="", y="per-gRNA fit_one duration (s)",
               title="E1c — fit_one distribution: pinned vs unpinned (n_jobs=32)",
               subtitle="Pinned = workers confined to NUMA node 0 (cpus 0-15,64-79).  "
                        "Jitter dots = individual fits.  Black bar = p5–p95.  White point = median.")
        + theme_minimal()
        + theme(legend_position="none",
                plot_subtitle=element_text(size=8, color="#666"))
    )
    out = out_dir / "fit_one_pinning_compare.png"
    ggsave(p, str(out), dpi=150, width=7, height=5, units="in")
    print(f"plots: wrote {out}")

    if summary_rows:
        s = pd.DataFrame(summary_rows).round(3)
        print()
        print(s.to_string(index=False))

        # Quick verdict line.
        if len(s) == 2:
            med_un = s.loc[s["condition"] == "unpinned", "median"].iloc[0]
            med_pn = s.loc[s["condition"] == "pinned",   "median"].iloc[0]
            delta = (med_un - med_pn) / med_un * 100.0
            verdict = "no effect" if abs(delta) < 5 else (
                "pinning helped" if delta > 0 else "pinning hurt"
            )
            print(f"\nmedian fit_one: unpinned={med_un:.3f}s  pinned={med_pn:.3f}s  "
                  f"({delta:+.1f}% → {verdict})")


if __name__ == "__main__":
    main()
