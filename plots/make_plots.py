"""Stage: plots.

plotnine-based figures:
  1. scaling_<phase>.png    wall-time vs n_jobs, log-log, with ideal-linear ref.
  2. speedup_<phase>.png    speedup vs n_jobs, log-log, ideal-linear ref.
  3. phases_<run>.png       Three stacked panels sharing the x-axis (time):
                              top    — Gantt rows for obkit phase spans
                              middle — CPU% (denet)
                              bottom — RSS MB (denet)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from plotnine import (
    ggplot, aes, geom_segment, geom_line, geom_abline, geom_point,
    geom_text,
    facet_grid, scale_x_log10, scale_y_log10, scale_x_continuous,
    scale_y_continuous,
    labs, theme, theme_minimal, element_text, element_blank, ggsave,
)


def _amdahl(n, p):
    """Amdahl's law: S(n) = 1 / ((1-p) + p/n). Monotone, asymptotes at 1/(1-p)."""
    return 1.0 / ((1.0 - p) + p / n)


def _usl(n, alpha, beta):
    """Universal Scalability Law: S(n) = n / (1 + α(n-1) + βn(n-1)).
    α = serial fraction (Amdahl-equivalent).
    β = coherence/coordination cost per added worker — produces a true peak
        and decline past n* = sqrt((1-α)/β), unlike Amdahl.
    """
    return n / (1.0 + alpha * (n - 1) + beta * n * (n - 1))


def _fit_amdahl(n_jobs, speedup):
    if len(n_jobs) < 2:
        return None, None
    try:
        popt, pcov = curve_fit(
            _amdahl, np.asarray(n_jobs, dtype=float),
            np.asarray(speedup, dtype=float),
            p0=[0.9], bounds=(0.0, 1.0),
        )
        return float(popt[0]), float(np.sqrt(np.diag(pcov))[0])
    except Exception:
        return None, None


def _fit_usl(n_jobs, speedup):
    """Returns (alpha, beta, n_star, s_star) or (None, None, None, None)."""
    if len(n_jobs) < 3:
        return None, None, None, None
    try:
        (alpha, beta), _ = curve_fit(
            _usl, np.asarray(n_jobs, dtype=float),
            np.asarray(speedup, dtype=float),
            p0=[0.05, 0.001], bounds=([0.0, 0.0], [1.0, 1.0]),
        )
        if beta > 0:
            n_star = float(np.sqrt((1.0 - alpha) / beta))
            s_star = float(_usl(n_star, alpha, beta))
        else:
            n_star, s_star = float("inf"), 1.0 / alpha if alpha else float("inf")
        return float(alpha), float(beta), n_star, s_star
    except Exception:
        return None, None, None, None


# -------------------------------------------------------------------- scaling

def plot_scaling(scaling_csv: Path, out_dir: Path):
    df = pd.read_csv(scaling_csv)
    df = df[df["phase"].isin(("ga_gauss", "ga_poisson_gauss"))].copy()
    if df.empty:
        print("plots: no scaling data")
        return
    df["n_jobs"] = df["n_jobs"].astype(int)

    p_wall = (
        ggplot(df, aes(x="n_jobs", y="wall_seconds", color="phase"))
        + geom_line() + geom_point()
        + scale_x_log10() + scale_y_log10()
        + labs(x="n_jobs", y="wall time (s)", title="Scaling")
        + theme_minimal()
    )
    ggsave(p_wall, str(out_dir / "scaling_curve.png"), dpi=150,
           width=6, height=4, units="in")
    print(f"plots: wrote {out_dir / 'scaling_curve.png'}")

    df_sp = df.dropna(subset=["speedup"])
    if not df_sp.empty:
        # Fit USL (and Amdahl for comparison) per phase. USL captures the
        # observed peak-and-decline shape that Amdahl can't reproduce.
        n_max = max(df_sp["n_jobs"].max() * 2, 128)
        n_grid = np.unique(np.geomspace(1.0, n_max, 80))
        usl_rows, amdahl_rows = [], []
        title_bits = []
        for phase, sub in df_sp.groupby("phase"):
            sub = sub.sort_values("n_jobs")
            n_arr = sub["n_jobs"].values.astype(float)
            s_arr = sub["speedup"].values
            alpha, beta, n_star, s_star = _fit_usl(n_arr, s_arr)
            p, p_err = _fit_amdahl(n_arr, s_arr)
            if alpha is not None:
                for n in n_grid:
                    usl_rows.append({"phase": phase, "n_jobs": n, "speedup": _usl(n, alpha, beta)})
                title_bits.append(
                    f"{phase}: USL α={alpha*100:.2f}%  β={beta:.5f}  "
                    f"peak n*={n_star:.0f} S*={s_star:.1f}×"
                )
            if p is not None:
                for n in n_grid:
                    amdahl_rows.append({"phase": phase, "n_jobs": n, "speedup": _amdahl(n, p)})
        fit_df = pd.DataFrame(usl_rows)
        amdahl_df = pd.DataFrame(amdahl_rows)
        df = df_sp
        title = "Speedup vs baseline\n" + "  ·  ".join(title_bits) if title_bits else "Speedup vs baseline"

        p_su = (
            ggplot()
            + geom_abline(slope=1, intercept=0, linetype="dashed", color="grey")
        )
        if not amdahl_df.empty:
            p_su = p_su + geom_line(amdahl_df, aes(x="n_jobs", y="speedup", color="phase"),
                                    linetype="dotted", alpha=0.5)
        if not fit_df.empty:
            p_su = p_su + geom_line(fit_df, aes(x="n_jobs", y="speedup", color="phase"),
                                    linetype="solid", alpha=0.7)
        p_su = (
            p_su
            + geom_point(df, aes(x="n_jobs", y="speedup", color="phase"), size=2.5)
            + scale_x_log10() + scale_y_log10()
            + labs(x="n_jobs", y="speedup",
                   title=title,
                   subtitle="Points = measured.  Solid = USL fit.  Dotted = Amdahl fit (for comparison).  Dashed grey = ideal linear.")
            + theme_minimal()
            + theme(plot_subtitle=element_text(size=8, color="#666"))
        )
        ggsave(p_su, str(out_dir / "speedup_curve.png"), dpi=150,
               width=7, height=4.5, units="in")
        print(f"plots: wrote {out_dir / 'speedup_curve.png'}")


# --------------------------------------------------------------- per-run plot

def _read_obkit(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    return df


def _read_denet(path: Path) -> pd.DataFrame:
    """Read denet samples, preferring the `aggregated` block (parent + all
    children) over `parent` alone. For multi-process runs the parent does
    very little; workers are children and only contribute to `aggregated`.
    """
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
            block = rec.get("aggregated") or rec.get("parent")
            if block is None:
                continue
            rows.append({
                "ts_ms": rec["ts_ms"],
                "cpu": block.get("cpu_usage"),
                "rss_mb": (block.get("mem_rss_kb") or 0) / 1024.0,
            })
    return pd.DataFrame(rows)


def _phase_spans(events: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    """One row per (event, occurrence): start/end in seconds since t0_ms."""
    spans = []
    for ev_name, group in events.groupby("event"):
        starts = group[group["phase"] == "start"].sort_values("ts")["ts"].tolist()
        ends = group[group["phase"] == "end"].sort_values("ts")["ts"].tolist()
        for s, e in zip(starts, ends):
            spans.append({
                "event": ev_name,
                "t_start": (s.timestamp() * 1000.0 - t0_ms) / 1000.0,
                "t_end":   (e.timestamp() * 1000.0 - t0_ms) / 1000.0,
            })
    return pd.DataFrame(spans)


def plot_phase_timeline(run_dir: Path, out_dir: Path):
    ev_path = run_dir / "obkit-events.jsonl"
    if not ev_path.exists():
        return
    events = _read_obkit(ev_path)
    if events.empty:
        return
    t0_ms = events["ts"].min().timestamp() * 1000.0

    spans = _phase_spans(events, t0_ms)
    if spans.empty:
        return

    denet_path = run_dir / "denet.jsonl"
    samples = _read_denet(denet_path) if denet_path.exists() else pd.DataFrame()
    if not samples.empty:
        samples["t"] = (samples["ts_ms"] - t0_ms) / 1000.0

    # Three stacked panels (facet rows): "1_phases", "2_CPU%", "3_RSS_MB".
    # Numeric y everywhere so facet_grid(scales='free_y') is happy. The
    # phases panel encodes the event name as both an integer y-position
    # and a text label drawn at the segment start.
    event_order = list(dict.fromkeys(spans["event"]))  # stable first-seen order
    event_to_y = {ev: i for i, ev in enumerate(event_order)}

    span_df = spans.copy()
    span_df["facet"] = "1_phases"
    span_df["y"] = span_df["event"].map(event_to_y)

    cpu_df = pd.DataFrame()
    rss_df = pd.DataFrame()
    if not samples.empty:
        cpu_df = samples[["t", "cpu"]].rename(columns={"cpu": "y"}).copy()
        cpu_df["facet"] = "2_CPU%"
        rss_df = samples[["t", "rss_mb"]].rename(columns={"rss_mb": "y"}).copy()
        rss_df["facet"] = "3_RSS_MB"

    x_max = max(
        [span_df["t_end"].max()] +
        ([cpu_df["t"].max()] if not cpu_df.empty else [])
    )
    x_min = 0.0

    p = (
        ggplot()
        + geom_segment(
            span_df,
            aes(x="t_start", xend="t_end", y="y", yend="y", color="event"),
            size=6,
        )
        + geom_text(
            span_df,
            aes(x="t_start", y="y", label="event"),
            ha="left", va="bottom", size=8, nudge_y=0.15,
        )
    )
    if not cpu_df.empty:
        p = p + geom_line(cpu_df, aes(x="t", y="y"), color="steelblue")
    if not rss_df.empty:
        p = p + geom_line(rss_df, aes(x="t", y="y"), color="darkorange")

    p = (
        p
        + facet_grid("facet ~ .", scales="free_y")
        + scale_x_continuous(limits=(x_min, x_max), name="time (s)")
        + scale_y_continuous(name="")
        + labs(title=run_dir.name)
        + theme_minimal()
        + theme(
            legend_position="none",
            strip_text_y=element_text(size=9, weight="bold"),
        )
    )

    out_path = out_dir / f"phases_{run_dir.name}.png"
    ggsave(p, str(out_path), dpi=150, width=10, height=6, units="in")
    print(f"plots: wrote {out_path}")


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_csv = root / "scaling.csv"
    if scaling_csv.exists():
        plot_scaling(scaling_csv, out_dir)
    else:
        print(f"plots: missing {scaling_csv} (run evaluation/scaling.py first)")

    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir() and (run_dir / "obkit-events.jsonl").exists():
            plot_phase_timeline(run_dir, out_dir)


if __name__ == "__main__":
    main()
