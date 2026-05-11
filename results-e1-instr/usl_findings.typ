// E1b — USL fits + empirical phase attribution.
// Compile: `typst compile usl_findings.typ` (produces usl_findings.pdf)

#set page(margin: 1.5cm, width: auto, height: auto)
#set text(font: "New Computer Modern", size: 10pt)
#show heading: set text(weight: "bold")

= Crispat parallelization — USL fits and empirical phase attribution

Schraivogel dataset, batch1 (172 gRNAs total: 86 gRNAs $times$ 2 methods),
omnibenchmark node (128 logical cores), `OMP_NUM_THREADS=1`, `--no-plots`
enabled, instrumented branch `parallel-gauss-instr`.

== USL fits (#emph[S(n) = n / (1 + α(n-1) + β · n(n-1))])

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*Method*], [*α (serial)*], [*β (coherence)*],
    [*n\**], [*S\* (USL)*],
    [*Observed peak*], [*Δα vs E1*], [*ΔS\* vs E1*],
  ),
  [`ga_gauss`],          [0.68 %], [2.20 × 10#super[−4]], [66.6], [27.5 ×], [26.31 × @ n=100], [−43 %], [+24 %],
  [`ga_poisson_gauss`],  [1.36 %], [3.20 × 10#super[−4]], [55.3], [20.4 ×], [19.87 × @ n=64],  [+37 %], [−6 %],
  [#emph[run (combined)]], [0.93 %], [2.60 × 10#super[−4]], [61.8], [24.3 ×], [23.51 × @ n=64], [—], [—],
)

#emph[E1 baseline (with per-gRNA matplotlib plots): ga_gauss α=1.20 %, β=2.80×10#super[−4], S\*=22.2×.]

== Empirical phase attribution at n = 64

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, left),
  stroke: 0.5pt,
  table.header(
    [*Phase*], [*Wall (s)*], [*Share of run*], [*USL role*],
  ),
  [`load_h5ad`],       [0.04],  [0.3 %], [α (true serial)],
  [`pool_setup`],      [1.74],  [11.8 %], [#strong[β] — linear in n_jobs],
  [`fit_loop`],        [11.86], [80.6 %], [parallelisable],
  [`assemble`],        [0.32],  [2.2 %], [α (true serial)],
  [`write_csv`],       [0.09],  [0.6 %], [α (true serial)],
  [`combine_batches`], [0.06],  [0.4 %], [α (true serial)],
  table.hline(),
  [#strong[Total non-fit (serial floor)]], [#strong[2.25]], [#strong[15.3 %]], [α + β],
  [#strong[Run total]], [#strong[14.70]], [100 %], [],
)

#emph[The USL fit predicted α = 0.93 % for the combined run. Sum of
non-fit_loop phases at n=64 is 2.25 s / T#sub[1] (345.5 s) = 0.65 %; at
n=100 it is 3.34 s / 345.5 s = 0.97 %. The model's α is the empirical
serial-floor fraction.]

== `pool_setup` is β

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*n_jobs*], [2], [4], [8], [16], [32], [64], [100],
  ),
  [`pool_setup` (s)],     [0.07], [0.14], [0.34], [0.50], [0.88], [1.74], [2.84],
  [per-worker (ms)],      [35],   [35],   [43],   [31],   [27],   [27],   [28],
  [share of run total],   [0.05%],[0.20%],[0.76%],[1.67%],[5.63%],[11.84%],[18.61%],
)

#emph[Linear in n_jobs at ~ 28 ms per worker spawn (fork + COW dirtying +
pyro/torch import). At n=100 this single phase is 19 % of total wall.
Concretely the β term made visible.]

== Memory & contention checks (denet eBPF)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*n_jobs*], [*Peak RSS (GB)*], [*Per-worker (MB)*],
    [*Off-CPU total (s)*], [*Major page faults*],
  ),
  [1],   [0.6],  [—],    [125.8], [4],
  [8],   [4.1],  [437],  [147.4], [0],
  [32],  [14.9], [445],  [48.2],  [8],
  [64],  [29.1], [445],  [41.9],  [26],
  [100], [43.7], [431],  [40.0],  [91],
)

#emph[Findings:]
- #emph[Per-worker RSS flat at ~440 MB (no leak; pure fork-COW cost).]
- #emph[Off-CPU time #strong[decreases] with n_jobs — workers are CPU-bound at scale, not blocked. Rules out lock contention.]
- #emph[Major page faults stay under 100 over the whole run on a 256 GB node. No memory pressure.]
- #emph[Minor page faults plateau at ~6 M for n ≥ 32: fork-COW fault budget fully amortised.]

== Unexplained: per-gRNA `fit_one` wall grows with n_jobs

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*n_jobs*], [2], [4], [8], [16], [32], [64], [100],
  ),
  [median fit_one (s)], [1.73], [1.70], [1.82], [2.41], [2.46], [3.62], [4.70],
  [p95 fit_one (s)],    [1.97], [2.04], [2.88], [5.36], [3.63], [5.30], [6.09],
)

#emph[Per-gRNA work is identical, yet wall doubles+. Off-CPU rules out
blocking; page-fault plateau rules out fault traffic; CPU saturation
is near n_jobs × 100 %. Strongest remaining hypothesis: NUMA cross-socket
traffic (workers reading parent-resident pages from the other socket).
Test queued as E1c — pin all workers to one socket via `os.sched_setaffinity`
in `_init_worker` and re-measure.]
