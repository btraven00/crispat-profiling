# Experiments

## E1 — Single-axis scaling (main DAG)

Sweep `n_jobs ∈ {1, 2, 4, 8, 16, 32, 64, 100}` with BLAS/OMP threads pinned to 1.
Driven by `benchmark.yaml`. Produces the headline scaling curve.

**Hypothesis:** near-linear speedup until `min(n_gRNAs, n_cores)`, then flat.

---

## E1b — Phase-resolved instrumented re-run (extra experiment)

**Question.** Where does the ~1.2% USL serial fraction (α from E1)
actually live? Is it `sc.read_h5ad`, `ProcessPoolExecutor` spawn, parent-side
`pd.concat` assembly, `to_csv` writes — or the per-gRNA straggler tail
inside `fit_loop`?

**Why a separate branch.** The instrumentation patch (obkit phase anchors
inside `crispat/gauss.py` and `crispat/poisson_gauss.py`) is throwaway —
not part of the upstream PR, which must stay minimal. It lives on
`parallel-gauss-instr` (commit `7820fcb`) branched off `parallel-gauss`
in `crispat-fork`.

**What it emits.** All events go through obkit; parent writes
`obkit-events.jsonl`, each worker writes its own
`obkit-events.worker.<pid>.jsonl` (no concurrent-append contention).

Parent phases (disjoint, contiguous):
- `load_h5ad` — `sc.read_h5ad` call
- `pool_setup` — `ProcessPoolExecutor` ctor + futures submission
- `fit_loop` — waiting on `fut.result()` (i.e. parallel work in flight)
- `assemble` — parent-side `pd.concat` / per-gRNA UMI lookup
- `write_csv` — `.to_csv` calls
- `combine_batches` — final cross-batch concat (gauss only)

Worker events:
- `worker_init` end — one per spawn (tells us when each worker came online)
- `fit_one` start/end — one pair per (worker, gRNA) with the gRNA in attrs

These are **orthogonal** to denet samples: denet measures CPU%/RSS of the
whole process tree at ~1 Hz, obkit anchors mark which code is running.
They're combined at plot time, not double-counted.

**Setup on the cluster.**

1. **Push the instrumented branch.** From local crispat-fork:
   ```bash
   git -C crispat-fork push btraven parallel-gauss-instr
   ```
2. **Point pixi at it.** Edit `pixi.toml`, swap the crispat branch:
   ```toml
   crispat = { git = "...", branch = "parallel-gauss-instr" }
   ```
   Then on the cluster: `git pull && pixi update crispat`.
3. **Run the sweep:**
   ```bash
   scripts/run_e1_instr_sweep.sh
   ```
   Writes to `results-e1-instr/parallel_n${n}/` so the headline
   `results-e1/` from E1 is untouched. Uses `--no-plots` to strip the
   matplotlib β contribution (per-gRNA loss/fitted-model PNGs).
4. **After the run, flip pixi.toml back** to `parallel-gauss` before any
   E1/E2 work resumes — the instrumented branch is not what we upstream.

**Analysis (next step, see ROADMAP).** Aggregate the per-worker JSONLs to
produce, per `n_jobs`:

- Sum of parent phase wall times → empirical "serial floor" (load_h5ad +
  pool_setup + assemble + write_csv). Compare to USL α·T₁.
- max(per-worker active time) vs `fit_loop` wall → straggler-tail share
  of β.
- p50 / p95 of `fit_one` duration → gRNA work distribution.
- sum(`fit_one` durations) / `fit_loop` wall → effective parallelism vs
  `n_jobs`.

### Results (run: 2026-05-11, omnibenchmark, `--no-plots` enabled)

**Phase attribution of α.** The USL-fitted α for the full run is **0.93%**.
The sum of non-fit_loop parent phases (`load_h5ad + pool_setup + assemble +
write_csv + combine_batches`) at n=64 is **2.25 s**, which is **0.65% of
T₁=345.5 s**. At n=100 it's 3.34 s (0.97%). The 1% serial fraction the USL
fit identified is *exactly these phases* — empirically confirmed.

**β = `pool_setup` growth.** `pool_setup` is linear in n_jobs:
0.07 s (n=2) → 0.50 s (n=16) → 1.74 s (n=64) → 2.84 s (n=100). At n=100
this single phase is **19% of total wall**. That linear-in-n cost is the
USL β term made concrete: ~28 ms per worker spawn (fork + COW dirtying
+ pyro/torch import in the child).

**New USL fits, `--no-plots`:**

| Method | α | β | n\* | S\* | observed peak |
|---|---|---|---|---|---|
| ga_gauss | **0.68%** | 0.000220 | 66.6 | **27.5×** | 26.31× @ n=100 |
| ga_poisson_gauss | 1.36% | 0.000320 | 55.3 | 20.4× | 19.87× @ n=64 |
| run (combined) | **0.93%** | 0.000260 | 61.8 | **24.3×** | 23.51× @ n=64 |

Comparing to the E1 fits with plots enabled (α=1.20%, β=0.000280,
S\*=22.2× for ga_gauss): `--no-plots` cut α by **43%** and pushed the
predicted peak from 22× → 27.5×. For `ga_poisson_gauss` the gain is
negligible because the inherent per-SVI-step cost of the `cont_Poisson`
distribution (custom `torch.log(prob)` evaluations) is what dominates,
not the matplotlib overhead. The `time.sleep(0.01)` per gRNA (BUG-4)
only sits in the `n_jobs == 1` branch — removing it would lower T₁ by
~1.7 s and slightly *reduce* measured speedup, not raise the peak.

**Surprises from eBPF + memory data:**

- Off-CPU time *decreases* with n_jobs (workers are CPU-bound, not blocked).
- No memory pressure (major page faults < 100 across the whole sweep on
  a 256 GB node; peak RSS 44 GB at n=100, linear at ~440 MB/worker).
- Minor page faults plateau at ~6 M for n≥32 — fork-COW fault budget
  fully amortized.
- Median per-gRNA `fit_one` wall *grows* from 1.7 s (n=2) to 4.7 s
  (n=100) despite no blocking and no memory pressure. **Unexplained by
  this run.** Strongest remaining hypothesis: NUMA cross-socket traffic
  (workers reading pages physically resident on the other socket).

### E1c — NUMA pinning sanity check (run: 2026-05-11)

**Goal.** Establish whether the unexplained `fit_one` slowdown at high
n_jobs is NUMA-driven by constraining workers to a single NUMA node.

**Topology found.** omnibenchmark is an AMD EPYC 7742 — single socket,
64 physical / 128 logical cores, but **4 NUMA nodes** of 16 physical
cores each (Zen 2 chiplet layout). Node distances 10 local / 12 remote.
Roland is an EPYC 7763 with the same topology (Zen 3 microarch only).
So "cross-socket" framing was wrong — this is cross-NUMA-die. One
NUMA node = 32 logical cores, perfectly fits the n=32 test.

**Three conditions at n=32, ga_gauss only:**

1. `n32_unpinned` — baseline, no affinity constraints.
2. `n32_pinned` — workers-only via `CRISPAT_WORKER_CPUS=0-15,64-79`
   (`os.sched_setaffinity` inside `_init_worker`).
3. `n32_taskset` — whole-process via `taskset -c 0-15,64-79` (parent +
   workers + AnnData allocation all on node 0).

**Implementation.** Added `_parse_cpu_list` helper and CPU-set env-var
handling in `_init_worker` on the `parallel-gauss-instr` branch
(commit `f196eb9`). Throwaway only — not for upstream.

**Results (workers-only pinning vs unpinned, omni mid-load):**

| metric | unpinned | pinned (workers) | Δ |
|---|---|---|---|
| run wall | 25.65 s | **22.67 s** | **−12 %** |
| median `fit_one` | 4.98 s | 6.29 s | +26 % |
| mean `fit_one` | 8.10 s | **6.98 s** | **−14 %** |
| p95 `fit_one` | 14.02 s | **9.41 s** | **−33 %** |
| max `fit_one` | 14.06 s | **9.52 s** | **−32 %** |

**Reading.** The unpinned distribution is **bimodal**: ~95 % of fits at
4–5 s, ~5 % at 14 s. That signature is NUMA roulette — workers land on
random nodes, lucky ones run fast, unlucky ones pay remote-fetch
latency. Pinning **collapses the bimodality**: everyone runs at ~6 s
with no tail past 9.5 s. Median goes up because the lucky-cohort
disappears, but wall drops 12 % because **the tail no longer bounds
the run**. The verdict line in `make_e1c_compare_plot.py` was updated
to weight wall time, not median.

**Whole-process taskset (third condition).** Pending — running. The
hypothesis: workers-only pinning leaves AnnData scattered across the
nodes where the parent happened to be (first-touch allocation runs
*before* `_init_worker`). With `taskset` on the parent, AnnData lands
on node 0 too, workers all access local memory, and the median should
drop toward the unpinned fast-cohort's 4–5 s while keeping the narrow
tail.

**Conclusion for the patch story.** NUMA effects are real and
quantifiable but not a blocker. The crispat parallelization patch ships
as-is; pinning is a deployment-layer concern (taskset/Slurm/cgroups,
not part of the upstream PR). The instrumentation branch demonstrates
the lever exists if someone wants it.

### E1d — Full pinned sweep (run: 2026-05-11, in progress)

**Goal.** Re-run the headline scaling sweep with whole-process pinning,
to produce the final wrap-up numbers for the writeup.

**Setup.**
- `taskset -c 0-31,64-95` (NUMA nodes 0+1, 32 physical / 64 logical
  cores) on the entire `pixi run …` invocation.
- `--no-plots` (per E1b finding: removes ~40 % of α for ga_gauss).
- n_jobs ∈ {1, 2, 4, 8, 16, 32, 64} — past 64 we'd over-subscribe the
  two-node cpuset; 64 is the post-knee region anyway.
- Output dir: `results-e1d-pinned/`.
- Same instrumentation as E1b → full obkit phase events + per-worker
  fit_one + denet eBPF + memory.

**Script.** `scripts/run_e1d_pinned_sweep.sh`.

**Expectation.** Pinning should:
- Cut `fit_loop` wall by 10–15 % at n=32, 64 (from E1c evidence).
- Reduce the p95 / max of per-worker `fit_one` (collapse the bimodal
  unpinned distribution).
- USL α likely unchanged (it's the serial-floor fraction, not
  parallelism quality); USL β should decrease (cleaner scaling, less
  coherence cost).
- Headline peak speedup ≥ 23.5× (E1b unpinned, `run` total), aiming
  for ≥ 26× given the E1c wall-time gain.

**Next.** Once data lands: regenerate the speedup-curve plot with the
new USL fit, compare side by side with the E1b unpinned fit, build a
wrap-up slide (`results-e1d-pinned/summary.typ` → PDF for presentation).

---

## E2 — Hybrid-axis sweep on fixed core budget (extra experiment)

**Question:** is process parallelism over gRNAs really the right axis,
or does a hybrid (fewer processes × more BLAS threads each) win?

**Why this is non-obvious:** the per-gRNA SVI fit is small (vector ops on
1D arrays of length n_cells, feeding a 2-component mixture — no big
matmuls). At that size BLAS multi-threading typically saturates at 2–4
threads and can hurt at higher counts due to spawn/coordination overhead.
The per-gRNA granule (seconds) is well-suited to coarse process
parallelism. We expect (n_jobs, omp) = (N, 1) to win, but we measure to
prove it.

**Design:** fixed total core budget = 64 (cluster node). Sweep:

| n_jobs | omp_threads | total cores |
|--------|-------------|-------------|
| 64     | 1           | 64          |
| 32     | 2           | 64          |
| 16     | 4           | 64          |
| 8      | 8           | 64          |
| 4      | 16          | 64          |
| 2      | 32          | 64          |
| 1      | 64          | 64          |

`ga_gauss` only (poisson_gauss should follow the same shape; running both
doubles cost without changing the conclusion).

**Why this is NOT in `benchmark.yaml`:** the main DAG is the story we
upstream; `n_jobs` is the only knob the patch exposes. OMP threads are an
environment variable users already control. This sweep is supporting
evidence that the patch chose the right axis, not part of the patch.

**Implementation sketch (post-patch, ad-hoc):**

```bash
# scripts/run_hybrid_sweep.sh
for combo in "64 1" "32 2" "16 4" "8 8" "4 16" "2 32" "1 64"; do
    read n_jobs omp <<< "$combo"
    out="results/hybrid/n${n_jobs}_omp${omp}"
    mkdir -p "$out"
    OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp OPENBLAS_NUM_THREADS=$omp \
        pixi run -- denet --json --out "$out/denet.jsonl" \
        run python methods/parallel/run.py \
            --in results/data/input.h5ad --out "$out" \
            --method gauss --n-jobs $n_jobs
done
```

Plot: speedup vs `n_jobs` (x-axis), one line per `omp_threads` value.
Knee location is the finding.

**Outcomes:**
- If (64,1) wins clearly → confirms the simple narrative. Patch is right.
- If a hybrid (e.g. (16,4)) wins → still doesn't change the patch
  (users set OMP themselves), but is worth noting in the PR description
  and any blog post.
