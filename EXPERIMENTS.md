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

### Follow-up: E1c — NUMA pinning sanity check

**Goal.** Establish whether the unexplained `fit_one` slowdown at high
n_jobs is NUMA-driven by constraining all workers + the parent to a
single socket via `os.sched_setaffinity` in `_init_worker`.

**Three levers, increasing in specificity:**

1. `numactl --cpunodebind=0 --membind=0 -- pixi run scripts/run_e1_instr_sweep.sh`
   — coarse. Wraps everything; can't selectively bind workers.
2. **`os.sched_setaffinity` inside `_init_worker`** — per-worker. Cleanest:
   each worker pins itself at spawn time to a CPU set defined by an env
   var (e.g. `CRISPAT_WORKER_CPUS=0-63` to confine to socket 0).
3. `psutil.Process().cpu_affinity()` — cross-platform equivalent of (2).

Recommended: **(2)**. Add to the throwaway `parallel-gauss-instr` branch
since this is profiling-only:

```python
def _init_worker(adata, obkit_log_dir=None):
    cpus = os.environ.get("CRISPAT_WORKER_CPUS")
    if cpus:
        os.sched_setaffinity(0, _parse_cpu_list(cpus))
    ...
```

**Sweep design:** run n_jobs ∈ {16, 32, 64} twice each — once unpinned,
once with `CRISPAT_WORKER_CPUS=0-63` (one socket, assuming 64 cores per
socket; check `lscpu | grep -i 'core(s)\|socket'`). Compare median
`fit_one` between the two. Hypotheses:

- Median drops at n=64 pinned vs unpinned → NUMA-driven, propose
  socket-aware worker spawning as a future optimisation.
- Median unchanged → not NUMA. Next candidate is BLAS allocator
  contention (try `MALLOC_ARENA_MAX=1`) or the kernel's TLB-shootdown
  overhead from frequent COW.

**Output:** one extra figure `fit_one_pinning_compare.png` — same plot
as `fit_one_distribution.png` but with `pinned ∈ {yes, no}` as a colour
facet.

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
