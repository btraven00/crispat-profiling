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
