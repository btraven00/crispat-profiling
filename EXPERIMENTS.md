# Experiments

## E1 — Single-axis scaling (main DAG)

Sweep `n_jobs ∈ {1, 2, 4, 8, 16, 32, 64, 100}` with BLAS/OMP threads pinned to 1.
Driven by `benchmark.yaml`. Produces the headline scaling curve.

**Hypothesis:** near-linear speedup until `min(n_gRNAs, n_cores)`, then flat.

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
