# E2 — Hybrid axis sweep: results

**Question.** On a fixed core budget, is per-gRNA process parallelism the
right axis for crispat, or does mixing in BLAS threads help?

**Setup.** Bundled Schraivogel dataset (86 gRNAs in batch1), `ga_gauss`
only, n_iter=250, single 16-core laptop. 8-core budget split four ways.
The single-axis comparison line uses the same patch with
`OMP_NUM_THREADS=1` and `n_jobs ∈ {1, 2, 4, 8}`.

## Numbers

| (n_jobs, omp) | Wall (s) | Speedup vs serial |
|---|---|---|
| (1, 1)  baseline       | 122.1 | 1.00× |
| (1, 8)  pure BLAS      | 100.1 | 1.22× |
| (2, 4)  hybrid         |  54.0 | 2.26× |
| (4, 2)  hybrid         |  32.1 | 3.80× |
| (8, 1)  pure processes |  21.2 | **5.75×** |

For each step from "more BLAS threads" to "more processes" — at the same
total core budget — wall time drops:

- (1,8) → (2,4): 1.85× faster
- (2,4) → (4,2): 1.68× faster
- (4,2) → (8,1): 1.51× faster

## Headline finding

**Process-over-gRNA is decisively the right axis.** Eight BLAS threads on a
single-process serial outer loop give a 1.22× speedup; eight independent
worker processes with BLAS pinned to one thread each give **5.75×**, and
correctness is preserved exactly (assignments.csv is byte-identical to the
serial baseline at all hybrid configurations — see
[`evaluation/correctness.py`](evaluation/correctness.py)).

This matches the prediction in [`EXPERIMENTS.md`](EXPERIMENTS.md): the
per-gRNA SVI fit operates on small 1-D arrays (length n_cells, ~10⁴),
where BLAS threading has too little useful work to amortize coordination
cost. Coarse-grained per-gRNA process parallelism is the right granule.

## Why this matters for the upstream PR

The crispat parallelization patch exposes only a single tuning knob —
`n_jobs`. This experiment defends that choice:

- A `blas_threads` parameter would add API surface for ~no benefit.
- Users who set `OMP_NUM_THREADS` higher *expecting* speedup will get
  almost none, and on multi-process runs they will create
  oversubscription contention with their `n_jobs`.
- The PR's run scripts therefore set `OMP_NUM_THREADS=1` (and friends)
  before the imports — single-axis is the intended operating mode.

## Figure

![E2 hybrid sweep](results/plots/e2_hybrid.png)

`results/plots/e2_hybrid.png` — speedup vs n_jobs, log-log, with the
ideal-linear line dashed. Two lines:

- **single-axis (omp=1)** — n_jobs ∈ {1, 2, 4, 8, 16} with `OMP_NUM_THREADS=1`.
  Tracks linear scaling closely until the laptop's physical-core ceiling
  (~8 cores) dominates around n_jobs=8.
- **hybrid (8-core budget)** — (1,8), (2,4), (4,2), (8,1). The (8,1)
  endpoint matches the single-axis line at n_jobs=8. The other points
  fall well below the single-axis line, confirming that any OMP > 1 on
  this workload trades useful work for coordination overhead.

Each point is labeled `(n_jobs, omp_threads)`.

## The memory cost — `e2_memory.png`

Process parallelism wins on speed, but each added **process** costs RAM.
Threads, by contrast, share memory. The hybrid (n_jobs=1, omp=8) point
makes the trade explicit:

| Config | Peak RSS (MB) | Notes |
|---|---|---|
| baseline   (1, 1) — 1 proc, 1 thread  | 1054 | reference |
| hybrid_n1_omp8 (1, 8) — 1 proc, **8 threads** |  868 | ≈ baseline; threads share memory |
| hybrid_n2_omp4 (2, 4) — 2 procs                | 1990 | |
| hybrid_n4_omp2 (4, 2) — 4 procs                | 3162 | |
| hybrid_n8_omp1 (8, 1) — 8 procs                | 5183 | |
| parallel_n16   (16, 1) — 16 procs              | 9909 | 9.4× baseline |

Memory grows roughly with **process count**, not core count. `(1,8)` uses
*less* RAM than the serial baseline (within noise) — because threads
share the address space. Going from 1 to 16 processes inflates RAM by
~9.4×; the same 16-fold core utilization via threads would inflate RAM
by ~1×.

### So why don't we just thread the per-gRNA loop?

Two reasons, both load-bearing:

1. **pyro's global state forces processes.** `pyro.clear_param_store()`
   and the module-level `global global_guide, svi` in `initialize()`
   would race under threading. Threading the per-gRNA loop is not
   available without a substantially larger upstream refactor.
2. **The GIL would erase most of the parallelism anyway.** SVI fits in
   crispat are predominantly Python-level glue around small tensor ops.
   CPU-bound Python under the GIL doesn't scale across threads.

So the trade is **forced** for this code: you only have processes, and
processes cost RAM linearly. Linux's fork-COW gives a small discount
over the pessimistic `n × adata_size` upper bound, but not much —
CPython's refcount-on-read dirties shared pages as soon as workers
touch the AnnData, and per-worker private state (pyro, torch tensors,
NumPy temporaries, matplotlib figures) accumulates linearly regardless.

This panel is **why the BPCells / scx-via-picklerick note in BUGS.md
matters more than I/O efficiency would suggest.** The headline scaling
plot shows you can buy speed with cores; this panel shows you also buy
RAM, and the threading escape hatch is closed. On the bundled
Schraivogel dataset that's fine — even n=16 fits in 10 GB. On a real
screen (50k cells × 5k gRNAs, multi-GB AnnData), the same pattern hits
the per-node memory ceiling well before the speedup ceiling. An
mmap-backed Rust-owned matrix that doesn't go through CPython
refcounts decouples the two axes: keep using processes for speed
(forced choice) but stop paying for them in RAM (because the matrix
data is genuinely shared across processes via the kernel page cache).

![E2 memory cost](results/plots/e2_memory.png)

## Caveats for the slides version

1. **Laptop, not cluster.** 16 logical / 8 physical cores. The (8,1) point
   is near the physical-core ceiling; cluster runs (E1) on 100 cores
   should let the single-axis line keep climbing. The hybrid result
   should remain qualitatively the same — the BLAS threads aren't
   suddenly going to find useful work — but the *gap* between hybrid and
   single-axis will widen.
2. **One method, one dataset.** `ga_gauss` only on Schraivogel. The
   workload shape (small per-step ops, many independent gRNAs) is the
   same in `ga_poisson_gauss` and most other crispat methods, so the
   conclusion should generalize, but a more exhaustive figure and study should
   include `ga_poisson_gauss` for symmetry.
3. **The serial baseline carries a small measurement bias** — it ran
   concurrently with one parallel sweep run, so the baseline wall is
   ~5–15% inflated. All speedup numbers in this table inherit that bias.
   Re-running the baseline alone for the a figure for the notebook should
   be cheap.

## Reproduce

From `crispat-profiling/`:

```bash
for combo in "8 1" "4 2" "2 4" "1 8"; do
  read n omp <<< "$combo"
  out="results/hybrid_n${n}_omp${omp}"
  rm -rf "$out" && mkdir -p "$out"
  OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp OPENBLAS_NUM_THREADS=$omp \
    pixi run -- denet --json --out "$out/denet.jsonl" \
      run python methods/parallel/run.py -- \
        --in results/data/input.h5ad --out "$out" \
        --method gauss --n-jobs $n
done
pixi run python evaluation/scaling.py --results results
pixi run python plots/make_e2_plot.py --results results
```
