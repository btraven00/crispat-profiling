# crispat-profiling

Profiling and parallelization exercise for [crispat](https://github.com/velten-group/crispat),
structured as an omnibenchmark workflow with denet (sampling profiler) and obkit
(phase-event anchors).

## Goal

Demonstrate that the per-gRNA fit loop in `ga_gauss` and `ga_poisson_gauss` is
embarrassingly parallel, and quantify the speedup on a real dataset (bundled
Schraivogel example) across `n_jobs ∈ {1, 2, 4, 8, 16, 32, 64, 100}`.

The eventual upstream patch to crispat is intentionally **minimal**: a new
`n_jobs` parameter (default 1, backwards compatible) wrapping the existing
gRNA loop with `concurrent.futures.ProcessPoolExecutor`. **No new dependencies.**
All profiling/benchmarking scaffolding lives in this directory.

## Layout

```
crispat-profiling/
├── pixi.toml                 env: python, pyro, scanpy, denet, obkit, crispat (path)
├── benchmark.yaml            omnibenchmark DAG
├── data/prepare.py           bundled Schraivogel → input.h5ad
├── methods/
│   ├── baseline/run.py       fork with n_jobs=1 (gated patch ⇒ upstream-equivalent)
│   └── parallel/run.py       fork with --n-jobs > 1
├── crispat-fork/             git worktree of ../crispat on branch parallel-gauss
├── evaluation/
│   ├── correctness.py        assert assignments identical across runs
│   └── scaling.py            timings → speedup/efficiency table
├── plots/make_plots.py       scaling curve + phase-attributed CPU/RSS
└── results/                  gitignored outputs
```

## Phase anchors (obkit)

Every `run.py` emits the same anchors so denet samples can be attributed to phases:

- `load_h5ad`
- `batch_<id>` (outer)
- `fit_gRNAs` (the hot span — should shrink ~linearly with n_jobs)
- `write_outputs`

## Running

**Convention: each invocation gets its own `--out` subdirectory.** Both the
obkit events file and denet samples live there. The run wrappers truncate
`obkit-events.jsonl` on entry, so a rerun into the same dir is fine, but
distinct experiments (baseline vs parallel-n4 vs parallel-n16 …) belong in
distinct directories so `evaluation/scaling.py` can iterate them cleanly.

```
results/
├── data/input.h5ad
├── baseline/                  # n_jobs=1, full
├── parallel_n2/, parallel_n4/, parallel_n8/, ...
└── hybrid/n16_omp4/, n8_omp8/, ...
```

Per-stage commands:

```bash
# 1. Prepare input
pixi run python data/prepare.py --out results/data/input.h5ad

# 2. Baseline (single core, BLAS pinned to 1 thread inside the script)
pixi run -- denet --json --out results/baseline/denet.jsonl \
    run python methods/baseline/run.py \
        --in results/data/input.h5ad --out results/baseline

# 3. Parallel (per n_jobs value, separate out dir)
pixi run -- denet --json --out results/parallel_n4/denet.jsonl \
    run python methods/parallel/run.py \
        --in results/data/input.h5ad --out results/parallel_n4 --n-jobs 4

# 4. Evaluate + plot
pixi run python evaluation/correctness.py --results results
pixi run python evaluation/scaling.py    --results results
pixi run python plots/make_plots.py      --results results
```

Or (eventually) the whole DAG:
```bash
pixi run ob -- run benchmark.yaml --cores N
```
