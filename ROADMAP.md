# Roadmap & status

Snapshot for resuming work. See `README.md` for layout, `EXPERIMENTS.md` for
the planned sweeps, `BUGS.md` for upstream issues found in passing.

## Where we are

### Done

- **Scaffold** — directory layout, pixi env, omnibenchmark `benchmark.yaml`,
  per-stage scripts (`data/prepare.py`, `methods/{baseline,parallel}/run.py`,
  `evaluation/{correctness,scaling}.py`, `plots/make_plots.py`).
- **Crispat fork** — `crispat-fork/` is a git worktree of `../crispat` on
  branch `parallel-gauss`, currently unchanged from upstream `8d96f10`.
- **Single crispat install** — fork only; "baseline" = fork with `n_jobs=1`,
  byte-equivalent to upstream because the (eventual) patch will be gated.
- **obkit on PyPI** — `obkit>=0.0.3` from PyPI, not editable path.
- **Numpy pinned `<2`** to work around `BUG-1` in upstream crispat.
- **BLAS/OMP threads pinned to 1** in both `run.py` files (set in
  `os.environ` before torch import). Without this, "single-threaded"
  baseline runs at hundreds of % CPU and compresses speedup numbers.
- **obkit log truncation** — both run.py files unlink
  `obkit-events.jsonl` before `init_logger()` so reruns into the same out
  dir don't stack events.
- **`make_plots.py` denet schema** — uses `ts_ms` + `parent.cpu_usage` /
  `parent.mem_rss_kb` (denet's actual JSONL fields, not what the first
  draft assumed). Phase timeline plot now overlays CPU%/RSS on obkit phase
  spans.
- **Smoke baseline** completed once on bundled Schraivogel (86 gRNAs).
  Headline numbers: `ga_gauss` ≈ 119s, `ga_poisson_gauss` ≈ 103s on one
  core with `n_iter=250`.

### Not done

- **Clean baseline rerun** after the obkit-truncation + plots fixes
  (the existing `results/baseline/` has stacked obkit events from earlier
  buggy runs and the scaling table is wrong as a result).
- **The parallelization patch** itself. `crispat-fork` is still upstream.
- **Correctness check** — has nothing to compare yet.
- **Scaling sweep (E1)** — needs the patch first.
- **Hybrid sweep (E2)** — needs the patch first.
- **Cluster runs** — local box has 8–16 useful cores; main numbers must
  come from the 100-core node.

## Resume here

1. **Clean baseline rerun.** From `crispat-profiling/`:
   ```bash
   rm -rf results/baseline
   pixi run -- denet --json --out results/baseline/denet.jsonl \
       run python methods/baseline/run.py \
           --in results/data/input.h5ad --out results/baseline
   pixi run python evaluation/scaling.py --results results
   pixi run python plots/make_plots.py  --results results
   cat results/scaling.csv
   ```
   Sanity check: `ga_gauss` and `ga_poisson_gauss` rows with realistic wall
   times (~120s and ~100s on one core), CPU near 100% in
   `results/plots/phases_baseline.png`.

2. **Apply the parallelization patch** on the `crispat-fork` worktree
   (branch `parallel-gauss`). Edit `crispat/gauss.py` and
   `crispat/poisson_gauss.py`:
   - Add `n_jobs: int = 1` (default preserves backwards compat).
   - When `n_jobs == 1`: existing code path **unchanged** so baseline ≡
     upstream is auditable from the diff.
   - When `n_jobs > 1`: dispatch the per-gRNA loop via
     `concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)`. Each
     worker calls the existing `fit_GMM` / equivalent on one gRNA;
     parent assembles results in the original gRNA order so output is
     deterministic.
   - **No new dependencies.** Stdlib only.
   - Keep the diff small; this is the upstream PR.
   - Worker subprocesses inherit `OMP_NUM_THREADS=1` etc. from
     `methods/parallel/run.py`'s env — no changes needed in the patch.

3. **Smoke parallel run.**
   ```bash
   pixi run -- denet --json --out results/parallel_n4/denet.jsonl \
       run python methods/parallel/run.py \
           --in results/data/input.h5ad --out results/parallel_n4 --n-jobs 4
   pixi run python evaluation/correctness.py --results results
   ```
   Correctness must pass byte-for-byte vs baseline. If not, diagnose
   before sweeping.

4. **Scaling sweep (E1).** On the 100-core node:
   ```bash
   for n in 1 2 4 8 16 32 64 100; do
     pixi run -- denet --json --out results/parallel_n${n}/denet.jsonl \
       run python methods/parallel/run.py \
         --in results/data/input.h5ad --out results/parallel_n${n} --n-jobs ${n}
   done
   pixi run python evaluation/correctness.py --results results
   pixi run python evaluation/scaling.py    --results results
   pixi run python plots/make_plots.py      --results results
   ```
   Expected ceiling ~86× (n_gRNAs in batch1), realistic 30–50× after
   Amdahl on `load_h5ad` and `write_outputs`.

5. **Hybrid sweep (E2).** See `EXPERIMENTS.md` — fixed 64-core budget,
   `(n_jobs, omp) ∈ {(64,1)…(1,64)}`, `ga_gauss` only.

6. **Write up & PR prep.** Once numbers are in:
   - `git diff main..parallel-gauss` should be small and clean.
   - Open the PR with the scaling figure as motivation.
   - File `BUG-1` (numpy 2.x) as a separate upstream issue/PR.

## Key constraints (don't drift from these)

- Backwards-compatible API (`n_jobs=1` default, gated path unchanged).
- **No new dependencies** in the patch — stdlib `concurrent.futures` only.
- Profiling/benchmarking scaffolding stays in `crispat-profiling/`,
  never bundled into the upstream PR.
- Upstream bugs (see `BUGS.md`) get separate, also-minimal fixes.
