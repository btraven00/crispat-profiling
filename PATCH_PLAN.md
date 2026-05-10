# Parallelization patch — design

Target: `crispat/gauss.py::ga_gauss` and `crispat/poisson_gauss.py::ga_poisson_gauss`.

## Constraints (recap)

- Backwards-compatible API (`n_jobs=1` default).
- `n_jobs == 1` code path **identical** to upstream — gated, not refactored.
- **No new dependencies.** `concurrent.futures.ProcessPoolExecutor` from stdlib.
- Determinism: assignments byte-identical to baseline regardless of `n_jobs`.
- Diff must be small enough to review trivially.

## Why processes (not threads)

`crispat/gauss.py::initialize()` writes to pyro's global param store
(`pyro.clear_param_store()`) and assigns `global global_guide, svi`. Threads
would race on these. Each subprocess has its own pyro state — safe.

## Worker pattern: initializer + per-gRNA submit

The hot loop currently does:

```python
for gRNA in gRNA_list:
    perturbed_cells, threshold, loss, map_estimates = fit_GMM(gRNA, adata_crispr_batch, …)
    …
```

`fit_GMM` already slices `adata_crispr[:, [gRNA]].X` internally. Pickling the
whole AnnData object **per task** would dominate cost. Instead, pickle it
**once per worker** via `ProcessPoolExecutor`'s `initializer`/`initargs`:

```python
# Module-private worker globals
_WORKER_ADATA = None

def _init_worker(adata):
    global _WORKER_ADATA
    _WORKER_ADATA = adata

def _fit_one(gRNA, output_dir, seed, n_iter, nonzero):
    # Calls the existing, unmodified fit_GMM.
    return gRNA, fit_GMM(gRNA, _WORKER_ADATA, output_dir, seed, n_iter, nonzero)
```

This means **`fit_GMM` itself is not modified**. The only change inside the
existing functions is the loop dispatch.

## Loop dispatch shape

```python
if n_jobs == 1:
    # EXISTING LOOP, UNCHANGED — preserves baseline equivalence.
    for gRNA in gRNA_list:
        perturbed_cells, threshold, loss, map_estimates = fit_GMM(...)
        …
else:
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=(adata_crispr_batch,),
    ) as ex:
        futures = {
            ex.submit(_fit_one, gRNA, output_dir+'batch'+str(batch)+'/', 2024, n_iter, nonzero): gRNA
            for gRNA in gRNA_list
        }
        results = {}
        for fut in as_completed(futures):
            gRNA, payload = fut.result()
            results[gRNA] = payload

    # Reassemble in original gRNA_list order — guarantees byte-identical
    # assignments.csv vs serial path.
    for gRNA in gRNA_list:
        perturbed_cells, threshold, loss, map_estimates = results[gRNA]
        …  # same per-gRNA aggregation code as the serial loop
```

The aggregation code (the part that does `pd.concat` etc.) is shared between
the two branches. Exact factoring TBD when writing — preference for
duplicating five lines over introducing a helper, if that keeps the diff
smaller.

## Output ordering — why it matters

`pd.concat([...])` in the serial loop produces a specific row order in
`assignments.csv`. The byte-equivalence claim depends on reproducing that
order. By collecting `results[gRNA]` and re-iterating `gRNA_list` in
original order, we get the same concat order ⇒ same CSV.

## Plotting in workers

`fit_GMM` calls `plot_loss` and `plot_fitted_model`, writing PNGs to disk.
Each gRNA writes its own filename, so no contention. matplotlib `Agg` backend
is set in our run.py wrapper before the worker pool spawns — children
inherit the env. **No code change needed** for plotting to work in workers.

## EM inference path

`ga_gauss` supports `inference="em"` via `fit_em`. Same loop shape —
parallelize identically. `_fit_one` dispatches on inference.

## ga_poisson_gauss

Looking at `crispat/poisson_gauss.py`, the structure is parallel: a per-gRNA
loop calling `fit_PoissonGaussianMixture` (or similar). Same patch shape:
add `n_jobs`, gate, dispatch via executor with adata in initargs. Verify the
loop shape matches before writing.

## Diff budget

Estimated patch size:
- `gauss.py`: ~40 lines added (worker helpers, dispatch branch, `n_jobs`
  parameter + docstring), 0 lines removed.
- `poisson_gauss.py`: same shape, ~40 lines added.
- New imports: `from concurrent.futures import ProcessPoolExecutor, as_completed` (1 line per file).

If the diff exceeds ~120 lines total, something is wrong — stop and revisit.

## Validation steps (before claiming done)

1. **Code review the diff:** `git diff main..parallel-gauss -- crispat/`. Must
   be small, additive, gated.
2. **Smoke run** `methods/parallel/run.py --n-jobs 4` — completes without errors.
3. **Correctness:** `evaluation/correctness.py` passes — assignments.csv from
   parallel run is byte-identical to baseline.
4. **Repeat correctness** for n_jobs ∈ {2, 8, 16}. If any drift, the bug is
   either ordering or RNG, not the worker pattern.
5. **Quick scaling spot-check:** run n_jobs=4 and verify wall time
   ≈ baseline / 4 for the `ga_gauss` span (within ~20% — Amdahl floor includes
   pyro init in each worker, AnnData pickle, etc.).

Only after all five pass do we move to the full sweep (E1) and write up.

## Out of scope for this patch (keep focused)

- BUG-1 (numpy 2 incompat) — separate upstream fix.
- BUG-2 (`make_plots` flag) — separate upstream proposal.
- Hybrid OMP/n_jobs tuning — users' env, not our patch surface.
- Refactoring `fit_GMM` — the diff stays small precisely because we don't
  touch it.
