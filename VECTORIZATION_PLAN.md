# Plan: vectorize the per-gRNA fit loop

## Background

Three orthogonal optimization axes on the per-gRNA Gaussian mixture fit:

1. **Coordination** — parallelize across gRNAs (`ProcessPoolExecutor`).
   Done. Branch `parallel-gauss`. Peak ~26× speedup at n_jobs=64.
2. **Per-fit cost** — reduce wasted iterations and JIT the SVI step.
   Done. Branch `fit-optimizations` (this commit). Expected ~1.5–3×
   on top, additive to (1).
3. **Vectorization** — process *many gRNAs in one pyro model* so torch
   batched ops amortize Python/torch overhead. This is the lever this
   plan targets.

## Why vectorization is the lever

The current per-step cost of `svi.step(data)` is dominated by:

- Python interpreter loop overhead (call into pyro, register
  parameters, build the trace).
- torch tensor allocation/teardown per step.
- BLAS overhead on tiny (~10k) 1D vectors.

None of these scale meaningfully with vector length until you hit a
few hundred K elements. Right now we fit ~86 gRNAs serially in the
worker, each with its own pyro state, each running 250 (or with early
stop, ~50–100) SVI steps. **All of that bookkeeping repeats per
gRNA.** Vectorizing means: one pyro model, one SVI step, *N gRNAs
processed simultaneously* with the same Python and torch overhead.

Conservative expectation: 5–10× speedup per worker, which combined
with parallelism puts the patch at 100–200× over upstream serial.

## Target shape

### Current per-gRNA model (`gauss.py`)

```python
@config_enumerate
def model(data):                                     # data: shape (n_cells,)
    weights = pyro.sample("weights", dist.Dirichlet(torch.tensor([0.99, 0.01])))
    with pyro.plate("components", 2):
        locs = pyro.sample("locs", dist.Normal(1.0, 2.0))   # shape (2,)
    scale = pyro.sample("scales", dist.LogNormal(-3.0, 2.0))
    with pyro.plate("data", len(data)):
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        pyro.sample("obs", dist.Normal(locs[assignment], scale), obs=data)
```

### Vectorized model

```python
@config_enumerate
def model(data, mask):                       # data: (n_grnas, n_cells_max)
                                             # mask: (n_grnas, n_cells_max) bool — for ragged inputs
    with pyro.plate("grnas", n_grnas):
        weights = pyro.sample("weights", dist.Dirichlet(torch.tensor([0.99, 0.01])))
        with pyro.plate("components", 2):
            locs = pyro.sample("locs", dist.Normal(1.0, 2.0))   # shape (2, n_grnas)
        scale = pyro.sample("scales", dist.LogNormal(-3.0, 2.0))   # shape (n_grnas,)
        with pyro.plate("data", n_cells_max):
            assignment = pyro.sample("assignment", dist.Categorical(weights))
            # Mask out padded positions so they don't contribute to ELBO.
            with poutine.mask(mask=mask):
                pyro.sample("obs", dist.Normal(
                    locs.gather(0, assignment.unsqueeze(0)).squeeze(0), scale
                ), obs=data)
```

Key changes:

- Outermost `pyro.plate("grnas", n_grnas)` makes every parameter
  per-gRNA. Pyro vectorizes everything within.
- Cell-counts are **ragged** across gRNAs (number of non-zero cells
  varies). Two ways:
  - **Padding + mask** (above). Simple, allows uniform tensor shape.
    `poutine.mask` zeros the ELBO contribution from padded positions.
  - **Bucket gRNAs by cell-count and run one SVI per bucket.** Avoids
    padding cost; more complex dispatch logic.

Recommended: padding + mask. n_cells is ~10k–100k regardless; the
padding overhead is dwarfed by the savings.

### How threshold extraction changes

`fit_GMM` currently extracts MAP estimates and computes a per-gRNA
threshold inside the Python loop. In the vectorized version:

- MAP estimates come out as per-gRNA tensors of shape `(n_grnas,)` or
  `(n_grnas, 2)`. One `global_guide()` call gets them all.
- Threshold computation (`prob_normal_component` + `df.loc[...].min()`)
  can be a single tensor op per gRNA on the same batched MAP outputs,
  or a small Python loop over n_grnas — cheap either way since it's
  post-SVI.

## Implementation steps

1. **Branch off `fit-optimizations`** as `vectorize-gauss`. Keep all
   prior optimizations in place. This is a stack:
   `parallel-gauss → fit-optimizations → vectorize-gauss`.

2. **Refactor data prep**: build `(n_grnas, n_cells_max)` tensor +
   mask in one pass. Currently `fit_GMM` does `adata[:, [gRNA]].X` per
   gRNA — needs to become one slice + reshape outside the loop.

3. **Write the vectorized `model_batched` and `guide_batched`** as
   above. Test on synthetic data first to verify pyro plate semantics
   work as expected.

4. **Replace the per-gRNA `fit_GMM` loop with one batched
   `fit_GMM_batched` call** that returns dicts keyed by gRNA. The
   public `ga_gauss` signature is unchanged — internally it just
   dispatches to the batched implementation.

5. **Backward-compatibility flag**: `vectorize: bool = True` on
   `ga_gauss`. Default True (the new code path), but expose the old
   per-gRNA path for debugging and byte-identical comparison.

6. **Correctness verification**: byte-identical assignments.csv vs
   the serial path within numerical tolerance. The threshold is the
   only quantity that affects assignment, and it's a function of the
   MAP estimates. If MAP estimates match per-gRNA between batched and
   per-gRNA fits, assignments will match.

   Expect *not* byte-identical because:
   - Different initialization order across gRNAs may change which
     `min((init(seed), seed) for seed in range(10))` wins.
   - Float ordering in batched ops differs from per-gRNA accumulation.

   Tolerance: assignments must match for ≥99% of (cell, gRNA) pairs.
   The remaining ~1% are gRNAs where the threshold is right on the
   boundary of an integer count.

7. **n_jobs interaction**: with vectorization, the per-worker cost
   is much higher (~all gRNAs of a batch at once). Parallelism
   should drop accordingly. Best operating point may be n_jobs=4–8,
   not n_jobs=64.

## Risks / open questions

- **JIT compatibility with the outer plate**. Need to verify pyro's
  JIT handles `plate("grnas", n_grnas)` with a `poutine.mask`. May
  need `jit=False` for the vectorized path initially.
- **The 10-initialization seed search**. Hard to vectorize — each
  seed produces a different parameter init. Options:
  - Run 10 batched inits, pick best per gRNA via argmin along seed
    axis (10× memory, 10× compute, but same SVI step cost).
  - Reduce to 3–5 inits (we suggested this in the unstructured-ideas
    note; here it becomes nearly free).
- **`fit_PGMM` (Poisson-Gaussian) vectorization** is harder because
  of the custom `cont_Poisson` distribution and `MaskedMixture`.
  Worth doing second after `fit_GMM` is proven out.
- **Memory ceiling at large n_cells × large n_grnas**: padded tensor
  is `(n_grnas, n_cells_max) × 4 bytes`. For a 200k-cell × 200-gRNA
  dataset that's ~160 MB per copy. Pyro keeps several copies during
  SVI. Should be fine but worth benchmarking on the biggest expected
  dataset.

## Validation plan

1. **Microbenchmark** `fit_GMM` (per-gRNA) vs `fit_GMM_batched`
   (vectorized) on a synthetic dataset with N_gRNAs ∈ {1, 10, 100}.
   Wall time per gRNA should drop steeply with N_gRNAs.
2. **Correctness**: re-run E1d sweep on the Schraivogel dataset
   with `vectorize=True` and `vectorize=False`. Compare
   assignments.csv directly. Expect ≥99% agreement.
3. **Scaling**: full sweep n_jobs ∈ {1, 2, 4, 8, 16, 32, 64} on the
   vectorized path. Fit USL. Expected α likely unchanged (still
   `pool_setup`-dominated at high n); β may *increase* because each
   worker now does more work and serialization of the batched output
   tensor across processes is heavier.
4. **End-to-end**: target total wall reduction of ≥3× at the
   single-worker level, ≥2× at the n_jobs=8 level vs
   `fit-optimizations` baseline.

## Out of scope (future work)

- **GPU dispatch**. With proper vectorization, putting the batched
  tensor on CUDA is one line. For the Schraivogel-scale dataset
  (~10k cells × ~100 gRNAs) it's probably not worth it (PCIe
  transfer cost dominates), but for huge perturb-seq datasets
  (~1M cells × ~1000 gRNAs) GPU vectorized fits would be game-changing.
- **EM fast-path**. The existing `fit_em` (sklearn-based) is already
  ~100× per-fit. A vectorized SVI may make EM unnecessary, but EM
  remains the cheapest option if the user doesn't need variational
  posteriors.
- **Subsampling**. For very large n_cells, fit on a random subsample
  and only use full data for the final threshold computation. Could
  push another 5-10× on huge datasets.

## Upstream story

Each branch is a self-contained PR:

1. `parallel-gauss` — the original parallelization. Small,
   reviewable, byte-identical output at n_jobs=1.
2. `fit-optimizations` — early stopping + JIT. Even smaller. Pure
   speed win, no semantic change.
3. `vectorize-gauss` — the big one. Real refactor, real speedup.
   Sell as the headline performance improvement.

The crispat maintainers can merge (1) and (2) immediately for the
quick win, then review (3) carefully as it's the substantive change.
