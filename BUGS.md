# Upstream crispat bugs encountered during profiling

These are pre-existing bugs in upstream crispat (as of pinned commit `8d96f10`),
discovered while setting up the profiling workflow. They are NOT part of the
parallelization patch we're preparing. Each should be filed/fixed separately
upstream so the parallelization PR stays minimal and focused.

---

## BUG-1: `max(2D ndarray)` fails on numpy ≥ 2.0

**Status:** worked around by pinning `numpy<2` in `pixi.toml`.

**Files:**
- `crispat/gauss.py:225`
- `crispat/poisson_gauss.py:255`

**Pattern:**
```python
X = np.arange(1, max(selected_guide.toarray()) + 1, 1)
```

`selected_guide.toarray()` returns a 2D `(n_cells, 1)` ndarray. Python's
built-in `max()` over a 2D array iterates rows (each a 1-element array), and
adding `1` to that produces an array. `np.arange` then can't convert it to a
scalar.

**Symptom on numpy 2.4.3:**
```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

**Fix (one-line, in both files):**
```python
X = np.arange(1, int(selected_guide.toarray().max()) + 1, 1)
```

**Root cause:** `setup.py` declares `numpy>=1.24.3` with no upper bound, but
the codebase predates numpy 2.x scalar-conversion semantics. Either the call
site needs the fix above, or `setup.py` needs `numpy<2`.

---

## BUG-2 (suspected): plotting in the per-gRNA hot loop

**Status:** not confirmed yet, just noted.

`fit_GMM` (gauss.py) and the equivalent in `poisson_gauss.py` save a loss
plot and a fitted-model plot per gRNA, inside the inner loop. For a
many-gRNA dataset this can be a meaningful fraction of total wall time, and
serves no purpose in batch/automated runs. Worth measuring once denet
samples land cleanly. If material, an opt-out flag (`make_plots=True` for
backwards compat) is the minimal upstream change.

---

## BUG-4: gratuitous `time.sleep(0.01)` inside the per-gRNA loop

**Status:** noticed during scaling analysis. Fix is one line.

**File:** `crispat/poisson_gauss.py:315`

```python
for gRNA in tqdm(gRNA_list):
    time.sleep(0.01)        # <-- this
    perturbed_cells, … = fit_PGMM(…)
```

86 gRNAs × 10 ms = ~0.9 s of dead time per call to `ga_poisson_gauss`. That
becomes a meaningful fraction of total wall time once the actual fits are
parallelized (run wall drops below ~20 s). The sleep is presumably
left over from `tqdm`-display debugging and serves no functional purpose.

**Fix:** delete the line.

**Note:** the sleep is in the *parent* loop and so cannot be parallelized
away — it's a pure serial floor.

---

## NOTE: AnnData/h5ad as a serial bottleneck

Not in scope for the parallelization PR, but worth recording. The serial
floor measured by Amdahl's law (~10% in early laptop sweeps; needs cluster
confirmation) is dominated by:

1. `sc.read_h5ad(input_file)` in the parent.
2. AnnData being passed to workers (pickle in spawn-mode, COW-shared in
   fork-mode — Linux defaults to fork so this is cheap, but still
   irreducibly parent-side at startup).
3. Final aggregation + CSV writes in the parent.

Two directions worth exploring **after** the parallelization patch is
merged, if speedup ceilings on the cluster turn out to be a real concern:

- **picklerick** (Python bindings via maturin to the **scx** Rust core) —
  lets workers attach to a shared on-disk single-cell experiment
  representation without paying the full AnnData deserialization cost
  per worker. The parent never has to fully materialize the AnnData;
  workers index in lazily.
- **BPCells** — sparse single-cell matrices stored as memory-mapped
  bitpacked blocks. Random column access is O(1)-ish and fits process
  parallelism perfectly: each worker mmaps the same file and reads only
  the gRNA column it needs. No pickle, no fork-COW, no read_h5ad in the
  parent at all.

### Why this is the *real* motivation, not just I/O

CPython's refcount-on-read defeats fork-COW: every time a worker reads
a Python object's data pages, refcount header writes dirty the page,
which the kernel COWs into a per-worker copy. So even though
`ProcessPoolExecutor` on Linux forks (and starts with all pages shared
with the parent), the AnnData footprint **converges towards
n_workers × adata_size as workers actually do work.** This is the cost
that scales with `n_jobs` on the laptop today and becomes binding on
big screens (50k cells × 5k gRNAs).

scx-via-picklerick fixes this on the **read** side: PyO3-wrapped Rust
objects are opaque `PyCapsule` pointers — touching them doesn't write
to any CPython refcount page, so the underlying Rust-owned buffer
stays shared across workers. Combined with an mmap-backed file format,
the kernel's page cache shares the same physical pages across all
worker processes for free. Memory cost goes from `n_workers × M` down
to roughly `1 × M` + tiny per-worker overhead.

The benefit only materializes if workers stay on the Rust side: any
`.toarray()`-style materialization back to NumPy reintroduces a
per-worker Python-heap allocation. A real integration would need
crispat to consume scx column views directly — which is also a much
larger change than the n_jobs patch.

Both options would additionally make the package usable on screens
that don't fit in RAM, which is a separate but related win. Either
one belongs in its own design discussion upstream.

---

## BUG-3 (note, not really a bug): BLAS/OMP thread bleed

The "single-threaded" baseline run hits 260–700% CPU on this machine,
because pyro/torch use BLAS/OMP threads for matrix ops inside SVI. For a
fair `n_jobs` comparison we should pin `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` (and possibly `OPENBLAS_NUM_THREADS=1`) in the run
scripts so each worker really is one core. Not an upstream bug, but a
benchmarking pitfall to document.
