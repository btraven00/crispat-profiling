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

## BUG-3 (note, not really a bug): BLAS/OMP thread bleed

The "single-threaded" baseline run hits 260–700% CPU on this machine,
because pyro/torch use BLAS/OMP threads for matrix ops inside SVI. For a
fair `n_jobs` comparison we should pin `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` (and possibly `OPENBLAS_NUM_THREADS=1`) in the run
scripts so each worker really is one core. Not an upstream bug, but a
benchmarking pitfall to document.
