"""Stage: method.baseline.

Runs upstream crispat (pinned) ga_gauss and ga_poisson_gauss single-threaded.
Wraps each phase with obkit anchors for joining with denet samples.

Run under denet:  denet -- python methods/baseline/run.py --in <h5ad> --out <dir>
"""
import argparse
import os
from pathlib import Path

# Force matplotlib to a non-interactive backend before any crispat import that
# may transitively import pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

# Pin BLAS/OMP threads to 1 so each Python process really is single-core.
# Without these, torch/pyro use BLAS threads inside SVI and the "n_jobs=1"
# baseline silently runs at hundreds of %CPU, making the parallel speedup
# look worse than it is. MUST be set before importing torch/pyro/numpy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from obkit.logger import init_logger, emit

# Baseline = the fork with n_jobs=1. The patch is gated so this code path is
# byte-equivalent to upstream 8d96f10 (verified by reviewing the diff).
import crispat as cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["gauss", "poisson_gauss", "both"], default="both")
    ap.add_argument("--n-iter", type=int, default=250)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Truncate any prior events file so reruns don't stack (init_logger opens append).
    (out / "obkit-events.jsonl").unlink(missing_ok=True)
    init_logger(str(out))

    emit("run", "start", {"method": args.method, "n_jobs": 1, "variant": "baseline"})

    if args.method in ("gauss", "both"):
        emit("ga_gauss", "start")
        cr.ga_gauss(
            input_file=args.input,
            output_dir=str(out / "gauss") + "/",
            n_iter=args.n_iter,
        )
        emit("ga_gauss", "end")

    if args.method in ("poisson_gauss", "both"):
        emit("ga_poisson_gauss", "start")
        cr.ga_poisson_gauss(
            input_file=args.input,
            output_dir=str(out / "poisson_gauss") + "/",
            n_iter=args.n_iter,
        )
        emit("ga_poisson_gauss", "end")

    emit("run", "end")


if __name__ == "__main__":
    main()
