"""Stage: method.parallel.

Runs the patched crispat fork (../crispat-fork on branch parallel-gauss) with
the new n_jobs parameter. Same obkit phase anchors as baseline so the runs
can be compared directly.

Run under denet:
  denet -- python methods/parallel/run.py --in <h5ad> --out <dir> --n-jobs 16
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Pin BLAS/OMP threads to 1 — see methods/baseline/run.py for rationale.
# These propagate to ProcessPoolExecutor workers via os.environ inheritance.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from obkit.logger import init_logger, emit

# Patched fork (path-installed in pixi env)
import crispat as cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["gauss", "poisson_gauss", "both"], default="both")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-iter", type=int, default=250)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "obkit-events.jsonl").unlink(missing_ok=True)
    init_logger(str(out))

    emit("run", "start", {"method": args.method, "n_jobs": args.n_jobs, "variant": "parallel"})

    if args.method in ("gauss", "both"):
        emit("ga_gauss", "start", {"n_jobs": args.n_jobs})
        cr.ga_gauss(
            input_file=args.input,
            output_dir=str(out / "gauss") + "/",
            n_iter=args.n_iter,
            n_jobs=args.n_jobs,
        )
        emit("ga_gauss", "end")

    if args.method in ("poisson_gauss", "both"):
        emit("ga_poisson_gauss", "start", {"n_jobs": args.n_jobs})
        cr.ga_poisson_gauss(
            input_file=args.input,
            output_dir=str(out / "poisson_gauss") + "/",
            n_iter=args.n_iter,
            n_jobs=args.n_jobs,
        )
        emit("ga_poisson_gauss", "end")

    emit("run", "end")


if __name__ == "__main__":
    main()
