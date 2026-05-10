#!/usr/bin/env bash
# E1 — single-axis n_jobs sweep on the cluster.
# Run from the crispat-profiling root after `pixi install`.
#
# Usage:
#   scripts/run_e1_sweep.sh             # default sweep
#   scripts/run_e1_sweep.sh "1 2 4 8"   # custom n_jobs list
#
# Each run lands in its own results/parallel_n${n}/ directory with:
#   - assignments.csv            (gauss, poisson_gauss subdirs)
#   - obkit-events.jsonl         (phase anchors)
#   - denet.jsonl                (sampling profile)
set -euo pipefail

SWEEP="${1:-1 2 4 8 16 32 64 100}"

echo "E1 sweep: n_jobs ∈ { $SWEEP }"
echo "host=$(hostname) cores=$(nproc)"

# 1. Prepare input (cheap, idempotent)
pixi run python data/prepare.py --out results/data/input.h5ad

# 2. Sweep
for n in $SWEEP; do
    out="results/parallel_n${n}"
    if [[ -f "$out/poisson_gauss/assignments.csv" ]]; then
        echo "[skip] $out already complete"
        continue
    fi
    rm -rf "$out"
    mkdir -p "$out"
    echo "[run]  n_jobs=$n -> $out"
    pixi run -- \
        denet --json --out "$out/denet.jsonl" \
        run python methods/parallel/run.py -- \
            --in results/data/input.h5ad \
            --out "$out" \
            --n-jobs "$n"
done

# 3. Aggregate
pixi run python evaluation/correctness.py --results results
pixi run python evaluation/scaling.py    --results results
pixi run python plots/make_plots.py      --results results
pixi run python plots/make_e2_plot.py    --results results || true

echo "[done] results in results/"
echo "[done] scaling table:"
cat results/scaling.csv
