#!/usr/bin/env bash
# E1b — phase-resolved instrumented sweep.
#
# Requires the crispat-fork branch parallel-gauss-instr to be the one pixi
# resolves. Swap pixi.toml's crispat branch to parallel-gauss-instr and run
# `pixi update crispat` before invoking this script.
#
# Writes to results-e1-instr/ (separate from results/ so the original E1
# headline numbers in E1_RESULTS.md aren't clobbered).
#
# Usage:
#   scripts/run_e1_instr_sweep.sh             # default sweep
#   scripts/run_e1_instr_sweep.sh "1 8 32 64" # custom n_jobs list
set -euo pipefail

ROOT="results-e1-instr"
SWEEP="${1:-1 2 4 8 16 32 64 100}"

echo "E1b instrumented sweep: n_jobs ∈ { $SWEEP }"
echo "host=$(hostname) cores=$(nproc)"

# Cheap, idempotent — reuse the existing data prep output if present.
mkdir -p "$ROOT/data"
if [[ ! -f "$ROOT/data/input.h5ad" ]]; then
    if [[ -f "results/data/input.h5ad" ]]; then
        ln -sf "$(realpath results/data/input.h5ad)" "$ROOT/data/input.h5ad"
    else
        pixi run python data/prepare.py --out "$ROOT/data/input.h5ad"
    fi
fi

for n in $SWEEP; do
    out="$ROOT/parallel_n${n}"
    if [[ -f "$out/poisson_gauss/assignments.csv" ]]; then
        echo "[skip] $out already complete"
        continue
    fi
    rm -rf "$out"
    mkdir -p "$out"
    echo "[run]  n_jobs=$n -> $out"
    pixi run -- \
        denet --json --out "$out/denet.jsonl" --enable-ebpf \
        run python methods/parallel/run.py -- \
            --in "$ROOT/data/input.h5ad" \
            --out "$out" \
            --n-jobs "$n" \
            --no-plots
done

echo "[done] instrumented results in $ROOT/"
echo "       parent phase events: $ROOT/parallel_n*/obkit-events.jsonl"
echo "       worker fit_one events: $ROOT/parallel_n*/obkit-events.worker.<pid>.jsonl"
