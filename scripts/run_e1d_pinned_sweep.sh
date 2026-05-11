#!/usr/bin/env bash
# E1d — full sweep with NUMA pinning. The headline run for the writeup.
#
# All runs are pinned with `taskset -c 0-31,64-95` (NUMA nodes 0+1, 32
# physical cores / 64 logical cores). This caps n_jobs at 64, which is
# the past-knee operating region anyway. AnnData allocated by the parent
# lands on nodes 0-1, workers access it locally → uniform topology, no
# scheduler bouncing.
#
# Results go to results-e1d-pinned/ so the headline E1b results
# (unpinned) are preserved alongside for comparison.
#
# Usage:
#   scripts/run_e1d_pinned_sweep.sh             # default sweep
#   scripts/run_e1d_pinned_sweep.sh "1 8 32 64" # custom n_jobs list
set -euo pipefail

# Ensure pixi is on PATH for non-login shells (cluster SSH).
export PATH="$HOME/.pixi/bin:$PATH"

ROOT="results-e1d-pinned"
SWEEP="${1:-1 2 4 8 16 32 64}"

# 2 NUMA nodes, 32 physical / 64 logical cores.
PIN_CPUS="0-31,64-95"

echo "E1d pinned sweep: n_jobs ∈ { $SWEEP }, taskset -c $PIN_CPUS"
echo "host=$(hostname) cores=$(nproc)"

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
    echo "[run]  n_jobs=$n -> $out  (taskset -c $PIN_CPUS)"
    taskset -c "$PIN_CPUS" \
        pixi run -- \
        denet --json --out "$out/denet.jsonl" --enable-ebpf \
        run python methods/parallel/run.py -- \
            --in "$ROOT/data/input.h5ad" \
            --out "$out" \
            --n-jobs "$n" \
            --no-plots
done

# Aggregate + plot
pixi run python evaluation/correctness.py    --results "$ROOT" || true
pixi run python evaluation/scaling.py        --results "$ROOT"
pixi run python evaluation/phases.py         --results "$ROOT"
pixi run python plots/make_plots.py          --results "$ROOT"
pixi run python plots/make_phase_plots.py    --results "$ROOT"
pixi run python plots/make_denet_plots.py    --results "$ROOT"

echo
echo "[done] pinned sweep in $ROOT/"
echo "[done] speedup curve + USL fit: $ROOT/plots/speedup_curve.png"
cat "$ROOT/scaling.csv"
