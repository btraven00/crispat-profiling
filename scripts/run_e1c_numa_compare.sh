#!/usr/bin/env bash
# E1c — NUMA pinning sanity check at n_jobs=32.
#
# Compares two runs:
#   - results-e1c/n32_unpinned/  (no affinity constraint)
#   - results-e1c/n32_pinned/    (workers pinned to NUMA node 0: cpus 0-15,64-79)
#
# Same instrumented branch (parallel-gauss-instr), so phase events from
# obkit + per-worker fit_one durations are produced for both runs.
# Method: ga_gauss only — keeps cluster cost low. ga_poisson_gauss should
# follow the same shape if NUMA is the cause.
#
# After running, compare median fit_one between the two runs:
#   - same  => NUMA is not the cause; look elsewhere (BLAS allocator,
#              TLB shootdown, ...)
#   - lower under pinning => NUMA cross-node traffic confirmed; argues for
#              an mmap-backed matrix (BPCells) or NUMA-aware spawner.
set -euo pipefail

ROOT="results-e1c"
N=32
# omnibenchmark: AMD EPYC 7742, 4 NUMA nodes of 16 physical / 32 logical cores.
# Node 0 = cores 0-15 (+ SMT siblings 64-79).
NODE0_CPUS="0-15,64-79"

mkdir -p "$ROOT/data"
if [[ ! -f "$ROOT/data/input.h5ad" ]]; then
    if [[ -f "results/data/input.h5ad" ]]; then
        ln -sf "$(realpath results/data/input.h5ad)" "$ROOT/data/input.h5ad"
    else
        pixi run python data/prepare.py --out "$ROOT/data/input.h5ad"
    fi
fi

run_one() {
    local out="$1" tag="$2"
    shift 2
    if [[ -f "$out/gauss/assignments.csv" ]]; then
        echo "[skip] $out already complete"
        return
    fi
    rm -rf "$out"
    mkdir -p "$out"
    echo "[run]  $tag -> $out"
    "$@" pixi run -- \
        denet --json --out "$out/denet.jsonl" --enable-ebpf \
        run python methods/parallel/run.py -- \
            --in "$ROOT/data/input.h5ad" \
            --out "$out" \
            --method gauss \
            --n-jobs "$N" \
            --no-plots
}

# Unpinned baseline
run_one "$ROOT/n${N}_unpinned" "n=$N unpinned" \
    env -u CRISPAT_WORKER_CPUS

# Pinned to NUMA node 0
run_one "$ROOT/n${N}_pinned"   "n=$N pinned to node 0 ($NODE0_CPUS)" \
    env CRISPAT_WORKER_CPUS="$NODE0_CPUS"

# Aggregate + summary + pinned-vs-unpinned overlay
pixi run python evaluation/phases.py          --results "$ROOT"
pixi run python plots/make_phase_plots.py     --results "$ROOT"
pixi run python plots/make_e1c_compare_plot.py --results "$ROOT"

echo
echo "=== fit_one comparison (median per run) ==="
pixi run python -c "
import pandas as pd
w = pd.read_csv('$ROOT/phases_workers.csv')
fit = w[w.event=='fit_one'].dropna(subset=['duration_s'])
# n_jobs is the same (32) for both runs; tag by which subdir the rows came from.
# phases.py keys by parallel_n<N> dir, so the two runs aren't distinguishable
# in the CSV alone — peek at each separately.
print('Note: phases_workers.csv currently lumps both runs under n_jobs=32.')
print('Per-run breakdown:')
for tag in ('n${N}_unpinned', 'n${N}_pinned'):
    import json, pathlib, re
    durs = []
    for f in pathlib.Path('$ROOT/' + tag).glob('obkit-events.worker.*.jsonl'):
        with f.open() as h:
            stack = []
            for line in h:
                r = json.loads(line)
                if r.get('event') != 'fit_one': continue
                if r['phase'] == 'start':
                    stack.append(pd.to_datetime(r['ts']))
                elif r['phase'] == 'end' and stack:
                    durs.append((pd.to_datetime(r['ts']) - stack.pop()).total_seconds())
    if durs:
        s = pd.Series(durs)
        print(f'  {tag:20s}  count={len(s):4d}  median={s.median():.3f}s  p95={s.quantile(0.95):.3f}s  max={s.max():.3f}s')
"
