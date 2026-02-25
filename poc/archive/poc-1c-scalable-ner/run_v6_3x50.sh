#!/usr/bin/env bash
# Run v6 strategy on 50 docs x 3 seeds, saving separate result files.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SEEDS=(42 100 999)
NDOCS=50
APPROACH="hybrid_v5"
STRATEGY="strategy_v6"
RESULTS_DIR="artifacts/results"

echo "=== V6 3x50 Benchmark ==="
echo "Seeds: ${SEEDS[*]}"
echo "Start: $(date)"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo "  RUN: seed=$seed, n_docs=$NDOCS"
    echo "  Start: $(date)"
    echo "========================================"

    python benchmark_comparison.py \
        --approach "$APPROACH" \
        --strategy "$STRATEGY" \
        --n-docs "$NDOCS" \
        --seed "$seed"

    # Rename the output to include the seed
    SRC="$RESULTS_DIR/${APPROACH}_${STRATEGY}_results.json"
    DST="$RESULTS_DIR/${APPROACH}_${STRATEGY}_seed${seed}_50docs.json"
    cp "$SRC" "$DST"
    echo "  Saved: $DST"
    echo ""
done

echo "=== All 3 runs complete ==="
echo "End: $(date)"
echo ""

# Quick summary
python -c "
import json, sys
seeds = [42, 100, 999]
print(f'{'Seed':<8} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'Time':>8}')
print('-' * 64)
for s in seeds:
    path = f'artifacts/results/hybrid_v5_strategy_v6_seed{s}_50docs.json'
    try:
        d = json.load(open(path))
        m = d['metrics']
        t = d['totals']
        print(f'{s:<8} {m[\"precision\"]*100:>7.1f}% {m[\"recall\"]*100:>7.1f}% {m[\"f1\"]:>8.3f} {t[\"tp\"]:>6} {t[\"fp\"]:>6} {t[\"fn\"]:>6} {d[\"total_elapsed_seconds\"]:>7.1f}s')
    except Exception as e:
        print(f'{s:<8} FAILED: {e}')
"
