#!/usr/bin/env bash
# Reproduce the benchmark tables: evaluate the mt checkpoint and the
# opus-mt baselines on the standard WMT test sets, then rebuild the table.
#
# Usage: scripts/run_benchmarks.sh [checkpoint]
set -euo pipefail

CHECKPOINT="${1:-artifacts/release/mt-small-deen.pt}"
PYTHON="${PYTHON:-.venv/bin/python}"
TOKENIZER="${TOKENIZER:-artifacts/tokenizer/tokenizer.model}"

echo "== mt checkpoint: ${CHECKPOINT}"
"${PYTHON}" -m mt.evaluate \
    --model "${CHECKPOINT}" \
    --tokenizer "${TOKENIZER}" \
    --test-sets wmt14,wmt19 \
    --directions de-en,en-de \
    --beam-size 5 \
    --name "mt-small-deen" \
    --output results/mt-small.json

for direction in de-en en-de; do
    echo "== opus-mt ${direction}"
    "${PYTHON}" scripts/run_baseline.py \
        --model-name "Helsinki-NLP/opus-mt-${direction}" \
        --direction "${direction}" \
        --test-sets wmt14,wmt19 \
        --beam-size 5 \
        --output "results/baseline-opus-mt.${direction}.json"
done

"${PYTHON}" scripts/make_tables.py \
    --inputs results/*.json \
    --output results/summary.md
echo "tables written to results/summary.md"
