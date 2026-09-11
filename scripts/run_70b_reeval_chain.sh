#!/bin/bash
# Sequential re-eval of Standard/naive/OAMP 70B adapters with stop_strings.
# Base already done via base_eval.py (see base_eval_reeval_mnt512.json).
set -euo pipefail

BASE=/home/yedam/HMA/HMA_Project/results/70b_accuracy
CONTAINER=hma-container

# Adapter dirs (seed42 chain from 2026-08-18/19).
STD_ADAPTER="results/70b_accuracy/accuracy__standard__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260818_121120_adapter"
NAIVE_ADAPTER="results/70b_accuracy/accuracy__naive_fp4__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260818_235122_adapter"
OAMP_ADAPTER="results/70b_accuracy/accuracy__oamp__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260819_075943_adapter"

# stop_strings passed as JSON to bypass shell escaping. batch_eval.py handles both
# comma-separated and JSON-list formats via _parse_stop_strings.
SS='["\nQ:", "\n\nQ:"]'
MAX_NEW=512
N=200

run_one() {
    local tag=$1
    local adapter=$2
    local log="${BASE}/reeval_${tag}_mnt${MAX_NEW}.log"
    echo "=== [$(date +%H:%M:%S)] ${tag} re-eval start ===" | tee -a "${log}"
    docker exec hma-container bash -c "cd /app/HMA_Project && python scripts/batch_eval.py \
        --adapter_path ${adapter} \
        --n_samples ${N} \
        --max_new_tokens ${MAX_NEW} \
        --stop_strings '${SS}'" >> "${log}" 2>&1
    echo "=== [$(date +%H:%M:%S)] ${tag} done ===" | tee -a "${log}"
}

run_one standard  "${STD_ADAPTER}"
run_one naive_fp4 "${NAIVE_ADAPTER}"
run_one oamp      "${OAMP_ADAPTER}"

echo "=== [$(date +%H:%M:%S)] all three re-evals complete ==="
