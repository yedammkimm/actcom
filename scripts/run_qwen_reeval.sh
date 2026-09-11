#!/usr/bin/env bash
# Qwen 3B adapter re-eval sweep (2026-08-24).
# Purpose: apples-to-apples comparison of INT4 vs E2M1 by re-evaluating three
# adapters with mnt=512 to eliminate truncation-induced accuracy drop.
#
# Original conditions (all 3):
#   n_samples=500, max_new_tokens=256, stop_strings=None
#   qwen3b_skip_check/oamp  (INT4):  acc 69.00  n_trunc 212 (42.4%)
#   qwen3b_skip_check/naive_fp4:      acc 68.20  n_trunc 300 (60.0%)
#   qwen3b_e2m1/oamp        (E2M1):  acc 74.60  n_trunc  29 ( 5.8%)  ← sanity
#
# Re-eval conditions:
#   n_samples=500, max_new_tokens=512, stop_strings=None (match original)
#
# Order in chain: runs AFTER gamma_full probe (waits for GPU),
# BEFORE 70b_gc_sweep (its wait pattern includes batch_eval.py).

set -u
CONTAINER=hma-container
OUT_HOST=/home/yedam/HMA/HMA_Project/results/qwen3b_reeval
mkdir -p ${OUT_HOST}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/results/qwen3b_reeval
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/qwen3b_reeval 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Qwen 3B re-eval sweep queued ====="

log "waiting for GPU (any run_experiment.py or gradient_error_probe.py)..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

reeval() {
    local tag="$1"
    local adapter="$2"
    local log_file="${OUT_HOST}/${tag}.log"
    log ""
    log "--- reeval ${tag} ---"
    log "  adapter: ${adapter}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/batch_eval.py \
         --adapter_path '${adapter}' \
         --n_samples 500 --max_new_tokens 512 \
         --no_stop_strings --seed 42" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null
    sleep 30
}

log ""
log "===== Qwen 3B re-eval sweep begin ====="

# INT4 adapters (original mnt=256, 42-60% truncated)
reeval "int4_oamp_mnt512" \
    "/app/HMA_Project/results/qwen3b_skip_check/accuracy__oamp__Qwen2.5-3B-Instruct__nf4__rmsbf16__seed42__20260817_213114_adapter"

reeval "int4_naive_fp4_mnt512" \
    "/app/HMA_Project/results/qwen3b_skip_check/accuracy__naive_fp4__Qwen2.5-3B-Instruct__nf4__rmsbf16__seed42__20260817_223318_adapter"

# E2M1 adapter (sanity — expected minimal change since only 5.8% truncated)
reeval "e2m1_oamp_mnt512" \
    "/app/HMA_Project/results/qwen3b_e2m1/accuracy__oamp__Qwen2.5-3B-Instruct__nf4__rmsbf16__seed42__20260824_125402_adapter"

log ""
log "===== Qwen 3B re-eval sweep complete ====="
