#!/usr/bin/env bash
# Post-GC investigation chain (2026-08-25):
#   [A] gradient_error_probe reproducibility test: run twice with identical
#       args, compare g_true_norm to isolate whether the OLD (08-20) vs NEW
#       (08-24) discrepancy is deterministic or stochastic.
#   [B] Qwen 3B naive_fp4 E2M1 adapter re-eval (mnt=512, no_stop_strings) for
#       apples-to-apples INT4 vs E2M1 comparison.
#
# Waits for 70B GC sweep in-flight, then runs A and B sequentially.

set -u
CONTAINER=hma-container
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Post-GC investigation chain queued ====="

log "waiting for GPU (any run_experiment.py, gradient_error_probe.py, or batch_eval.py)..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py|python.*batch_eval.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

# --------------------------------------------------------------------------
# [A] probe reproducibility -- run twice with same args, compare g_true_norm
# --------------------------------------------------------------------------
ADAPTER=/app/HMA_Project/results/pilot_sr_lr/accuracy__oamp__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260816_003830_adapter
FILTERS="gamma_full,drop_4d,attn_4d,qk_only,v_only,o_only,qkv_all"

run_probe() {
    local run_id="$1"
    local ts=$(date +%Y%m%d_%H%M%S)
    local out="/app/HMA_Project/results/audit/gamma_full_probe_repro${run_id}_${ts}.json"
    local log_file="/home/yedam/HMA/HMA_Project/results/audit/gamma_full_probe_repro${run_id}_${ts}.log"
    log ""
    log "--- probe run #${run_id} ---"
    log "  filters: ${FILTERS}"
    log "  output:  ${out}"
    docker exec \
        -e GRAD_PROBE_ADAPTER="${ADAPTER}" \
        ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python oamp_train_engine/audit/gradient_error_probe.py \
         --filters ${FILTERS} \
         --output ${out}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  probe run #${run_id} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/audit 2>/dev/null
    sleep 30
}

log ""
log "===== [A] Probe reproducibility test ====="
run_probe 1
run_probe 2

# --------------------------------------------------------------------------
# [B] Qwen naive_fp4 E2M1 adapter re-eval (mnt=512, no_stop_strings)
# --------------------------------------------------------------------------
log ""
log "===== [B] Qwen naive_fp4 E2M1 adapter re-eval ====="
ADAPTER_B=/app/HMA_Project/results/qwen3b_e2m1/accuracy__naive_fp4__Qwen2.5-3B-Instruct__nf4__rmsbf16__seed42__20260824_135009_adapter
LOG_B=/home/yedam/HMA/HMA_Project/results/qwen3b_reeval/naive_fp4_e2m1_mnt512.log

# Find actual adapter path (timestamp may vary)
ACTUAL_ADAPTER=$(docker exec ${CONTAINER} bash -c "ls -d /app/HMA_Project/results/qwen3b_e2m1/accuracy__naive_fp4__*_adapter 2>/dev/null | head -1")
if [ -z "${ACTUAL_ADAPTER}" ]; then
    log "  ERROR: Qwen naive_fp4 E2M1 adapter not found"
else
    log "  adapter: ${ACTUAL_ADAPTER}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/batch_eval.py \
         --adapter_path '${ACTUAL_ADAPTER}' \
         --n_samples 500 --max_new_tokens 512 \
         --no_stop_strings --seed 42" \
        > "${LOG_B}" 2>&1
    log "  naive_fp4 E2M1 mnt=512 exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null
fi

log ""
log "===== Post-GC chain complete ====="
