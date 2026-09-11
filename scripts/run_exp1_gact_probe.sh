#!/usr/bin/env bash
# Experiment 1 (GACT-style quantizer probe) — 2026-08-25.
# Waits for Exp 2 (naive 4-bit E2M1 training) to finish, then runs a probe
# that compares three 4-bit blockwise quantizers on the Q/K and V head views:
#   - qk_int4_block  : INT4 symmetric absmax, group=128 (ours, baseline)
#   - qk_e2m1_block  : E2M1 non-uniform, group=128 (ours, current default)
#   - qk_gact_block  : GACT-style affine, group=256, SR (Chen et al. 2021)
#
# If gact_block cos on Q/K also collapses (< 0.5), §5.1's mechanism claim
# generalizes across 4-bit blockwise quantizers.

set -u
CONTAINER=hma-container
ADAPTER=/app/HMA_Project/results/pilot_sr_lr/accuracy__oamp__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260816_003830_adapter
FILTERS="qk_gact_block,v_gact_block,qk_e2m1_block,v_e2m1_block,qk_int4_block,v_int4_block,qk_only,v_only,drop_4d,attn_4d"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Exp 1 (GACT-style probe) queued ====="

log "waiting for GPU..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py|python.*batch_eval.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

TS=$(date +%Y%m%d_%H%M%S)
OUT=/app/HMA_Project/results/audit/gact_style_probe_${TS}.json
LOG_HOST=/home/yedam/HMA/HMA_Project/results/audit/gact_style_probe_${TS}.log

log ""
log "===== Running GACT-style probe ====="
log "  filters: ${FILTERS}"
log "  output:  ${OUT}"

docker exec \
    -e GRAD_PROBE_ADAPTER="${ADAPTER}" \
    ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python oamp_train_engine/audit/gradient_error_probe.py \
     --filters ${FILTERS} \
     --output ${OUT}" \
    > "${LOG_HOST}" 2>&1
rc=$?
log "  probe exit=${rc}"

docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/audit 2>/dev/null

log "===== Exp 1 complete ====="
log "  JSON: ${OUT}"
log "  LOG:  ${LOG_HOST}"
