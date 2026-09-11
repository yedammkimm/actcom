#!/usr/bin/env bash
# Experiment 1b (GACT deterministic probe, SR-off) — 2026-08-26.
# Isolates the contribution of stochastic rounding from flat-reshape/granularity
# in explaining the cos 0.05 result of qk_gact_block (SR on).
#
# Comparison:
#   qk_gact_block      : SR on  → previously 0.05 (fresh) / 0.16 (step300)
#   qk_gact_block_det  : SR off → this run
#
# Also re-runs qk_e2m1_block and qk_int4_block for internal consistency
# within the same session (adapter g_norm state).

set -u
CONTAINER=hma-container
ADAPTER=/app/HMA_Project/results/pilot_sr_lr/accuracy__oamp__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260816_003830_adapter
FILTERS="qk_gact_block,qk_gact_block_det,v_gact_block,v_gact_block_det,qk_e2m1_block,qk_int4_block,v_e2m1_block,v_int4_block,drop_4d,attn_4d"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Exp 1b (GACT SR-off probe) queued ====="

log "waiting for GPU..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py|python.*batch_eval.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

TS=$(date +%Y%m%d_%H%M%S)
OUT=/app/HMA_Project/results/audit/gact_sr_off_probe_${TS}.json
LOG_HOST=/home/yedam/HMA/HMA_Project/results/audit/gact_sr_off_probe_${TS}.log

log ""
log "===== Running GACT SR-off probe ====="
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

log "===== Exp 1b complete ====="
log "  JSON: ${OUT}"
