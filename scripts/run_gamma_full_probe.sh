#!/usr/bin/env bash
# Follow-up audit after Qwen E2M1 chain: gradient_error_probe with gamma_full.
# Waits for the Qwen chain (PID passed in $1 or docker's naive_fp4) to finish,
# then runs the grad probe with gamma_full + companion filters to close the
# "3D + 4D vs Everything" gap in §5.1.

set -u
CONTAINER=hma-container
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Follow-up: gamma_full probe queued ====="

# Wait for any run_experiment.py inside the container to finish.
log "waiting for GPU (any run_experiment.py inside container)..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/yedam/HMA/HMA_Project/results/audit/gamma_full_probe_${TS}.json
LOG_HOST=/home/yedam/HMA/HMA_Project/results/audit/gamma_full_probe_${TS}.log
ADAPTER=/app/HMA_Project/results/pilot_sr_lr/accuracy__oamp__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260816_003830_adapter

log "starting gradient_error_probe with gamma_full + companion filters"
log "  model:    meta-llama/Llama-3.2-3B-Instruct"
log "  adapter:  ${ADAPTER}"
log "  filters:  gamma_full,drop_4d,attn_4d,qk_only,v_only,o_only,qkv_all"
log "  output:   ${OUT}"

docker exec \
    -e GRAD_PROBE_ADAPTER="${ADAPTER}" \
    ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python oamp_train_engine/audit/gradient_error_probe.py \
     --filters gamma_full,drop_4d,attn_4d,qk_only,v_only,o_only,qkv_all \
     --output /app/HMA_Project/results/audit/gamma_full_probe_${TS}.json" \
    > "${LOG_HOST}" 2>&1
rc=$?
log "  probe exit=${rc}"

docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/audit 2>/dev/null

log "===== Follow-up complete ====="
log "  JSON: ${OUT}"
log "  LOG:  ${LOG_HOST}"
