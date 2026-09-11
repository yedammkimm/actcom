#!/usr/bin/env bash
# Post-axis-3 tail — re-run seed 789 PPL (initial run exited=1 before creating log).
# Adapter is intact (results/axis3_b4dfp8/accuracy__..._seed789__20260901_202010_adapter).
# Wait for the axis 3 chain to finish, then run PPL for seed 789 only.

set -u
CONTAINER=hma-container
PPL_OUT=results/ppl_axis3
PPL_HOST=/home/yedam/HMA/HMA_Project/${PPL_OUT}

ADAPTER_REL="results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260901_202010_adapter"
PPL_TAG="ppl_axis3__B_4DFP8__seed789"
PPL_JSON="/app/HMA_Project/${PPL_OUT}/${PPL_TAG}.json"
PPL_LOG="${PPL_HOST}/${PPL_TAG}.log"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=25200   # 7h — enough for seed 3033/4042 remainder
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/(batch_eval|eval_perplexity))" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU"; return 1
        fi
        if (( waited % 900 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

log "===== ppl_axis3 seed 789 re-run — waiting for axis 3 chain ====="
wait_gpu || { log "abort"; exit 1; }

if [[ -f "${PPL_HOST}/${PPL_TAG}.json" ]]; then
    log "  ${PPL_TAG} already done — nothing to do"; exit 0
fi

log ""
log "--- ${PPL_TAG} PPL re-run (4 corpora, ~5 min) ---"
timeout --signal=KILL 1500 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python scripts/eval_perplexity.py \
     --adapter_path ${ADAPTER_REL} \
     --weight_quant nf4 --bf16_rmsnorm \
     --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
     --extra_corpora narrativeqa,govreport --extra_chunks 200 \
     --seed 789 --output ${PPL_JSON}" \
    > "${PPL_LOG}" 2>&1
rc=$?
log "  ${PPL_TAG} exit=${rc}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null

if (( rc == 0 )); then
    log "  n=8 B_4DFP8 complete; run scripts/analyze_axis3.py"
fi
