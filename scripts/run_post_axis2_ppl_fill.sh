#!/usr/bin/env bash
# Post-axis-2 follow-up:
#   * INT4 seed 42 → NarrativeQA + GovReport PPL (existing JSON only has wiki + gsm8k)
#   * A_4DFP4 seed 789 → 4-corpus PPL (missed due to earlier permission bug, now fixed)
#
# Waits for axis 2 chain to finish before running.

set -u
CONTAINER=hma-container
PPL_OUT=/home/yedam/HMA/HMA_Project/results/ppl_eval
AXIS2_PPL_OUT=/home/yedam/HMA/HMA_Project/results/ppl_axis2

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=54000   # 15 h cap while axis 2 finishes 5 seeds
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

# Both PPL runs are read-only on the adapter and lightweight (~5 min).
run_ppl() {
    local adapter="$1"
    local out_json="$2"
    local log_file="$3"
    local seed="$4"
    if [[ -f "$out_json" ]]; then
        log "  SKIP $(basename $out_json) (already exists)"; return 0
    fi
    log "starting $(basename $log_file .log)"
    # Add-only mode: we want narr + gov only; skip gsm/wiki to save time
    # (Actually just run everything — 5 extra minutes at most, keeps one JSON per adapter.)
    timeout --signal=KILL 1500 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed ${seed} --output ${out_json}" \
        > "${log_file}" 2>&1
    log "  exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/ppl_eval /app/HMA_Project/results/ppl_axis2 2>/dev/null
    sleep 15
}

log "===== post-axis-2 follow-up (adapter-only PPL fills) ====="
wait_gpu || { log "abort"; exit 1; }

# 1. INT4 seed 42 — needs narr + gov (existing ppl__D_INT4__seed42.json only has wiki + gsm)
INT4_S42_ADAPTER="/app/HMA_Project/results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260831_082311_adapter"
# Overwrite the existing 2-corpus file with a full 4-corpus JSON.
INT4_S42_OUT_HOST=${PPL_OUT}/ppl__D_INT4__seed42.json
if [[ -f "$INT4_S42_OUT_HOST" ]] && python3 -c "import json,sys; d=json.load(open('$INT4_S42_OUT_HOST')); sys.exit(0 if 'narrativeqa_ppl' in d else 1)" 2>/dev/null; then
    log "  INT4 seed 42 already has 4 corpora; skipping"
else
    log ""
    log "--- INT4 seed 42 → 4-corpus PPL ---"
    run_ppl "$INT4_S42_ADAPTER" \
        "/app/HMA_Project/results/ppl_eval/ppl__D_INT4__seed42.json" \
        "${PPL_OUT}/ppl__D_INT4__seed42.log" \
        42
fi

# 2. A_4DFP4 seed 789 — missed due to permission bug
S789_ADAPTER_HOST=$(ls -d /home/yedam/HMA/HMA_Project/results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__*_adapter 2>/dev/null | head -1)
if [[ -n "$S789_ADAPTER_HOST" ]]; then
    S789_ADAPTER=${S789_ADAPTER_HOST#/home/yedam/HMA/HMA_Project/}
    S789_ADAPTER_DOCKER="/app/HMA_Project/${S789_ADAPTER}"
    log ""
    log "--- A_4DFP4 seed 789 → 4-corpus PPL ---"
    run_ppl "$S789_ADAPTER_DOCKER" \
        "/app/HMA_Project/results/ppl_axis2/ppl_axis2__A_4DFP4__seed789.json" \
        "${AXIS2_PPL_OUT}/ppl_axis2__A_4DFP4__seed789.log" \
        789
else
    log "  seed 789 adapter not found; skipping"
fi

log ""
log "===== post follow-up done ====="
