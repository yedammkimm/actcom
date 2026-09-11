#!/usr/bin/env bash
# Perplexity sweep over the 3B adapters (2026-08-26).
#
# Compares three configurations on GSM8K test (in-dist) and WikiText-2 (OOD):
#   A) 4D FP4 + E2M1 body  (naive_fp4 pack_4d=fp4)   Exp 2, seeds {42, 123, 456}
#   B) 4D FP8 + E2M1 body  (naive_fp4 pack_4d=fp8)   pilot_e2m1*, seeds {42, 123, 456, 789}
#   C) Standard            (no compression)          seeds {42, 456}   (seed 123 adapter missing)
#
# 8 adapters x ~5 min = ~40 min at 3B. Must be launched after the 70B run frees
# the GPU. Reads all adapters that already exist and skips missing ones.

set -u

CONTAINER=hma-container
OUT_HOST=/home/yedam/HMA/HMA_Project/results/ppl_eval
mkdir -p ${OUT_HOST}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited % 600 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

eval_one() {
    local adapter="$1"
    local group="$2"
    local seed="$3"
    if [[ ! -d "/home/yedam/HMA/HMA_Project/${adapter}" ]]; then
        log "  SKIP ${group}/seed${seed}: adapter missing (${adapter})"
        return 0
    fi
    local tag="ppl__${group}__seed${seed}"
    local log_file="${OUT_HOST}/${tag}.log"
    local out_json="/app/HMA_Project/results/ppl_eval/${tag}.json"
    if [[ -f "${OUT_HOST}/${tag}.json" ]]; then
        log "  SKIP ${tag} (already done)"
        return 0
    fi
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --seed ${seed} \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/ppl_eval 2>/dev/null
    sleep 15
}

log "===== PPL sweep begin ====="
log "waiting for 70B run to finish..."
wait_gpu

log ""
log "--- Group A: 4D FP4 + E2M1 body ---"
eval_one results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260825_193617_adapter  A_4DFP4  42
eval_one results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter A_4DFP4  123
eval_one results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260826_104720_adapter A_4DFP4  456

log ""
log "--- Group B: 4D FP8 + E2M1 body ---"
eval_one results/pilot_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_164235_adapter        B_4DFP8  42
eval_one results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260822_010922_adapter B_4DFP8  123
eval_one results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260822_075818_adapter B_4DFP8  456

log ""
log "--- Group C: Standard ---"
eval_one results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter                C_STD    42
eval_one results/pilot_pack4d_fp8_matrix/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260816_140828_adapter  C_STD    456

log ""
log "===== PPL sweep complete ====="
