#!/usr/bin/env bash
# Post-70B chain: PPL sweep (~45min) + [4] mechanism probe (~10min).
#
# Order:
#   1. Wait for the 70B 4D-FP4 python to finish.
#   2. Run PPL sweep over the 8 available 3B adapters
#      (results/ppl_eval/ppl__<group>__seed<N>.json).
#   3. Run Exp 4 mechanism probe: INT4 vs E2M1 across residual, MLP, Q/K, V.
#      Single fresh-model session (reproducible). Skips trained-adapter path.
#
# Each stage is idempotent: skips if output already exists.

set -u

CONTAINER=hma-container
PPL_OUT_HOST=/home/yedam/HMA/HMA_Project/results/ppl_eval
MECH_OUT_HOST=/home/yedam/HMA/HMA_Project/results/audit
mkdir -p ${PPL_OUT_HOST} ${MECH_OUT_HOST}
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

eval_ppl() {
    local adapter="$1"
    local group="$2"
    local seed="$3"
    if [[ ! -d "/home/yedam/HMA/HMA_Project/${adapter}" ]]; then
        log "  SKIP ${group}/seed${seed}: adapter missing (${adapter})"
        return 0
    fi
    local tag="ppl__${group}__seed${seed}"
    local log_file="${PPL_OUT_HOST}/${tag}.log"
    local out_json="/app/HMA_Project/results/ppl_eval/${tag}.json"
    if [[ -f "${PPL_OUT_HOST}/${tag}.json" ]]; then
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

log "===== chain begin ====="
log "waiting for 70B run to finish..."
wait_gpu

log ""
log "===== stage 1/2: PPL sweep (8 adapters) ====="

log ""
log "--- Group A: 4D FP4 + E2M1 body ---"
eval_ppl results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260825_193617_adapter  A_4DFP4  42
eval_ppl results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter A_4DFP4  123
eval_ppl results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260826_104720_adapter A_4DFP4  456

log ""
log "--- Group B: 4D FP8 + E2M1 body ---"
eval_ppl results/pilot_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_164235_adapter        B_4DFP8  42
eval_ppl results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260822_010922_adapter B_4DFP8  123
eval_ppl results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260822_075818_adapter B_4DFP8  456

log ""
log "--- Group C: Standard ---"
eval_ppl results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter                C_STD    42
eval_ppl results/pilot_pack4d_fp8_matrix/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260816_140828_adapter  C_STD    456

log ""
log "===== stage 2/2: [4] mechanism probe (INT4 vs E2M1 across roles) ====="

MECH_TS=$(date +%Y%m%d_%H%M%S)
MECH_TAG="mechanism_int4_vs_e2m1_${MECH_TS}"
MECH_LOG="${MECH_OUT_HOST}/${MECH_TAG}.log"
MECH_JSON="/app/HMA_Project/results/audit/${MECH_TAG}.json"

MECH_FILTERS="qk_int4_block,qk_e2m1_block,v_int4_block,v_e2m1_block,residual_int4,residual_e2m1,mlp_int4,mlp_e2m1"

log "starting ${MECH_TAG}"
docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python oamp_train_engine/audit/gradient_error_probe.py \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --adapter '' \
     --filters '${MECH_FILTERS}' \
     --body_encoding e2m1 \
     --output ${MECH_JSON}" \
    > "${MECH_LOG}" 2>&1
RC=$?
log "  ${MECH_TAG} exit=${RC}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/audit 2>/dev/null

log ""
log "===== chain complete ====="
log "  PPL results:       ${PPL_OUT_HOST}/ppl__*.json"
log "  Mechanism result:  ${MECH_OUT_HOST}/${MECH_TAG}.json"
