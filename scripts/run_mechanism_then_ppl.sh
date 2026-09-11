#!/usr/bin/env bash
# 2026-08-31 rev: 3-stage chain — mechanism, PPL, INT4 §5 baseline.
#
# Stage 1 and 2 need no watchdog (single fw+bw on 3B, forward-only PPL).
# Stage 3 uses the eval watchdog in oamp/evaluate.py + wall timeout.
# 70B accuracy is launched separately after we judge stage 3's outcome.
#
# Stages:
#   1. [4] Mechanism probe (13 filters, ~1 min):
#        dist_stats + qk/v/residual_pre_attn/residual_pre_mlp/mlp/o × {int4,e2m1}
#   2. PPL sweep over the 8 available 3B adapters (GSM8K test + WikiText-2).
#   3. naive_fp4 body=int4 pack_4d=fp4 seed 42 (3.5 h)
#        Supplies the missing "cos 0.36" datapoint for the §5 threshold plot
#        (matched-method ablation against naive_fp4+e2m1+fp4 = 55.33).
#
# Idempotent: skips if the output already exists.
# Per-stage timeout guards against silent hang.

set -u

CONTAINER=hma-container
PPL_OUT_HOST=/home/yedam/HMA/HMA_Project/results/ppl_eval
MECH_OUT_HOST=/home/yedam/HMA/HMA_Project/results/audit
mkdir -p ${PPL_OUT_HOST} ${MECH_OUT_HOST}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Wait for any lingering python jobs (should be none, but sanity).
wait_gpu() {
    local waited=0
    local MAX_WAIT=1800   # 30 min max; 70B is not running at this stage
    while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU (${waited}s); killing stray python"
            docker exec ${CONTAINER} pkill -9 -f "python run_experiment.py" 2>/dev/null
            break
        fi
        if (( waited % 300 == 0 )); then
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
    timeout --signal=KILL 900 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --seed ${seed} \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT (>15 min); killing python inside container"
        docker exec ${CONTAINER} pkill -9 -f "eval_perplexity" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/ppl_eval 2>/dev/null
    sleep 15
}

log "===== chain begin ====="
wait_gpu

# ------------------------------------------------------------------
# Stage 1: [4] mechanism probe (13 filters, one fresh session, ~15 min)
# ------------------------------------------------------------------
log ""
log "===== stage 1/2: [4] mechanism probe (INT4 vs E2M1 across roles + dist stats) ====="

MECH_TS=$(date +%Y%m%d_%H%M%S)
MECH_TAG="mechanism_int4_vs_e2m1_${MECH_TS}"
MECH_LOG="${MECH_OUT_HOST}/${MECH_TAG}.log"
MECH_JSON="/app/HMA_Project/results/audit/${MECH_TAG}.json"

MECH_FILTERS="dist_stats,\
qk_int4_block,qk_e2m1_block,\
v_int4_block,v_e2m1_block,\
residual_pre_attn_int4,residual_pre_attn_e2m1,\
residual_pre_mlp_int4,residual_pre_mlp_e2m1,\
mlp_int4,mlp_e2m1,\
o_int4,o_e2m1"

log "starting ${MECH_TAG}"
timeout --signal=KILL 1800 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python oamp_train_engine/audit/gradient_error_probe.py \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --adapter '' \
     --filters '${MECH_FILTERS}' \
     --body_encoding e2m1 \
     --output ${MECH_JSON}" \
    > "${MECH_LOG}" 2>&1
RC=$?
if (( RC == 137 || RC == 124 )); then
    log "  ${MECH_TAG} TIMEOUT (>30 min); killing gradient_error_probe python"
    docker exec ${CONTAINER} pkill -9 -f "gradient_error_probe" 2>/dev/null
fi
log "  ${MECH_TAG} exit=${RC}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/audit 2>/dev/null

sleep 30

# ------------------------------------------------------------------
# Stage 2: PPL sweep over 8 adapters (~45 min)
# ------------------------------------------------------------------
log ""
log "===== stage 2/2: PPL sweep (8 adapters) ====="

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
log "===== chain complete ====="
log "  Mechanism result:  ${MECH_OUT_HOST}/${MECH_TAG}.json"
log "  PPL results:       ${PPL_OUT_HOST}/ppl__*.json"
