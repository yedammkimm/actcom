#!/usr/bin/env bash
# 200-sample uniform re-eval — stop+mnt=512 — 9 adapters
# INT4 s42 included as validation cross-check against the 500-sample value.
# Waits for the ongoing INT4 s42 500-sample eval to finish first.

set -u
CONTAINER=hma-container
OUT_DIR=/home/yedam/HMA/HMA_Project/results/reeval_stop_mnt512_n200
mkdir -p ${OUT_DIR}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/results/reeval_stop_mnt512_n200

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=5400   # 90 min for INT4 s42 500-sample to finish
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/(batch_eval|eval_perplexity))" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU"; return 1
        fi
        if (( waited % 300 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

# (rel_path | label | seed)
ADAPTERS=(
    "results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260831_082311_adapter|D_INT4|42"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260825_193617_adapter|A_4DFP4|42"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter|A_4DFP4|123"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260826_104720_adapter|A_4DFP4|456"
    "results/pilot_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_164235_adapter|B_4DFP8|42"
    "results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260822_010922_adapter|B_4DFP8|123"
    "results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260822_075818_adapter|B_4DFP8|456"
    "results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter|C_STD|42"
    "results/pilot_pack4d_fp8_matrix/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260816_140828_adapter|C_STD|456"
)

log "===== 200-sample reeval sweep — wait for INT4 s42 500 first ====="
wait_gpu || { log "abort"; exit 1; }

log "===== ${#ADAPTERS[@]} adapters, 200 samples, stop+mnt=512 (~4.5h) ====="

for entry in "${ADAPTERS[@]}"; do
    IFS='|' read -r rel label seed <<< "$entry"
    if [[ ! -d "/home/yedam/HMA/HMA_Project/${rel}" ]]; then
        log "  SKIP ${label}/${seed}: adapter missing"
        continue
    fi
    tag="reeval__${label}__seed${seed}__stop_mnt512_n200"
    log_file="${OUT_DIR}/${tag}.log"
    out_json="/app/HMA_Project/results/reeval_stop_mnt512_n200/${tag}.json"
    if [[ -f "${OUT_DIR}/${tag}.json" ]]; then
        log "  SKIP ${tag} (done)"
        continue
    fi
    log "starting ${tag}"
    timeout --signal=KILL 2400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/batch_eval.py \
         --adapter_path ${rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --n_samples 200 --max_new_tokens 512 --seed ${seed} \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT (>40min)"
        docker exec ${CONTAINER} pkill -9 -f "batch_eval" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/reeval_stop_mnt512_n200 2>/dev/null
    sleep 10
done

log ""
log "===== 200-sample sweep done ====="
log "STOP here as instructed. Judgment first, then decide seed 123/456 training + GACT + 70B."
