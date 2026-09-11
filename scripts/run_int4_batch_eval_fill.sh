#!/usr/bin/env bash
# INT4 seed 123 + 456 → batch_eval at stop+mnt=512, n=200
# Matches the A_4DFP4/B_4DFP8/C_STD reeval protocol so INT4 vs A can be
# compared apples-to-apples on the §5 accuracy row.
# INT4 seed 42 already scored at stop+mnt=512 n=200 (45.5%).
# Waits for GPU idle before running.

set -u
CONTAINER=hma-container
OUT_DIR=/home/yedam/HMA/HMA_Project/results/reeval_stop_mnt512_n200
mkdir -p ${OUT_DIR}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=54000
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

ADAPTERS=(
    "results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260831_194822_adapter|D_INT4|123"
    "results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260831_230549_adapter|D_INT4|456"
)

log "===== INT4 seed 123 + 456 batch_eval (stop+mnt=512, n=200) ====="
wait_gpu || { log "abort"; exit 1; }

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
        log "  SKIP ${tag} (done)"; continue
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
        log "  ${tag} TIMEOUT"
        docker exec ${CONTAINER} pkill -9 -f "batch_eval" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/reeval_stop_mnt512_n200 2>/dev/null
    sleep 15
done

log ""
log "===== INT4 seeds 123/456 reeval done. Combine with seed 42 (45.5%) for 3-seed accuracy row. ====="
