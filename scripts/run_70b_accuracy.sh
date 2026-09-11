#!/usr/bin/env bash
# 70B accuracy proof-of-concept (2026-08-18).
#
# Design (per user directive):
#   L=512 (max_seq_len, standard fits comfortably)
#   n_train=2000 epochs=1 → 500 opt steps
#   lr=5e-5 (8B/70B convention; different from 3B's 2e-4)
#   eval 200 samples
#   mem_abort_gb=80 (container=95 GB, 15 GB safety margin)
#
# Method order: standard → naive_fp4 → oamp
# Skip logic: if the JSON already exists for a method, skip it (safe restart).

set -u

CONTAINER=hma-container
OUT=results/70b_accuracy
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

COMMON="--mode accuracy --model 70B --weight_quant nf4 --bf16_rmsnorm \
--pack_4d_mode fp8 --seed 42 --task gsm8k \
--n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 5e-5 --optimizer paged_adamw8bit \
--scheduler cosine --warmup_ratio 0.0 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
--eval_samples 200 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 --mem_abort_gb 80 \
--output_dir ${OUT}"

extras_standard=""
extras_naive_fp4=""
extras_oamp="--fp8_ratio 0.2 --group_size 128 --mask_seed 42"

already_done() {
    local method="$1"
    ls ${LOG_HOST}/accuracy__${method}__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__*.json >/dev/null 2>&1
}

run_one() {
    local method="$1"
    local ev="extras_${method}"
    local extras="${!ev}"
    local tag="${method}_seed42"
    local log_file="${LOG_HOST}/${tag}.log"

    if already_done "${method}"; then
        log "  SKIP ${tag} (JSON already exists)"
        return 0
    fi

    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --method ${method} ${extras} ${COMMON}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    return ${rc}
}

log "===== 70B accuracy chain begin ====="
log "container memory limit = 95 GB, watchdog = 80 GB"
log ""

for method in standard naive_fp4 oamp; do
    run_one "${method}"
    sleep 30
done

log "===== 70B accuracy chain complete ====="
