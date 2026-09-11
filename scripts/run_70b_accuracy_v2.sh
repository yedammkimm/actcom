#!/usr/bin/env bash
# 70B accuracy chain v2 (2026-08-18).
#
# Flow:
#   1. Wait for the standard_seed42 python to finish (already started at 12:11).
#   2. Run BASE eval (LoRA=off, 200 samples) — required to interpret the
#      Standard/Naive/OAMP accuracies. 70B-Instruct has strong zero-shot
#      GSM8K; if base ≥ fine-tuned, all method comparisons are moot.
#   3. Run naive_fp4 (skip if JSON exists).
#   4. Run oamp (skip if JSON exists).
#
# NOTE: run_experiment.py now saves the LoRA adapter BEFORE eval, so an
# eval-OOM on the naive/oamp arms costs eval time only, not the training.

set -u

CONTAINER=hma-container
OUT=results/70b_accuracy
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}
UNSLOTH_ID="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

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

COMMON="--mode accuracy --model 70B --weight_quant nf4 --bf16_rmsnorm \
--pack_4d_mode fp8 --seed 42 --task gsm8k \
--n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 5e-5 --optimizer paged_adamw8bit \
--scheduler cosine --warmup_ratio 0.0 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
--eval_samples 200 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 --mem_abort_gb 80 \
--output_dir ${OUT}"

extras_naive_fp4=""
extras_oamp="--fp8_ratio 0.2 --group_size 128 --mask_seed 42"

already_done_accuracy() {
    local method="$1"
    ls ${LOG_HOST}/accuracy__${method}__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__*.json >/dev/null 2>&1
}

base_eval_done() {
    ls ${LOG_HOST}/base_eval_*.json >/dev/null 2>&1
}

run_accuracy() {
    local method="$1"
    local ev="extras_${method}"
    local extras="${!ev}"
    local tag="${method}_seed42"
    local log_file="${LOG_HOST}/${tag}.log"

    if already_done_accuracy "${method}"; then
        log "  SKIP ${tag} (JSON exists)"
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

run_base_eval() {
    local tag="base_eval_70b"
    local log_file="${LOG_HOST}/${tag}.log"

    if base_eval_done; then
        log "  SKIP ${tag} (JSON exists)"
        return 0
    fi

    log "starting ${tag}  (LoRA=off, 200 samples, ${UNSLOTH_ID})"
    docker exec \
        -e BASE_EVAL_MODEL="${UNSLOTH_ID}" \
        -e BASE_EVAL_N=200 \
        -e BASE_EVAL_SEED=42 \
        -e BASE_EVAL_OUT="/app/HMA_Project/${OUT}/base_eval_70b_gsm8k_n200_$(date +%Y%m%d_%H%M%S).json" \
        ${CONTAINER} bash -c "cd /app/HMA_Project && python scripts/base_eval.py" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    return ${rc}
}

log "===== 70B accuracy chain v2 begin ====="
log "waiting for standard python to finish..."
wait_gpu

log ""
log "--- BASE eval (LoRA=off) ---"
run_base_eval
sleep 30

log ""
log "--- Compressed methods ---"
for method in naive_fp4 oamp; do
    run_accuracy "${method}"
    sleep 30
done

log "===== 70B accuracy chain v2 complete ====="
