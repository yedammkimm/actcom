#!/usr/bin/env bash
# 3B naive 4-bit E2M1 pack_4d=fp4 seeds 123 and 456 (2026-08-26).
# Confirms whether Exp 2's 57.60% (seed 42) is real or a seed-42 fluke.
# Same config as pilot_e2m1 (7473 x 2ep).
#
# Decision from mean of {42, 123, 456}:
#   mean ~= 57  ->  4D FP8 unnecessary. Option A. Rewrite section 3 method.
#   mean ~= 52-55 ->  Comparable to DCR. FP8 is safety net, not essential.
#   mean ~= 48  ->  57.60 was outlier. Keep current structure.

set -u
CONTAINER=hma-container
OUT=results/naive4bit_e2m1
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

COMMON="--mode accuracy --method naive_fp4 --model 3B \
--weight_quant nf4 --bf16_rmsnorm \
--body_encoding e2m1 --pack_4d_mode fp4 \
--group_size 128 --task gsm8k \
--n_train 7473 --epochs 2 --lr 2e-4 \
--batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
--optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
--lora_dropout 0.05 --eval_samples 500 \
--output_dir ${OUT}"

run_seed() {
    local seed="$1"
    local tag="naive_4dfp4_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"
    log ""
    log "--- ${tag} ---"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --seed ${seed} ${COMMON}" \
        > "${log_file}" 2>&1
    log "  ${tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 60
}

log ""
log "===== 3B naive 4-bit E2M1 pack_4d=fp4  seed 123 & 456 ====="
log "  baseline seed 42 = 57.60%; verifying reproducibility"

run_seed 123
run_seed 456

log ""
log "===== Verification chain complete ====="
