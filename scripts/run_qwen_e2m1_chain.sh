#!/usr/bin/env bash
# Qwen 3B accuracy chain with E2M1 body encoding (2026-08-24).
# Replicates qwen3b_skip_check config exactly, adding --body_encoding e2m1.
#
# Expected duration: ~3.5h per arm; total ~7h.
#
# Prior INT4 results (qwen3b_skip_check):
#   naive_fp4 : acc 68.2  n_nonfin=21   loss 1.717
#   oamp      : acc 69.0  n_nonfin= 0   loss 3.208
#
# Standard 66.0 (from pilot_qwen3b_pack4d_fp8) is body_encoding-independent.

set -u

CONTAINER=hma-container
OUT=results/qwen3b_e2m1
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

COMMON="--mode accuracy --model qwen3B --weight_quant nf4 --bf16_rmsnorm \
--body_encoding e2m1 --pack_4d_mode fp8 --seed 42 --task gsm8k \
--n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit \
--scheduler cosine --warmup_ratio 0.0 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
--eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 \
--output_dir ${OUT}"

run_arm() {
    local method="$1"
    local extras="$2"
    local tag="${method}_seed42"
    local log_file="${LOG_HOST}/${tag}.log"
    log "===================================================="
    log "starting ${tag}  method=${method}"
    log "  extras: ${extras}"
    log "===================================================="
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --method ${method} ${extras} ${COMMON}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 60
}

log ""
log "===== Qwen 3B E2M1 chain begin ====="

# DCR (oamp) first — replicates 69.0 baseline
run_arm oamp "--fp8_ratio 0.2 --group_size 128 --mask_seed 42"

# Uniform-FP4 (naive_fp4) — replicates 68.2 baseline
# n_nonfin=21 under INT4; check if E2M1 lowers this count.
run_arm naive_fp4 "--fp8_ratio 0.0 --group_size 128"

log ""
log "===== Qwen 3B E2M1 chain complete ====="
