#!/usr/bin/env bash
# 3B INT4 body + pack_4d=fp4 seed 42 (2026-08-31).
#
# Purpose: verify the missing INT4 baseline referenced as "48.2 ± 6.64" in
# the §5 threshold argument. This exact config (body=int4, pack_4d=fp4) has
# never been run through the current code path (E2M1 default was added on
# 08-20 and never re-verified with INT4). If accuracy ≈ 48%, the threshold
# claim (cos 0.36 → 48; cos 0.58 → 55; cos 0.99 → 55) is confirmed.
#
# Config matches 08-25 naive_fp4_e2m1 seed 42 (57.60%) except body='int4'.

set -u
CONTAINER=hma-container
OUT=results/naive4bit_int4
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

TAG="naive_4dfp4_int4_s42"
LOG_FILE="${LOG_HOST}/${TAG}.log"

log "===== 3B naive INT4 body + pack_4d=fp4 seed 42 ====="
log "  target ~= 48% (threshold argument first datapoint)"
log "  log ${LOG_FILE}"

# 8h wall-time cap so a hang cannot swallow the GPU indefinitely.
timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --mode accuracy --method naive_fp4 --model 3B \
     --weight_quant nf4 --bf16_rmsnorm \
     --body_encoding int4 --pack_4d_mode fp4 \
     --group_size 128 --task gsm8k --seed 42 \
     --n_train 7473 --epochs 2 --lr 2e-4 \
     --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
     --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
     --lora_dropout 0.05 --eval_samples 500 \
     --checkpoint_every 10 --mem_abort_gb 70 \
     --output_dir ${OUT}" \
    > "${LOG_FILE}" 2>&1

RC=$?
if (( RC == 137 || RC == 124 )); then
    log "  ${TAG} TIMEOUT (>8h); killing"
    docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
fi
log "  ${TAG} exit=${RC}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
