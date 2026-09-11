#!/usr/bin/env bash
# 70B naive_fp4 body=e2m1 pack_4d_mode=fp4 seed 42 (2026-08-26 / 2026-08-31 rev).
# Purpose: verify Option A at 70B scale.
#
# Baselines (70B, adapter reeval, 200 samples, mnt=512):
#   Standard              89.00
#   naive_fp4 pack_4d=fp8 86.50 (int4 body)
#
# Decision from this run's gsm8k_accuracy:
#   ~= 88   ->  Option A confirmed at 70B; drop pack_4d_mode uplift entirely.
#   >> 86.5 ->  Option A safe; pack_4d=fp4 no regression.
#   << 84   ->  scale-specific fragility; keep pack_4d=fp8 for 70B.
#
# 2026-08-31 changes vs previous attempt (which OOM-hung at step ~175):
#   * checkpoint_every 25 -> 10   (finer trace if hang recurs)
#   * mem_abort_gb 80 -> 70       (leave 10 GB safety inside 80 GB cgroup)
#   * eval watchdog now active in oamp/evaluate.py (aborts on eval OOM)
#   * outer `timeout --signal=KILL 28800` (8h wall cap, exceeds ETA 5.5h)

set -u
CONTAINER=hma-container
OUT=results/70b_naive_e2m1_4dfp4
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

TS=$(date +%Y%m%d_%H%M%S)
TAG="naive_4dfp4_e2m1_s42_${TS}"
LOG_FILE="${LOG_HOST}/${TAG}.log"

log "===== 70B naive_fp4 E2M1 pack_4d=fp4 seed 42 ====="
log "  target ~= 88% (Option A)   floor 86.5% (prior naive_fp4 pack_4d=fp8)"
log "  log ${LOG_FILE}"
log "  checkpoint_every=10  mem_abort_gb=70  wall_timeout=8h"

timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --mode accuracy --method naive_fp4 --model 70B \
     --weight_quant nf4 --bf16_rmsnorm \
     --body_encoding e2m1 --pack_4d_mode fp4 \
     --group_size 128 --task gsm8k --seed 42 \
     --n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
     --max_seq_len 512 --lr 5e-5 \
     --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
     --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
     --eval_samples 200 --eval_fewshot 8 --eval_max_new_tokens 256 \
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

log ""
log "===== chain complete ====="
