#!/usr/bin/env bash
# Follow-up chain (2026-08-25):
#   (a) Qwen 3B OAMP K=0.4 + E2M1 seed 42 (~3.5h)
#       Tests §6 hypothesis: does raising anchor ratio make Qwen WORSE?
#         K=0.2  → OAMP 76.00% (mnt=512 reeval)
#         K=0.4  → ?  (this run)
#         If worse: anchor is architecture-dependent, story closes
#         If better: hypothesis rejected, alternative needed
#
#   (b) 3B Uniform-FP4 (naive_fp4) E2M1 seed 789 (~3.5h)
#       Fills the n=3 vs n=4 gap in §4.3: DCR has seeds 42/123/456/789 but
#       Uniform-FP4 only has seeds 42/123/456. This run yields Uniform-FP4
#       seed 789 for symmetric n=4 statistics.
#
# Waits for any current run to end before starting.

set -u
CONTAINER=hma-container
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== Post-morning chain queued ====="

log "waiting for GPU..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py|python.*batch_eval.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

# --------------------------------------------------------------------------
# (a) Qwen OAMP K=0.4 E2M1 seed 42
# --------------------------------------------------------------------------
OUT_A=results/qwen3b_e2m1_k040
LOG_HOST_A=/home/yedam/HMA/HMA_Project/${OUT_A}
mkdir -p ${LOG_HOST_A}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT_A}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT_A} 2>/dev/null

COMMON_A="--mode accuracy --model qwen3B --weight_quant nf4 --bf16_rmsnorm \
--body_encoding e2m1 --pack_4d_mode fp8 --seed 42 --task gsm8k \
--n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit \
--scheduler cosine --warmup_ratio 0.0 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
--eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 \
--output_dir ${OUT_A}"

log ""
log "===== (a) Qwen 3B OAMP K=0.4 E2M1 seed 42 ====="
tag_a="oamp_k040_seed42"
log_a="${LOG_HOST_A}/${tag_a}.log"
docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --method oamp --fp8_ratio 0.4 --group_size 128 --mask_seed 42 \
     ${COMMON_A}" \
    > "${log_a}" 2>&1
log "  (a) exit=$?"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT_A} 2>/dev/null
sleep 60

# --------------------------------------------------------------------------
# (b) 3B naive_fp4 E2M1 seed 789 -- match pilot_e2m1 config exactly
# --------------------------------------------------------------------------
OUT_B=results/pilot_e2m1_seeds
LOG_HOST_B=/home/yedam/HMA/HMA_Project/${OUT_B}
mkdir -p ${LOG_HOST_B}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT_B}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT_B} 2>/dev/null

log ""
log "===== (b) 3B naive_fp4 E2M1 seed 789 ====="
tag_b="naive_fp4_seed789"
log_b="${LOG_HOST_B}/${tag_b}.log"
docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --mode accuracy --method naive_fp4 --model 3B \
     --weight_quant nf4 --bf16_rmsnorm \
     --body_encoding e2m1 --pack_4d_mode fp8 \
     --seed 789 --task gsm8k \
     --n_train 7473 --epochs 2 --lr 2e-4 \
     --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
     --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
     --lora_dropout 0.05 --eval_samples 500 \
     --output_dir ${OUT_B}" \
    > "${log_b}" 2>&1
log "  (b) exit=$?"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT_B} 2>/dev/null

log ""
log "===== Follow-up chain complete ====="
