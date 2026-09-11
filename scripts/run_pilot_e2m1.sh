#!/bin/bash
# Phase 1 pilot for body_encoding='e2m1' rollout (2026-08-20).
# 3 arms x seed 42 on Llama-3.2-3B, GSM8K accuracy.
#
# Gates (user-defined):
#   standard : gsm8k_accuracy approx 56.4  (pipeline sanity; unaffected by body_encoding)
#   naive    : n_nonfinite=0  AND  reasonable retention
#   oamp     : >= 52.4  (INT4 baseline 54.40 minus 2pp)
#
# All arms MUST have n_nonfinite_grad_steps == 0.
set -euo pipefail

OUTDIR=results/pilot_e2m1
mkdir -p /home/yedam/HMA/HMA_Project/${OUTDIR} \
         /home/yedam/HMA/HMA_Project/logs

run_one () {
    local METHOD=$1
    local LOG=/home/yedam/HMA/HMA_Project/logs/pilot_e2m1_${METHOD}.log
    echo "=== [$(date +%H:%M:%S)] START ${METHOD} (seed 42) ===" | tee -a "${LOG}"
    docker exec hma-container bash -c "cd /app/HMA_Project && python run_experiment.py \
        --mode accuracy --method ${METHOD} --model 3B \
        --weight_quant nf4 --bf16_rmsnorm \
        --body_encoding e2m1 --pack_4d_mode fp8 \
        --seed 42 --n_train 7473 --epochs 2 --lr 2e-4 \
        --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
        --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
        --lora_dropout 0.05 --eval_samples 500 \
        --output_dir ${OUTDIR}" 2>&1 | tee -a "${LOG}"
    echo "=== [$(date +%H:%M:%S)] DONE ${METHOD} ===" | tee -a "${LOG}"
}

run_one standard
run_one naive_fp4
run_one oamp

echo "=== [$(date +%H:%M:%S)] all three arms complete ==="
ls -lh /home/yedam/HMA/HMA_Project/${OUTDIR}/
