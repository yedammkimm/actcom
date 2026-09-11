#!/usr/bin/env bash
# Qwen K-sweep v3 (2026-08-17): Standard baseline + group_size sweep.
# Step 0 diagnostic confirmed:
#   - fp16 scale underflow occurs on both models but only on all-zero groups
#     (harmless). Not the root cause. → skip user's Step 2 (scale->bf16).
#   - Qwen absmax=2752 vs Llama 342 (8x), fp4 max_scale 394 vs 48.75.
#   - group_size=128 dynamic range 27520:1 on Qwen. gs sweep is the axis.
#
# Sequence:
#   1. qwen3b_standard          -- grad_norm baseline (no compression)
#   2. qwen3b_oamp_k020_gs32    -- default K, gs=32 (isolates outlier 4x tighter)
#   3. qwen3b_oamp_k020_gs16    -- default K, gs=16 (8x tighter)
#   4. qwen3b_oamp_k020_gs8     -- default K, gs=8  (16x tighter, extreme)
#
# n_train=2000 epochs=1 -> 500 opt steps (covers step 300 naive_fp4 cliff).
#
# Pass = n_nonfinite_grad_steps == 0 AND grad_norm within order-of-magnitude
# of Qwen Standard baseline AND loss trending down.

set -euo pipefail

CONTAINER=hma-container
OUT="results/qwen3b_k_sweep"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for GPU..."
    sleep 60
done
log "GPU free -- starting Qwen baseline+gs sweep"

COMMON="--mode accuracy --model qwen3B --weight_quant nf4 --seed 42 --bf16_rmsnorm \
        --task gsm8k --n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
        --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine \
        --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
        --checkpoint_every 25"

run() {
    local tag="$1"; local method="$2"; shift 2
    log "starting ${tag}  method=${method}  extra=($*)"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --method ${method} ${COMMON} $* --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# Step 1: Standard baseline (no compression -> grad_norm ground truth)
run "qwen3b_standard_500step" standard

# Step 2-4: gs sweep with default K=0.2, pack_4d_mode=fp8
run "qwen3b_oamp_k020_gs32" oamp --fp8_ratio 0.2 --group_size 32 --pack_4d_mode fp8
run "qwen3b_oamp_k020_gs16" oamp --fp8_ratio 0.2 --group_size 16 --pack_4d_mode fp8
run "qwen3b_oamp_k020_gs8"  oamp --fp8_ratio 0.2 --group_size 8  --pack_4d_mode fp8

log "Qwen sweep v3 complete -- inspect results/qwen3b_k_sweep/*.json"
