#!/usr/bin/env bash
# Noise-floor probe: rerun OAMP K=20% at a different seed to check whether the
# 54.4% -> 44.2% shift from warmup is a real effect or single-seed volatility.
#
# Sequential (one GPU): warmup=0.0 first, then warmup=0.03. Total ~5.6h.
#
#   seed=123, mask_seed=123 (dedicated stream, INVARIANT-6)

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_seed123"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_seed() {
    local warmup="$1"
    local tag="oamp_wu${warmup}"
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method oamp --model 3B --weight_quant nf4 --seed 123 --mask_seed 123 --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio ${warmup} --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# Compare against seed=42 baseline (pilot 54.4%) and seed=42 warmup=0.03 (44.2%).
run_seed 0.0
run_seed 0.03

log "seed=123 probe complete"
