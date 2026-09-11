#!/usr/bin/env bash
# SR (stochastic rounding) probe:
#   1. OAMP seed=42  --stochastic_rounding (sr_seed=42 via fallback)
#   2. OAMP seed=123 --stochastic_rounding (sr_seed=123 via fallback)
# Compare against non-SR OAMP baseline (n=3, std=6.6pp) at same seeds.
# Sequential (one GPU), ~5.6h total.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_sr"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_one() {
    local seed="$1"
    local tag="oamp_sr_s${seed}"
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method oamp --mask_seed ${seed} --stochastic_rounding --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

run_one 42
run_one 123

log "SR probe complete"
