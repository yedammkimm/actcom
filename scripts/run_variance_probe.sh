#!/usr/bin/env bash
# Two-arm noise-floor probe:
#   1. Standard seed=123        - is variance a pipeline issue or method issue?
#   2. OAMP seed=456 wu=0.0     - third OAMP data point (n>=3 for range check)
# Sequential (one GPU), ~5.6h total.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_var"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_one() {
    local method="$1"; local seed="$2"; local extra="${3:-}"
    local tag="${method}_s${seed}"
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method ${method} ${extra} --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# 1. Standard seed=123 -- pipeline sanity
run_one standard 123
# 2. OAMP seed=456 -- third data point for OAMP variance range
run_one oamp 456 "--mask_seed 456"

log "variance probe complete"
