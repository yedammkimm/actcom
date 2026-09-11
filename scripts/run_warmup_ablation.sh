#!/usr/bin/env bash
# warmup=3% ablation: does the paper's Table 19 warmup fix γ's Q1 instability?
# Sequential OAMP -> Standard so both use the same allocator/session.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_warmup"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_warmup() {
    local method="$1"
    log "starting warmup ablation: ${method}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method ${method} --model 3B --weight_quant nf4 --seed 42 --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.03 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${method}_warmup_run.log" 2>&1
    log "  ${method} warmup done"
}

# OAMP first — this decides whether all-run condition changes.
run_warmup oamp
# Standard second so the same warmup baseline is available for comparison.
run_warmup standard

log "warmup chain complete"
