#!/usr/bin/env bash
# Pilot accuracy chain: 5 methods × 3B, GSM8K 7473×2ep, Arm 4 dtype.
# Each method runs in its own process for allocator isolation.
#
# Scheduler policy: cosine with 0% warmup — matches legacy benchmark_multiseed
# (paper Table 19's "3% warmup" is treated as a documentation error, see
# repo/hma_project.md). --legacy_repro is NOT used because it would also
# force lora_dropout=0.0; the pilot uses production dropout 0.05.
#
# Each run wall time (3B, 3736 steps, PagedAdamW8bit): ~4h → chain ~20h.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"

METHODS=(standard naive_fp4 uniform_fp8 random_mixed oamp)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_pilot() {
    local method="$1"
    local extra=""
    if [[ "${method}" == "random_mixed" ]]; then
        # Dedicated mask stream, isolated from global RNG (INVARIANT-6).
        extra="--mask_seed 42"
    fi
    log "starting pilot: ${method}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py \
        --mode accuracy --method ${method} ${extra} \
        --model 3B --weight_quant nf4 --seed 42 --bf16_rmsnorm \
        --task gsm8k --n_train 7473 --epochs 2 \
        --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
        --lr 2e-4 --optimizer paged_adamw8bit \
        --scheduler cosine --warmup_ratio 0.0 \
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --grad_clip 1.0 \
        --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
        --checkpoint_every 100 \
        --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${method}_run.log" 2>&1
    log "  ${method} done"
}

for m in "${METHODS[@]}"; do
    run_pilot "${m}"
done

log "pilot chain complete"
