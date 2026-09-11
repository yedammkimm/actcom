#!/usr/bin/env bash
# 3-seed training verification for pack_4d_mode=fp8.
# Waits for the running memory-mode job to release the GPU, then runs
# OAMP + fp8 (dim=4 override) at seed 42 / 123 / 456 sequentially.
# Target: seed std should drop from 6.6pp (non-SR baseline) to ~1pp
# (α-level stability) if the head-view precision hypothesis holds.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_pack4d_fp8"

# Wait for any running run_experiment (memory or accuracy) to finish.
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for prior run_experiment.py to finish..."
    sleep 120
done
log "GPU free -- starting 3-seed chain for pack_4d_mode=fp8"

docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

run_one() {
    local seed="$1"
    local tag="oamp_fp8_s${seed}"
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method oamp --mask_seed ${seed} --pack_4d_mode fp8 --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

run_one 42
run_one 123
run_one 456

log "pack_4d_mode=fp8 3-seed chain complete"
