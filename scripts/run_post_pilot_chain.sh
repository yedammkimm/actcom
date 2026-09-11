#!/usr/bin/env bash
# Post-pilot chain (queued behind pilot_pack4d_fp8_matrix):
#   Phase 1  Qwen2.5-3B, γ + pack_4d_mode=fp8, seed 42
#            method: standard / naive_fp4 / oamp
#            Paper's cross-architecture Table 8 shows Qwen has +4.9pp OAMP
#            advantage over Naive-FP4 (unlike Llama where they tie).
#            We must confirm whether the max-abs anchor benefit survives
#            under the pack_4d_mode=fp8 discovery.
#   Phase 2  Naive-FP4 (Llama 3B) seed 123 / 456 with pack_4d_mode=fp8
#            Fills the variance side of the Llama comparison — OAMP has
#            std=1.59pp (n=3); we need Naive-FP4 std to argue any residual
#            stability advantage.
# Sequential (single GPU), waits for prior run_experiment to finish.

set -euo pipefail

CONTAINER=hma-container
OUT_QWEN="results/pilot_qwen3b_pack4d_fp8"
OUT_NAIVE_EXT="results/pilot_pack4d_fp8_matrix"    # add to matrix dir
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT_QWEN}"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT_NAIVE_EXT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT_QWEN}" "/app/HMA_Project/${OUT_NAIVE_EXT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for prior run_experiment.py to finish..."
    sleep 180
done
log "GPU free -- starting post-pilot chain"

run_qwen() {
    local method="$1"; local extra="${2:-}"
    local tag="qwen3b_${method}_s42"
    log "starting ${tag}  (${extra})"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method ${method} ${extra} --pack_4d_mode fp8 --model qwen3B --weight_quant nf4 --seed 42 --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT_QWEN}" \
        > "/home/yedam/HMA/HMA_Project/${OUT_QWEN}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

run_naive_llama() {
    local seed="$1"
    local tag="naive_fp4_s${seed}"
    log "starting ${tag} (Llama 3B extension)"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method naive_fp4 --pack_4d_mode fp8 --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT_NAIVE_EXT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT_NAIVE_EXT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# Phase 1: Qwen 3-way (highest information yield)
run_qwen standard
run_qwen naive_fp4
run_qwen oamp

# Phase 2: Naive-FP4 Llama variance completion
run_naive_llama 123
run_naive_llama 456

log "post-pilot chain complete"
