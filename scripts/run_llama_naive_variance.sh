#!/usr/bin/env bash
# Independent of Qwen diagnosis: fills Llama Naive-FP4 s123/s456 for n=3 variance.
set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_pack4d_fp8_matrix"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Wait for anything currently running
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for GPU..."
    sleep 60
done
log "GPU free -- starting Llama Naive-FP4 variance chain"

run_naive_llama() {
    local seed="$1"
    local tag="naive_fp4_s${seed}"
    log "starting ${tag} (Llama 3B)"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method naive_fp4 --pack_4d_mode fp8 --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

run_naive_llama 123
run_naive_llama 456

log "Llama Naive-FP4 variance chain complete"
