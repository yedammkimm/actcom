#!/usr/bin/env bash
# Overnight chain (2026-08-17): Qwen skip validation + Llama matrix completion.
#
# The chain runs 6 experiments sequentially after the currently-running
# Qwen uniform_fp8 (PID 539742) finishes. Any exit code is tolerated so
# a single failure does not stop the night.
#
# Order (Qwen skip validation first, Llama matrix at the tail):
#   [already running]  Qwen uniform_fp8 s42          (~40 min)
#   1. Qwen OAMP s42            fp8_ratio=0.2, gs=128 (~40 min)
#   2. Qwen naive_fp4 s42                              (~40 min)
#   3. Llama naive_fp4 s123     3736 steps            (~3.5 h)
#   4. Llama naive_fp4 s456     3736 steps            (~3.5 h)
#   5. Llama uniform_fp8 s42    3736 steps            (~3.5 h)
#   6. Llama random_mixed s42   3736 steps            (~3.5 h)
#
# Total ~16h. All runs use the new nonfinite-grad skip logic
# (run_experiment.py::_run_accuracy, 2026-08-17).

set -u  # NO -e: chain must tolerate individual failures.

CONTAINER=hma-container
QWEN_OUT=results/qwen3b_skip_check
LLAMA_OUT=results/llama_skip_matrix

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${QWEN_OUT} /app/HMA_Project/${LLAMA_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${QWEN_OUT} /app/HMA_Project/${LLAMA_OUT} 2>/dev/null
mkdir -p /home/yedam/HMA/HMA_Project/${QWEN_OUT} /home/yedam/HMA/HMA_Project/${LLAMA_OUT}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited % 600 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

# run <tag> <out_dir> -- <cli args including --method ...>
run() {
    local tag="$1"; shift
    local out="$1"; shift
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py $* --output_dir ${out}" \
        > "/home/yedam/HMA/HMA_Project/${out}/${tag}.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "  ${tag} exited rc=${rc} (chain continues)"
    fi
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${out} 2>/dev/null
    log "  ${tag} done"
}

QWEN_COMMON="--mode accuracy --model qwen3B --weight_quant nf4 --seed 42 --bf16_rmsnorm \
--task gsm8k --n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine \
--warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
--grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 --pack_4d_mode fp8"

LLAMA_COMMON="--mode accuracy --model 3B --weight_quant nf4 --bf16_rmsnorm \
--task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine \
--warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
--grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 100 --pack_4d_mode fp8"

log "===== overnight chain begin ====="
log "waiting for the currently-running Qwen uniform_fp8 to finish..."
wait_gpu

# 1. Qwen OAMP
run qwen_oamp_skip           ${QWEN_OUT}  --method oamp --fp8_ratio 0.2 --group_size 128 --mask_seed 42 ${QWEN_COMMON}
wait_gpu

# 2. Qwen naive_fp4
run qwen_naive_fp4_skip      ${QWEN_OUT}  --method naive_fp4 ${QWEN_COMMON}
wait_gpu

# 3. Llama naive_fp4 s123
run llama_naive_fp4_s123     ${LLAMA_OUT} --method naive_fp4 --seed 123 ${LLAMA_COMMON}
wait_gpu

# 4. Llama naive_fp4 s456
run llama_naive_fp4_s456     ${LLAMA_OUT} --method naive_fp4 --seed 456 ${LLAMA_COMMON}
wait_gpu

# 5. Llama uniform_fp8 s42
run llama_uniform_fp8_s42    ${LLAMA_OUT} --method uniform_fp8 --seed 42 ${LLAMA_COMMON}
wait_gpu

# 6. Llama random_mixed s42
run llama_random_mixed_s42   ${LLAMA_OUT} --method random_mixed --seed 42 --mask_seed 42 ${LLAMA_COMMON}
wait_gpu

log "===== overnight chain complete ====="
