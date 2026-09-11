#!/usr/bin/env bash
# Qwen2.5-3B K-sweep with uniform_fp8 upper-bound probe first (2026-08-17 v2).
#
# Sequence:
#   0. uniform_fp8              — establishes whether FP8 alone is enough. If this
#                                 fails to learn, no K adjustment can save OAMP
#                                 and 1-3 are informationally void.
#   1. OAMP K=0.4  gs=128       — is doubling anchor ratio enough?
#   2. OAMP K=0.6  gs=128       — is tripling anchor ratio enough?
#   3. OAMP K=0.2  gs=32        — does finer grouping isolate outliers?
#
# n_train=2000 epochs=1 → 500 opt steps. Covers Qwen naive_fp4's step 300 cliff
# and OAMP's step 1200 degradation onset (well beyond both).
#
# Pass/fail decision (from result JSON, added 2026-08-17):
#   result['results']['n_nonfinite_grad_steps'] == 0
#     AND grad_norm_trace shows no divergent trend
#     AND step_losses are non-monotonically decreasing over last 100 steps
#
# Absolute loss threshold is NOT used — 500 steps is too short to reach 0.21 that
# Standard achieves on the full 3736-step run.

set -euo pipefail

CONTAINER=hma-container
OUT="results/qwen3b_k_sweep"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Wait for any prior run_experiment.py to finish
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for GPU..."
    sleep 60
done
log "GPU free -- starting Qwen K-sweep v2 (uniform_fp8 upper bound first)"

COMMON="--mode accuracy --model qwen3B --weight_quant nf4 --seed 42 --bf16_rmsnorm \
        --task gsm8k --n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
        --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine \
        --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 \
        --checkpoint_every 25 --pack_4d_mode fp8"

run() {
    local tag="$1"; local method="$2"; shift 2
    log "starting ${tag}  method=${method}  extra=($*)"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --method ${method} ${COMMON} $* --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# Step 0: FP8 upper bound. If uniform_fp8 fails on Qwen, K-sweep is void.
run "qwen3b_uniform_fp8" uniform_fp8

# Step 1-3: OAMP K sweep
run "qwen3b_oamp_k040_gs128" oamp --fp8_ratio 0.4 --group_size 128
run "qwen3b_oamp_k060_gs128" oamp --fp8_ratio 0.6 --group_size 128
run "qwen3b_oamp_k020_gs32"  oamp --fp8_ratio 0.2 --group_size 32

log "Qwen K-sweep v2 complete -- inspect results/qwen3b_k_sweep/*.json"
log "Decision: n_nonfinite_grad_steps==0 + no divergent grad_norm + loss trending down."
