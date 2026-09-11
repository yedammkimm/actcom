#!/usr/bin/env bash
# 4-method pilot re-run under pack_4d_mode=fp8.
# Rationale: original pilot (naive_fp4=32.2 / random_mixed=50.8 / OAMP=54.4 /
# uniform_fp8=55.4) was CONFOUNDED because pack_4d_mode='fp4' left the
# gradient-corrupting head-view compression in every method except uniform_fp8.
# We must re-run to know whether the "cliff" is bit-precision or head-view damage.
#
# Order:
#   1. Standard seed 456           -- brings Standard n to 3 (retention denom).
#   2. naive_fp4    seed 42        -- crucial: does 32.2 recover with 4-D FP8?
#   3. uniform_fp8  seed 42        -- reference (should match prior 55.4).
#   4. random_mixed seed 42 mask=42 -- bilevel body under fp8 4-D.
# OAMP already has n=3 at pack_4d_mode=fp8; not repeated.
# Total: ~11 h sequential.

set -euo pipefail

CONTAINER=hma-container
OUT="results/pilot_pack4d_fp8_matrix"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"
docker exec ${CONTAINER} chown -R 4051:4051 "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Wait for any competing run_experiment (the s456 fp8 accuracy run above finishes ~10:25 UTC).
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py" > /dev/null 2>&1; do
    log "waiting for prior run_experiment.py to finish..."
    sleep 120
done
log "GPU free -- starting confound re-run matrix"

run_one() {
    local method="$1"; local seed="$2"; local extra="${3:-}"
    local tag="${method}_s${seed}"
    log "starting ${tag}  (${extra})"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode accuracy --method ${method} ${extra} --pack_4d_mode fp8 --model 3B --weight_quant nf4 --seed ${seed} --bf16_rmsnorm --task gsm8k --n_train 7473 --epochs 2 --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 --lr 2e-4 --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 --eval_samples 500 --eval_fewshot 8 --eval_max_new_tokens 256 --checkpoint_every 100 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# 1. Standard seed 456 (pack_4d_mode is inert here — no pack context; harmless flag)
run_one standard 456
# 2. naive_fp4 seed 42 — critical question: does 32.2 % survive without head-view FP4 damage?
run_one naive_fp4 42
# 3. uniform_fp8 seed 42 — should match prior 55.4 % (4-D already FP8 under legacy).
run_one uniform_fp8 42
# 4. random_mixed seed 42 with mask_seed 42
run_one random_mixed 42 "--mask_seed 42"

log "confound re-run matrix complete"
