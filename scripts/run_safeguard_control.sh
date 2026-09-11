#!/usr/bin/env bash
# The missing control for the non-finite safeguard claim.
#
# §3.6 asserted that four-bit training on Qwen2.5-3B "collapses to 0.2% without
# the safeguard and recovers to 68.2% with it". The two runs behind that
# sentence differ by more than the safeguard: 3,736 optimizer steps against 500,
# and the earlier one predates the counter, so its non-finite count is
# unrecorded rather than zero. This run supplies the matched control.
#
# Config replicates results/qwen3b_skip_check/accuracy__naive_fp4__Qwen2.5-3B
# field for field -- body_encoding int4 (the field was unset there, and INT4 was
# the only body encoding at that commit), pack_4d_mode fp8, n_train 2000,
# epochs 1, lr 2e-4, paged_adamw8bit, seed 42, eval_samples 500,
# eval_max_new_tokens 256, checkpoint_every 25 -- with --no_nonfinite_skip as
# the single change. Only output_dir differs otherwise.
#
# Mechanism, and why the prediction is nearly certain. One inf in any LoRA grad
# makes clip_grad_norm_'s global norm inf, so the clip coefficient is 0 and the
# product 0 * inf is NaN. NaN enters Adam's first and second moments and the
# affected parameters never recover. The guarded run hit 21 non-finite steps in
# 500, so the first should arrive early and the loss trace should go NaN within
# minutes rather than at the end.
#
# Judgement, fixed before the data arrives:
#   loss goes NaN / accuracy collapses  -> the safeguard is the cause. §3.6
#       recovers its causal claim and cites this pair instead of the
#       length-confounded one.
#   run completes with normal accuracy  -> clip_grad_norm_ does not behave as
#       the mechanism paragraph says, which is a more interesting result and
#       means §3.6 has to be rewritten around what actually happens.
#   run completes but degraded          -> report the accuracy delta
#       descriptively; the safeguard helps but is not the whole story.
#
# The counter still increments with the guard off (run_experiment.py counts
# before it acts), so this run also reports how many steps would have been
# skipped -- directly comparable to the 21 of the guarded run.
#
# _save_partial_adapter fires on NaNError, so an adapter is kept either way.
# Runtime: ~25 min training + ~18 min eval.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/qwen_safeguard_control
OUT_HOST=${ROOT_HOST}/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
# Hazard type 1: the container creates these as root and the host shell then
# cannot open its own redirect targets. chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p "${OUT_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 4: timeout kills the docker exec client, not the container-side job.
cleanup() {
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    log "cleanup: container-side jobs killed if any remained"
}
trap cleanup EXIT INT TERM

# Hazard type 3: order queued chains rather than racing them.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

log "===== Qwen safeguard control -- --no_nonfinite_skip ====="
timeout --signal=KILL 7200 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --mode accuracy --method naive_fp4 --model_id Qwen/Qwen2.5-3B-Instruct \
     --weight_quant nf4 --bf16_rmsnorm \
     --body_encoding int4 --pack_4d_mode fp8 \
     --no_nonfinite_skip \
     --group_size 128 --min_numel 1024 --task gsm8k --seed 42 \
     --n_train 2000 --epochs 1 --lr 2e-4 \
     --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
     --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
     --lora_dropout 0.05 --lora_r 16 --lora_alpha 32 \
     --eval_samples 500 --eval_max_new_tokens 256 --eval_fewshot 8 \
     --checkpoint_every 25 \
     --output_dir ${OUT}" \
    > "${OUT_HOST}/control.log" 2>&1
rc=$?
if (( rc == 137 || rc == 124 )); then
    log "  TIMEOUT -- killing the container-side process"
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    sleep 10
fi
log "  exit=${rc}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

log ""
log "===== done ====="
log "  guarded reference: 21/500 non-finite steps, 68.2% (341/500)"
log "  read: n_nonfinite_grad_steps, the loss trace in the checkpoints sidecar,"
log "        and gsm8k_accuracy."
