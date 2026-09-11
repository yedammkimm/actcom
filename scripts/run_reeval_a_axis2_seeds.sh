#!/usr/bin/env bash
# Complete the unified accuracy protocol for arm A across all eight runs.
#
# The §5 accuracy row uses stop_strings with max_new_tokens=512 at n=200, but
# only seeds 42, 123 and 456 were ever scored that way. The other five carry
# their in-training numbers, which use no stop strings, max_new_tokens=256, and
# — for the three original seeds — n=500 rather than n=200. Pooling those into
# one figure or table mixes three protocol differences at once.
#
# This scores the five axis-2 adapters at the unified protocol so arm A has
# eight comparable accuracy values. CLI matches run_reeval_stop_mnt512_n200.sh
# exactly. Roughly 20 minutes per adapter.
set -u
CONTAINER=hma-container
ROOT=/home/yedam/HMA/HMA_Project
OUT=results/reeval_stop_mnt512_n200
mkdir -p "${ROOT}/${OUT}"
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }
cleanup() { docker exec ${CONTAINER} pkill -f "batch_eval.py" 2>/dev/null; log "cleanup done"; }
trap cleanup EXIT INT TERM

exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock"; flock -x 9; log "GPU lock acquired"

wait_gpu() {
    local waited=0 idle=0
    while (( idle < 3 )); do
        if docker exec ${CONTAINER} pgrep -f \
             "python (run_experiment|scripts/(batch_eval|eval_perplexity))" >/dev/null 2>&1; then
            idle=0; else idle=$((idle+1)); fi
        sleep 60; waited=$((waited+60))
        (( waited > 43200 )) && { log "TIMEOUT"; return 1; }
    done
    log "  GPU free"
}

declare -A AD=(
  [789]=results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260831_224859_adapter
  [1011]=results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed1011__20260901_045636_adapter
  [2024]=results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed2024__20260901_083847_adapter
  [3033]=results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed3033__20260901_114401_adapter
  [4042]=results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260901_144026_adapter
)

log "===== arm A unified-protocol accuracy, five remaining seeds ====="
for seed in 789 1011 2024 3033 4042; do
    tag="reeval__A_4DFP4__seed${seed}__stop_mnt512_n200"
    [[ -f "${ROOT}/${OUT}/${tag}.json" ]] && { log "  ${tag} already done"; continue; }
    wait_gpu || exit 1
    log ""
    log "--- ${tag} ---"
    timeout --signal=KILL 5400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/batch_eval.py \
         --adapter_path ${AD[$seed]} \
         --weight_quant nf4 --bf16_rmsnorm \
         --n_samples 200 --max_new_tokens 512 --seed ${seed} \
         --output ${OUT}/${tag}.json" \
        > "${ROOT}/${OUT}/${tag}.log" 2>&1
    rc=$?
    (( rc == 137 || rc == 124 )) && docker exec ${CONTAINER} pkill -f "batch_eval.py" 2>/dev/null
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10
done
log ""
log "===== done — arm A now has eight accuracy values at one protocol ====="
