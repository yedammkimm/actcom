#!/usr/bin/env bash
# Post-PPL chain: waits for the current mechanism_then_ppl chain to release the
# GPU, then runs the §5 threshold argument's first datapoint —
# naive_fp4 + body=int4 + pack_4d=fp4 seed 42.
#
# Decision after this run:
#   ~48%  ->  threshold hypothesis holds. Add seeds 123 + 456 (7h more).
#   ~55%  ->  threshold hypothesis falsified. Skip 123/456 and go to 70B.

set -u
CONTAINER=hma-container
OUT=results/naive4bit_int4
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=7200   # 2h wait cap for PPL to finish
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/eval_perplexity|gradient_error_probe)" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU (${waited}s); aborting"
            return 1
        fi
        if (( waited % 300 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

TAG="naive_4dfp4_int4_s42"
LOG_FILE="${LOG_HOST}/${TAG}.log"

log "===== post-PPL chain: waiting for GPU ====="
wait_gpu || { log "abort"; exit 1; }

log ""
log "===== 3B naive_fp4 int4 pack_4d=fp4 seed 42 ====="
log "  target ~48% (threshold hypothesis first datapoint)"
log "  log ${LOG_FILE}"
log "  checkpoint_every=10  mem_abort_gb=70  wall_timeout=8h"

timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --mode accuracy --method naive_fp4 --model 3B \
     --weight_quant nf4 --bf16_rmsnorm \
     --body_encoding int4 --pack_4d_mode fp4 \
     --group_size 128 --task gsm8k --seed 42 \
     --n_train 7473 --epochs 2 --lr 2e-4 \
     --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
     --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
     --lora_dropout 0.05 --eval_samples 500 \
     --checkpoint_every 10 --mem_abort_gb 70 \
     --output_dir ${OUT}" \
    > "${LOG_FILE}" 2>&1

RC=$?
if (( RC == 137 || RC == 124 )); then
    log "  ${TAG} TIMEOUT (>8h); killing"
    docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
fi
log "  ${TAG} exit=${RC}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

log ""
log "===== done ====="
log "  DECISION POINT: read accuracy from JSON in ${OUT}"
log "    ~48%  ->  launch seeds 123 + 456 (run_3b_naive_int4_seeds.sh)"
log "    ~55%  ->  skip 123/456; proceed with 70B (run_70b_naive_e2m1_4dfp4.sh)"
