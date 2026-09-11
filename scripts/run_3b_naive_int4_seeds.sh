#!/usr/bin/env bash
# Seeds 123 + 456 for the §5 threshold argument.
# Launch ONLY after seed 42 lands near 48% (confirming threshold hypothesis).
# Together with seed 42 these give mean + sd for the "cos 0.36" row of the §5 table.

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
    local MAX_WAIT=1800
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/eval_perplexity|gradient_error_probe)" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU"
            return 1
        fi
    done
    log "  GPU free"
}

run_seed() {
    local seed="$1"
    local tag="naive_4dfp4_int4_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"
    log ""
    log "--- ${tag} ---"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding int4 --pack_4d_mode fp4 \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 500 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output_dir ${OUT}" \
        > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT (>8h); killing"
        docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 60
}

log "===== INT4 seeds 123 + 456 (§5 threshold table sd recovery) ====="
wait_gpu || { log "abort"; exit 1; }

run_seed 123
run_seed 456

log ""
log "===== done ====="
log "  Extract mean+sd across seeds {42, 123, 456} and populate §5 table cos-0.36 row."
