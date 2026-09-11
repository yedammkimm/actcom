#!/usr/bin/env bash
# 70B Gradient-Checkpointing (GC) memory sweep (2026-08-24).
# Standard + gc for L in {1024, 1536, 2048, 3072}. GC is only supported with
# method='standard' (run_experiment.py rejects it otherwise).
#
# Waits for any current run_experiment.py or gradient_error_probe.py in the
# container to finish before starting.
#
# Compared against prior DCR values:
#   L=1024  oamp E2M1  peak_reserved  66.69 GB
#   L=2048  oamp E2M1  peak_reserved  83.16 GB
#   L=3072  oamp E2M1  peak_reserved  99.81 GB
#
# Hypothesis: at 70B (80 layers) GC recompute cost per step should exceed the
# 3B (28-layer) reference (~+32%); observing >+50% at 70B would strengthen §5.4.

set -u
CONTAINER=hma-container
OUT=results/70b_gc
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

log "===== 70B GC sweep queued ====="

log "waiting for GPU (any run_experiment.py, gradient_error_probe.py, or batch_eval.py)..."
while docker exec ${CONTAINER} pgrep -f "python run_experiment.py|python.*gradient_error_probe.py|python.*batch_eval.py" > /dev/null 2>&1; do
    sleep 30
done
log "GPU free"
sleep 30

COMMON="--mode memory --method standard --gc \
--model 70B --weight_quant nf4 --bf16_rmsnorm \
--mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
--seed 42 --output_dir ${OUT}"

# L=1024/1536/2048: abort_gb=100 to keep 20 GB safety headroom (matches prior).
# L=3072: abort_gb=115 to match the earlier 70b_oom probe conditions.
run_L() {
    local L="$1"; local abort_gb="$2"
    local tag="standard_gc_L${L}"
    local log_file="${LOG_HOST}/${tag}.log"
    log ""
    log "--- ${tag} (abort_gb=${abort_gb}) ---"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         ${COMMON} \
         --mem_seq_len ${L} --mem_abort_gb ${abort_gb}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 60
}

log ""
log "===== 70B GC memory sweep begin ====="

run_L 1024 100
run_L 1536 100
run_L 2048 100
run_L 3072 115

log ""
log "===== 70B GC memory sweep complete ====="
