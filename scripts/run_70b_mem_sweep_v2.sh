#!/usr/bin/env bash
# 70B memory sweep v2 (2026-08-18). Refinements after L=1024 Standard result:
#   - Standard L=1024 already complete → skip
#   - L=1024, L=2048 keep mem_abort_gb=100
#   - L=4096: force all three methods with mem_abort_gb=95 to keep 15 GB
#     safety margin from the docker cgroup ceiling (110 GB). All three are
#     expected to abort; the abort itself is citable data.
#
# Order stays outer L, inner method.

set -u

CONTAINER=hma-container
OUT=results/70b_mem_sweep
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

declare -A method_oom
method_oom[standard]=0
method_oom[naive_fp4]=0
method_oom[oamp]=0

COMMON="--mode memory --model 70B --weight_quant nf4 --bf16_rmsnorm \
--pack_4d_mode fp8 --mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
--seed 42 --output_dir ${OUT}"

extras_standard=""
extras_naive_fp4=""
extras_oamp="--fp8_ratio 0.2 --group_size 128 --mask_seed 42"

# run_one <method> <L> <abort_gb>
run_one() {
    local method="$1"; local L="$2"; local abort_gb="$3"
    local tag="${method}_L${L}"
    local ev="extras_${method}"
    local extras="${!ev}"
    local log_file="${LOG_HOST}/${tag}.log"

    log "starting ${tag}  (abort_gb=${abort_gb})"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --method ${method} ${extras} ${COMMON} \
         --mem_seq_len ${L} --mem_abort_gb ${abort_gb}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    return ${rc}
}

log "===== 70B memory sweep v2 begin ====="
log ""

# L=1024, L=2048: mem_abort_gb=100 with OOM propagation.
for L in 1024 2048; do
    log "--- L=${L} (abort_gb=100) ---"
    for method in standard naive_fp4 oamp; do
        # Standard L=1024 already completed on the first run.
        if [ "${L}" = "1024" ] && [ "${method}" = "standard" ]; then
            log "  SKIP standard_L1024 (already OK from prior chain)"
            continue
        fi
        if [ "${method_oom[$method]}" = "1" ]; then
            log "  SKIP ${method}_L${L} (prior OOM on ${method})"
            continue
        fi
        run_one "${method}" "${L}" 100
        rc=$?
        if [ "${rc}" != "0" ]; then
            method_oom[$method]=1
            log "  MARKED ${method} as OOM/ERROR"
        fi
        sleep 30
    done
    log ""
done

# L=4096: forced sweep at abort_gb=95 (15 GB below cgroup ceiling).
# All three expected to abort; each abort JSON is citable data.
log "--- L=4096 (abort_gb=95, forced across all methods) ---"
for method in standard naive_fp4 oamp; do
    run_one "${method}" 4096 95
    sleep 30
done

log "===== 70B memory sweep v2 complete ====="
