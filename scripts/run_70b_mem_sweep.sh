#!/usr/bin/env bash
# 70B memory sweep (2026-08-18). Figure 1 data for the enabling claim.
#
# Design (per user directive):
#   L    = 1024, 2048, 4096            (8192 skipped: activation ≈ 232 GB, both OOM by arithmetic)
#   M    = standard, naive_fp4, oamp   (naive answers "why OAMP over naive")
#   step = warmup 3 + measured 7       (peak stabilises within a few steps)
#   pack_4d_mode = fp8 (Llama condition)
#
# One process per (M, L) so allocator fragmentation from a prior config never
# leaks into the next peak. When a method OOMs at some L, its remaining larger
# L are skipped (exit code check). --mem_abort_gb=100 fires our in-process
# watchdog before the docker cgroup SIGKILL — so an OOM leaves a citable JSON.
#
# Loop is outer L, inner method. This means all three methods hit L=1024 first
# (fast, all pass), then L=2048, then L=4096 (where Standard likely OOMs).

set -u

CONTAINER=hma-container
OUT=results/70b_mem_sweep
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Track OOM per method so later L for that method is skipped.
declare -A method_oom
method_oom[standard]=0
method_oom[naive_fp4]=0
method_oom[oamp]=0

# Common CLI args (identical across all runs so the memory numbers compare fairly).
COMMON="--mode memory --model 70B --weight_quant nf4 --bf16_rmsnorm \
--pack_4d_mode fp8 --mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
--mem_abort_gb 100 --seed 42 --output_dir ${OUT}"

# Per-method extras.
extras_standard=""
extras_naive_fp4=""
extras_oamp="--fp8_ratio 0.2 --group_size 128 --mask_seed 42"

# run_one <method> <L>  -> exit code 0 on OK, non-zero on OOM/NAN/ERROR
run_one() {
    local method="$1"; local L="$2"
    local tag="${method}_L${L}"
    local ev="extras_${method}"
    local extras="${!ev}"
    local log_file="${LOG_HOST}/${tag}.log"

    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --method ${method} ${extras} ${COMMON} --mem_seq_len ${L}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    return ${rc}
}

log "===== 70B memory sweep begin ====="
log "L set = {1024, 2048, 4096}   methods = {standard, naive_fp4, oamp}"
log ""

for L in 1024 2048 4096; do
    log "--- L=${L} ---"
    for method in standard naive_fp4 oamp; do
        if [ "${method_oom[$method]}" = "1" ]; then
            log "  SKIP ${method}_L${L} (prior OOM on ${method})"
            continue
        fi

        run_one "${method}" "${L}"
        rc=$?
        if [ "${rc}" != "0" ]; then
            method_oom[$method]=1
            log "  MARKED ${method} as OOM/ERROR — will skip its larger L"
        fi
        sleep 30    # allocator cleanup between runs
    done
    log ""
done

log "===== 70B memory sweep complete ====="
log "Results in ${OUT}/*.json"
