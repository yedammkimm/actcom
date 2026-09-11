#!/usr/bin/env bash
# 70B memory sweep for E2M1 body encoding (2026-08-21).
# Fills the E2M1 column of the enabling-claim table by re-running naive_fp4 and
# oamp at L in {1024, 2048, 3072}. Standard is NOT re-run because it does not
# compress activations and therefore is independent of body_encoding; the INT4
# measurements in results/70b_mem_sweep/ and results/70b_oom/ apply verbatim.
#
# oamp L=1024 was already measured on 2026-08-21 (peak 66.69 GB), so we skip it.
#
# Order: outer L (small first), inner method (naive before oamp).
# Abort thresholds mirror prior sweeps to keep results directly comparable.

set -u

CONTAINER=hma-container
OUT=results/70b_mem_sweep_e2m1
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

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
         --method ${method} ${extras} \
         --body_encoding e2m1 \
         --mode memory --model 70B --weight_quant nf4 --bf16_rmsnorm \
         --pack_4d_mode fp8 \
         --mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
         --seed 42 --output_dir ${OUT} \
         --mem_seq_len ${L} --mem_abort_gb ${abort_gb}" \
        > "${log_file}" 2>&1
    local rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    return ${rc}
}

log "===== 70B E2M1 memory sweep begin ====="
log ""

# L=1024: fill the missing naive cell (oamp already measured on 2026-08-21).
log "--- L=1024 (abort_gb=100) ---"
run_one naive_fp4 1024 100
sleep 30
log ""

# L=2048: both methods, abort_gb=100 (matches prior INT4 sweep).
log "--- L=2048 (abort_gb=100) ---"
run_one naive_fp4 2048 100
sleep 30
run_one oamp 2048 100
sleep 30
log ""

# L=3072: both methods, abort_gb=115 (matches 70b_oom probe).
log "--- L=3072 (abort_gb=115) ---"
run_one naive_fp4 3072 115
sleep 30
run_one oamp 3072 115
log ""

log "===== 70B E2M1 memory sweep complete ====="
