#!/usr/bin/env bash
# Re-measure the two points the 70B Standard extrapolation rests on, in one session.
#
# The slope 0.0288 GB/token, the 130.4 GB figure at L=3072 and the L~2990
# crossing all come from exactly two measurements, and those two were made two
# days and two commits apart:
#     L=1024  71.368  2026-08-18  849f1a8  70b_mem_sweep
#     L=1536  86.132  2026-08-20  43a99c3  70b_oom
# Reserved memory varies by 0.5-1.7 GB between sessions elsewhere in this
# project. Propagated through a two-point fit over a 512-token base, +/-1.5 GB
# on either point moves the crossing from 2988 to between 2807 and 3210 -- a
# spread the paper's three-significant-figure claim does not have.
#
# Both points are re-measured here rather than only L=1536: fixing one endpoint
# would leave the other in the old session and the slope would still span two.
#
# Config replicates both originals field for field. Two differences between the
# originals themselves are immaterial and are reconciled here: pack_4d_mode was
# fp8 at L=1024 and fp4 at L=1536, which is a no-op for method=standard (no pack
# hooks fire), and mem_abort_gb was 100 against 115, which neither run
# approached. We use fp8 and 115 for both.
#
# Reading, fixed before the data arrives:
#   both within ~0.5 GB of the originals  -> the two-session fit was sound. Say
#       so in the paper: the uncompressed configuration allocates no packing
#       buffers and its reserved figure is stable across sessions, which is why
#       a two-point fit is defensible.
#   either off by more than ~1.5 GB       -> the slope is session-dependent.
#       Refit on the new pair, quote the crossing to two significant figures,
#       and state the sensitivity.
#   the two disagree in opposite directions -> the slope changes most. Report
#       the refit and the range rather than a point estimate.
#
# Runtime: two 70B loads plus 10 steps each, roughly 35-45 min.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/70b_std_slope
OUT_HOST=${ROOT_HOST}/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
# Hazard type 1: chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p "${OUT_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }
cleanup() {
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    log "cleanup: container-side jobs killed if any remained"
}
trap cleanup EXIT INT TERM

exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock"
flock -x 9
log "GPU lock acquired"

run_L() {
    local L="$1"
    local j="${OUT}/std_L${L}.json"
    if [[ -f "${ROOT_HOST}/${j}" ]]; then log "  L=${L} already done"; return 0; fi
    log ""
    log "--- Standard, L=${L}, 7+3 steps ---"
    timeout --signal=KILL 5400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode memory --method standard \
         --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \
         --weight_quant nf4 --bf16_rmsnorm \
         --pack_4d_mode fp8 --group_size 128 --min_numel 1024 \
         --mem_batch_size 1 --mem_seq_len ${L} \
         --mem_steps 7 --mem_warmup 3 --mem_abort_gb 115 \
         --optimizer adamw --lr 2e-4 --seed 42 \
         --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
         --output ${j}" \
        > "${OUT_HOST}/L${L}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  L=${L} TIMEOUT — killing the container-side process"
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
        sleep 10
    fi
    log "  L=${L} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10
}

log "===== 70B Standard slope re-measurement ====="
run_L 1024
run_L 1536

log ""
log "===== done ====="
log "  originals: L=1024 reserved 71.368 (08-18), L=1536 reserved 86.132 (08-20)"
log "  refit slope = (r1536 - r1024) / 512; crossing = 1024 + (128 - r1024)/slope"
