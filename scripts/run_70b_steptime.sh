#!/usr/bin/env bash
# 70B step time — does compression cost less time than recomputation?
#
# §4.2 claims 70B does not fit without compression. A reviewer will notice that
# gradient checkpointing also fits, at 52.67 GB against 90.95 for four-bit FP8,
# and ask why the memory is being spent. The answer has to be time, and the
# number is not currently measured for the compressed arm.
#
# Existing values at L = 1024, both method=standard, both nf4, both mem_steps=7
# and mem_warmup=3, batch 1, adamw:
#     no gradient checkpointing   38.728 s/step   71.37 GB   (commit 849f1a8a, 08-18)
#     gradient checkpointing      56.675 s/step   44.53 GB   (commit 6b2aa55a, 08-25)
#     ratio 1.463, i.e. +46.3 %
# Use seconds_per_step, not train_wall_s: the wall covers mem_warmup + mem_steps
# (10 forwards) while seconds_per_step is amortised over the 7 measured ones.
# Dividing the wall by 10 gives 1.446 and is wrong.
#
# Two changes from those runs. Both baselines are re-measured here so all four
# arms come from one session — the pair above sits a week apart, and this
# project has seen a 39.8 % reproduction spread between same-day repeats of one
# 70B cell. And mem_steps rises from 7 to 15 at L = 1024: seven steps is enough
# for a peak-memory reading but not for timing, as the existing GC sweep shows
# by reporting L = 3072 (157 s) as faster than L = 2048 (215 s), which cannot
# happen. Fifteen steps halves the standard error for about five extra minutes
# a cell.
#
# The optimiser is adamw, matching the runs being compared against, not the
# paged_adamw8bit used for the accuracy experiments. That difference belongs in
# the caption.
#
# L = 3072 is measured second and matters more: it is the length the enabling
# claim is about, the length where the uncompressed baseline does not fit at
# all, and therefore the length at which "why not just recompute" has to be
# answered. Seven steps there, since each is already slow.
#
# Reading, on the compressed-to-uncompressed ratio at L = 1024:
#   < 1.30      compression is clearly faster than recomputation
#   1.30-1.45   comparable; present them as different trade-offs
#   > 1.46      recomputation wins on time too; shrink §4.4 to a limitation
# At L = 3072 there is no uncompressed baseline, so the direct GC-against-FP8
# comparison is the one to report.

set -u
CONTAINER=hma-container
ROOT=/home/yedam/HMA/HMA_Project
OUT=results/70b_steptime
mkdir -p "${ROOT}/${OUT}"
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }
cleanup() { docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null; log "cleanup done"; }
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
        (( waited > 86400 )) && { log "TIMEOUT"; return 1; }
    done
    log "  GPU free"
}

run_cell() {
    local tag="$1"; shift
    local json="${OUT}/steptime__${tag}.json"
    [[ -f "${ROOT}/${json}" ]] && { log "  ${tag} already done"; return 0; }
    wait_gpu || return 1
    log ""
    log "--- ${tag} ---"
    timeout --signal=KILL 10800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode memory --model 70B --weight_quant nf4 --bf16_rmsnorm \
         --group_size 128 --task gsm8k --seed 42 \
         --optimizer adamw --scheduler cosine --warmup_ratio 0.0 --lora_dropout 0.05 \
         $* --output ${json}" \
        > "${ROOT}/${OUT}/${tag}.log" 2>&1
    local rc=$?
    (( rc == 137 || rc == 124 )) && { log "  ${tag} TIMEOUT"; docker exec ${CONTAINER} pkill -f run_experiment.py 2>/dev/null; sleep 10; }
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10
}

L1024="--mem_seq_len 1024 --mem_batch_size 1 --mem_steps 15 --mem_warmup 3 --mem_abort_gb 100"
L3072="--mem_seq_len 3072 --mem_batch_size 1 --mem_steps 7  --mem_warmup 3 --mem_abort_gb 115"

log "===== part 1 — L = 1024, four arms, one session ====="
run_cell L1024_std_nogc  $L1024 --method standard
run_cell L1024_std_gc    $L1024 --method standard --gc
run_cell L1024_dcr_fp8   $L1024 --method naive_fp4 --body_encoding int4 --pack_4d_mode fp8
run_cell L1024_dcr_chan  $L1024 --method naive_fp4 --body_encoding int4 --pack_4d_mode chan_int4

log ""
log "===== part 2 — L = 3072, the length the enabling claim is about ====="
run_cell L3072_std_gc    $L3072 --method standard --gc
run_cell L3072_dcr_fp8   $L3072 --method naive_fp4 --body_encoding int4 --pack_4d_mode fp8

log ""
log "===== done ====="
log "  Read seconds_per_step, not train_wall_s. Reference at L=1024, mem_steps=7:"
log "    standard no-GC 38.728 s/step, standard GC 56.675 s/step, ratio 1.463"
