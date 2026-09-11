#!/usr/bin/env bash
# §4.4 matrix, re-measured at nf4 with the recommended routing.
#
# The existing matrix (results/verify/matrix, commit 849f1a8a) reports throughput
# for a configuration this paper calls unusable: bf16 weights, an INT4 body and
# blockwise four-bit head views — arm D, which damaged 3 of 3 runs. A reviewer
# reading "4 of 8 damaged" in §5.2 and turning back to §4.4 finds the numbers
# describing the wrong thing.
#
# Three cells, all nf4 weights, L = 4096, batch 2, 100 measured steps after 10
# warm-up, matching the old matrix's step budget so the two are comparable on
# everything except what changed:
#   1  naive_fp4, INT4 body, 4-D FP8   the recommended configuration
#   2  standard                        uncompressed baseline
#   3  standard + gradient checkpointing
#
# Using an INT4 body also removes the E2M1 correction problem entirely: the old
# table's step times were measured with an INT4-era body, and the correction
# that would have been needed rests on a measurement whose own reproduction
# spread is 39.8 %. With INT4 as the recommendation there is nothing to correct.
#
# Batch is 2, not the old matrix's 4. At batch 4 with nf4 weights the
# uncompressed baseline reserves 76.65 GB and trips the 70 GB watchdog at step
# zero; the old bf16 matrix reached 98.02 GB reserved, which the current 80 GiB
# cgroup no longer permits. Halving the batch keeps the sequence length — the
# axis that actually drives activation memory — and leaves the baseline headroom.
# The three cells stay internally comparable; against the bf16 matrix they are
# not, but that matrix differs in weight precision regardless.
#
# ~1 h per cell at L = 4096 batch 2 on 3B, so about 3 h for the three.

set -u
CONTAINER=hma-container
ROOT=/home/yedam/HMA/HMA_Project
OUT=results/matrix_nf4
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
        (( waited > 43200 )) && { log "TIMEOUT"; return 1; }
    done
    log "  GPU free"
}

run_cell() {
    local tag="$1"; shift
    local json="${OUT}/${tag}.json"
    [[ -f "${ROOT}/${json}" ]] && { log "  ${tag} already done"; return 0; }
    wait_gpu || return 1
    log ""
    log "--- ${tag} ---"
    timeout --signal=KILL 14400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode memory --model 3B --weight_quant nf4 --bf16_rmsnorm \
         --group_size 128 --task gsm8k --seed 42 \
         --mem_seq_len 4096 --mem_batch_size 2 --mem_steps 100 --mem_warmup 10 \
         --optimizer adamw --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --mem_abort_gb 70 \
         $* --output ${json}" \
        > "${ROOT}/${OUT}/${tag}.log" 2>&1
    local rc=$?
    (( rc == 137 || rc == 124 )) && { log "  ${tag} TIMEOUT"; docker exec ${CONTAINER} pkill -f run_experiment.py 2>/dev/null; sleep 10; }
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10
}

log "===== §4.4 matrix at nf4, recommended routing ====="
run_cell standard_nf4          --method standard
run_cell standard_nf4_gc       --method standard --gc
run_cell recommended_int4_fp8  --method naive_fp4 --body_encoding int4 --pack_4d_mode fp8

log ""
log "===== done ====="
log "  Compare against results/verify/matrix (bf16, INT4-era body, legacy 4-D):"
log "    standard 88.55 alloc / 2744.0 s,  naive_fp4 58.47 / 3312.6,  GC 38.89 / 3618.8"
