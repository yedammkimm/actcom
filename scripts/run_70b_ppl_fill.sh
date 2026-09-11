#!/usr/bin/env bash
# 70B perplexity fill — the two runs the first 70B chain did not finish.
#
# The FP4 arm was scored successfully (results/ppl_70b/ppl_70b__A_4DFP4__seed42.json).
# Two runs remain:
#   B_4DFP8  the 2026-09-02 attempt reached gsm8k, wikitext2 and narrativeqa but
#            hit its 4 h cap during govreport, so no JSON was written. Its three
#            completed corpora are already usable (gsm8k 2.4091, wikitext2
#            5.9310, narrativeqa 15.4974) and the pre-registered WikiText verdict
#            is already settled; this run exists to complete the four-corpus row.
#   C_STD    never attempted. Gives the uncompressed reference point so 70B has
#            the same three-arm structure as 3B.
#
# Why the first attempt failed, and what changed here:
#   * The 4 h timeout was set from an uncontended estimate. The run was in fact
#     sharing the GPU with a 3B training job, which stretched each corpus to
#     ~4900 s. The cap is raised to 8 h.
#   * `timeout` kills the `docker exec` client, not the process inside the
#     container (handoff doc 4.0, hazard type 4). Last time that left a
#     perplexity job running for 4.5 h after the chain believed it dead, and it
#     kept `pgrep` non-empty so other chains blocked on a phantom. Every timeout
#     path here is followed by an explicit container-side pkill, and the script
#     cleans up on exit as well.
#   * The chain takes the shared GPU lock so a later chain queues behind it in
#     request order rather than racing it (hazard type 3).
#
# Usage: scripts/run_70b_ppl_fill.sh

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
PPL_OUT=results/ppl_70b
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

FP8_ADAPTER=results/70b_accuracy_e2m1/accuracy__naive_fp4__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260821_081519_adapter
STD_ADAPTER=results/70b_accuracy/accuracy__standard__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260818_121120_adapter

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${PPL_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
mkdir -p "${PPL_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 4: make sure nothing of ours outlives the script, however it ends.
cleanup() {
    docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
    log "cleanup: container-side eval_perplexity killed if any remained"
}
trap cleanup EXIT INT TERM

# Hazard type 3: order queued chains rather than racing them.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

wait_gpu() {
    local waited=0 idle=0
    local MAX_WAIT=${1:-43200}
    local NEED_IDLE=3
    while (( idle < NEED_IDLE )); do
        if docker exec ${CONTAINER} pgrep -f \
             "python (run_experiment|scripts/(batch_eval|eval_perplexity))" \
             > /dev/null 2>&1; then
            idle=0
        else
            idle=$((idle + 1))
        fi
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU (${waited}s)"; return 1
        fi
        if (( waited % 900 == 0 )); then
            log "  still waiting for GPU (${waited}s, idle streak ${idle}/${NEED_IDLE})"
        fi
    done
    log "  GPU free (idle for ${NEED_IDLE} consecutive polls)"
}

run_ppl() {
    local adapter_rel="$1" tag="$2"
    if [[ -f "${PPL_HOST}/${tag}.json" ]]; then
        log "  ${tag} already done"; return 0
    fi
    if [[ ! -d "${ROOT_HOST}/${adapter_rel}" ]]; then
        log "  ${tag} adapter missing (${adapter_rel}); skipping"; return 0
    fi
    wait_gpu 43200 || return 1
    log ""
    log "  ${tag} — PPL over 4 corpora on 70B (~2 h uncontended)"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter_rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed 42 --output /app/HMA_Project/${PPL_OUT}/${tag}.json" \
        > "${PPL_HOST}/${tag}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT — killing the container-side process (hazard type 4)"
        docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
        sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 15
}

log "===== 70B perplexity fill — B_4DFP8 and C_STD ====="
run_ppl "${FP8_ADAPTER}" "ppl_70b__B_4DFP8__seed42"
run_ppl "${STD_ADAPTER}" "ppl_70b__C_STD__seed42"

log ""
log "===== 70B perplexity fill done ====="
log "  Three-arm 70B row is complete; the WikiText verdict was already settled"
log "  from the FP4 and FP8 values and does not change."
