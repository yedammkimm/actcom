#!/usr/bin/env bash
# Dtype decomposition (c) — one session, at the §4.4 regime.
#
# The 11.4 GB figure that earlier notes filed under "dtype" turned out to be two
# separate effects measured across two files and two commits. This re-measures
# both inside one session, at the configuration §4.4 already uses, so the paper
# reads as one chain instead of an inter-file subtraction:
#
#     neither -> rms_only -> both -> compressed
#              RMSNorm     LoRA    compression
#                                  (30.15, already measured)
#
# Original conditions (bf16 weights, B=4, L=4096) cannot be re-run: peak was
# 105.57 GB against an 80 GB container.
#
# Pre-registered predictions, fixed before the data arrives. Both dtype steps
# are the same mechanism — a (B, L, 3072) tensor saved for backward at four
# bytes per element instead of two — so each is (sites x 3072 x 2 bytes x B x L):
#
#     RMSNorm   56 sites  = 28 layers x 2 norms        -> 2.819 GB
#     LoRA     112 sites  = 28 layers x 4 lora targets -> 5.646 GB
#
# giving, from the measured §4.4 uncompressed cell of 43.37 GB allocated:
#
#     both      43.37   (must reproduce §4.4; see tolerance below)
#     rms_only  49.02   = 43.37 + 5.646     <- LoRA broken, RMSNorm fixed
#     neither   51.83   = 49.02 + 2.819
#
# Note the middle cell carries the LoRA delta, not the RMSNorm one: RMSNorm has
# half as many sites, so the larger of the two steps is the adapter cast.
#
# Reading, also fixed in advance:
#     both within 2 GB of 43.37     -> the two experiments are consistent and
#                                      the decomposition joins §3.5 to §4.4
#     both further than 2 GB        -> session variation; report the offset and
#                                      quote the ladder in deltas, not absolutes
#     each step within ~5 % of its
#     prediction                    -> the bf16 B=4 mechanism carries to nf4
#     a step off by more than that  -> something differs at nf4 and the write-up
#                                      says so rather than reconciling it
#
# bf16_params (norms / embed / lm_head) stays ON in all three cells. Turning it
# off would also revive the prepare_model_for_kbit_training fp32 upcast, which
# never ran in the BF16-base experiment this is being compared against, and
# would put three effects in a two-step ladder. A fourth cell can measure it
# separately if the paper ever needs it.
#
# method=standard throughout: this measures dtype, not compression.
# Runtime: ~3 min preflight + ~15 min per cell = ~50 min.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/dtype_decomp
OUT_HOST=${ROOT_HOST}/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
# Hazard type 1: the container creates these as root and the host shell then
# cannot open its own redirect targets. chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p "${OUT_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 4: timeout kills the docker exec client, not the container-side
# process. Make sure nothing of ours outlives the script, however it ends.
cleanup() {
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    docker exec ${CONTAINER} pkill -f "_verify_dtype_gates.py" 2>/dev/null
    log "cleanup: container-side jobs killed if any remained"
}
trap cleanup EXIT INT TERM

# Hazard type 3: order queued chains rather than racing them.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

wait_gpu() {
    local waited=0 idle=0 MAX_WAIT=${1:-43200} NEED_IDLE=3
    while (( idle < NEED_IDLE )); do
        if docker exec ${CONTAINER} pgrep -f \
             "python (run_experiment|scripts/(batch_eval|eval_perplexity))" \
             > /dev/null 2>&1; then idle=0; else idle=$((idle + 1)); fi
        sleep 60; waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then log "  TIMEOUT waiting for GPU"; return 1; fi
    done
    log "  GPU free (idle for ${NEED_IDLE} consecutive polls)"
}

COMMON="--mode memory --method standard --model 3B --weight_quant nf4 \
        --mem_seq_len 4096 --mem_batch_size 2 --mem_steps 100 --mem_warmup 10 \
        --mem_abort_gb 70 --optimizer adamw --seed 42"

log "===== dtype decomposition — preflight ====="
wait_gpu 43200 || exit 1
timeout --signal=KILL 1800 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python scripts/_verify_dtype_gates.py" \
    > "${OUT_HOST}/preflight.log" 2>&1
pf=$?
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
tail -12 "${OUT_HOST}/preflight.log"
if (( pf != 0 )); then
    log "PREFLIGHT FAILED (exit=${pf}) — the gates do not do what they claim."
    log "Not spending GPU time on numbers that would be uninterpretable."
    exit 1
fi
log "preflight PASS"

run() {
    local tag="$1"; shift
    local j="${OUT}/dtype__${tag}.json"
    if [[ -f "${ROOT_HOST}/${j}" ]]; then log "  ${tag} already done"; return 0; fi
    wait_gpu 43200 || return 1
    log ""
    log "--- ${tag}  ($*) ---"
    timeout --signal=KILL 5400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py ${COMMON} $* --output ${j}" \
        > "${OUT_HOST}/${tag}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT — killing the container-side process"
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
        sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10
}

log ""
log "===== three cells ====="
run neither  --no_bf16_rmsnorm --no_bf16_lora
run rms_only --no_bf16_lora
run both

log ""
log "===== dtype decomposition done ====="
log "  predictions: neither 51.83  rms_only 49.02  both 43.37 (allocated, GB)"
log "  steps:       RMSNorm 2.819 (56 sites)   LoRA 5.646 (112 sites)"
log "  check each step / (sites x 3072 x 2 x 4096) == 2.00 B/elem/site"
