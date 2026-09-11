#!/usr/bin/env bash
# Verify the perplexity eval path and adapter loading against known-good values.
#
# A 500-step C_STD run (no activation compression at all) reported WikiText-2
# perplexity 5646 with healthy training loss, normal in-distribution perplexity,
# and adapter weights smaller than the 3736-step reference. Before that is
# reported as a real phenomenon, the eval path itself has to be cleared:
# if adapter loading were broken, every perplexity number in the project would
# be suspect, not just this one.
#
# Measures WikiText-2 on the identical deterministic 200 windows every run uses
# (build_wikitext_id_lists takes no seed) for base, the known-good 3736-step
# C_STD adapter, the catastrophic 500-step run and a healthy one from the same
# cohort. Reading is in the python file's docstring.
set -u
CONTAINER=hma-container
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }
cleanup() { docker exec ${CONTAINER} pkill -f "_diag_wikitext_ppl.py" 2>/dev/null; }
trap cleanup EXIT INT TERM

exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock"
flock -x 9
log "GPU lock acquired"

waited=0; idle=0
while (( idle < 3 )); do
    if docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/(batch_eval|eval_perplexity))" >/dev/null 2>&1; then
        idle=0
    else
        idle=$((idle+1))
    fi
    sleep 60; waited=$((waited+60))
    (( waited > 21600 )) && { log "TIMEOUT waiting for GPU"; exit 1; }
done
log "GPU free"

timeout --signal=KILL 5400 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python scripts/_diag_wikitext_ppl.py" \
    > /home/yedam/HMA/HMA_Project/logs/diag_wikitext_ppl.log 2>&1
rc=$?
(( rc == 137 || rc == 124 )) && docker exec ${CONTAINER} pkill -f "_diag_wikitext_ppl.py" 2>/dev/null
log "diagnostic exit=${rc}"
