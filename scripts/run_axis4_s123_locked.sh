#!/usr/bin/env bash
# Axis 4 — is A_4DFP4 damage a property of the SEED or of the RUN?
#
# Motivation. B_4DFP8 seed 789 was trained twice under an identical
# configuration (results/pilot_e2m1_seeds, 2026-08-25 and results/axis3_b4dfp8,
# 2026-09-01). The two runs share bit-identical losses at optimiser steps 0 and
# 1 and diverge from step 2 onward (1.226646 vs 1.226328), finishing at 54.0%
# and 56.0% accuracy. So a seed does not determine a run: GPU non-determinism
# splits the trajectory within the first few steps.
#
# That makes the subject of the §5.3 claim ambiguous. "4/8 seeds damaged"
# presumes damage is a function of the seed; if it is a function of the run,
# the correct statement is "4/8 runs damaged" and a practitioner cannot dodge
# the failure by picking a seed.
#
# Design. Re-run damaged A_4DFP4 seeds and see whether they come back safe.
#   seed-deterministic  ->  re-run stays damaged          (p ~ 1)
#   run-lottery         ->  re-run comes back safe        (p ~ 0.5)
# A flip is decisive, because determinism essentially forbids flips. No flip is
# weaker evidence: under the lottery hypothesis the likelihood ratio for one
# non-flip is only 2:1, which is why more than one seed is re-run here.
#   0 flips of 2  ->  lottery p = 0.25   (leans deterministic)
#   0 flips of 3  ->  lottery p = 0.125  (leans deterministic)
#   >= 1 flip     ->  lottery confirmed
#
# Config fidelity. Each seed is re-run with the exact config of its ORIGINAL
# run, including fields that do not affect training mathematics
# (checkpoint_every, eval_samples, mem_abort_gb), so no reviewer can object
# that conditions differed. The two source directories disagree on those
# fields, so the table below carries per-seed values. Only output_dir
# deliberately differs, so re-runs land beside the originals rather than mixing
# into the same directory as same-seed duplicates.
#
# Code fidelity. Originals ran at 6b2aa55 (naive4bit_e2m1 seeds 123/456) and
# 7bde870 (axis2_seeds seeds 3033/4042); HEAD is 77e9837. The functional diff
# over run_experiment.py, configs/base.py and oamp/pack_hooks.py between
# 6b2aa55 and HEAD is additive only and unreachable for body_encoding=e2m1
# with pack_4d_mode=fp4: the GACT encodings require body_encoding=gact_*, the
# partial-adapter save only fires on NaN, and the eval watchdog only fires on
# abort. 7bde870 is identical to HEAD on every training-path file. All four
# damaged seeds are therefore valid re-run targets at HEAD.
#
# Damaged A_4DFP4 seeds and their WikiText PPL (pre-registered tau = 16.65):
#   123  18.4320   most extreme; a return to ~16 would be the most dramatic flip
#   456  16.9550   near the boundary
#   3033 17.0183   near the boundary
#   4042 16.9350   closest to the boundary; most likely to flip
#
# Usage:
#   scripts/run_axis4_rerun_damaged.sh              # defaults to 456 3033
#   scripts/run_axis4_rerun_damaged.sh 123 4042     # extreme + closest-to-tau
#
# Runtime: ~3.5h training + ~5min PPL per seed, plus a one-off ~8min PPL on the
# 2026-08-25 B_4DFP8 seed-789 adapter (see below).

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/axis4_rerun
PPL_OUT=results/ppl_axis4
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

SEEDS=("${@:-456 3033}")
# shellcheck disable=SC2206
SEEDS=(${SEEDS[@]})

# Per-seed replication settings, read off each original run's saved config.
#   seed : checkpoint_every : eval_samples : mem_abort_gb : origin dir
declare -A ORIG_CKPT=( [123]=50  [456]=50  [3033]=10  [4042]=10 )
declare -A ORIG_EVALN=( [123]=500 [456]=500 [3033]=200 [4042]=200 )
declare -A ORIG_ABORT=( [123]=100 [456]=100 [3033]=70  [4042]=70 )
declare -A ORIG_DIR=( [123]=naive4bit_e2m1 [456]=naive4bit_e2m1 \
                      [3033]=axis2_seeds   [4042]=axis2_seeds )
declare -A ORIG_WT=( [123]=18.4320 [456]=16.9550 [3033]=17.0183 [4042]=16.9350 )

# The 2026-08-25 B_4DFP8 seed-789 adapter never had a perplexity run. Scoring
# it gives a second, already-trained observation of the same arm and seed for
# free, so run-to-run reproducibility can be read directly off B as well.
B789_OLD_ADAPTER=results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260825_130253_adapter

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
# The container creates these as root, which leaves the host shell unable to
# open its own redirect targets inside them. chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 \
    /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT} 2>/dev/null
mkdir -p "${OUT_HOST}" "${PPL_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 3 (handoff doc 4.0): a pgrep polling loop cannot order two chains
# that are both waiting -- both see the GPU go idle at the same moment and both
# start. Take an exclusive lock so queued chains are granted in request order.
# The lock releases when fd 9 closes, so a killed chain does not strand it.
# The 70B chain now running was started before this lock existed and does not
# hold it, so wait_gpu below is still needed to wait that one out.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

# Require the GPU to look idle on several consecutive polls before proceeding.
# A single pgrep miss lands in the gap between one chain's training exit and
# its own follow-up eval, which is how an earlier chain started early.
wait_gpu() {
    local waited=0 idle=0
    local MAX_WAIT=${1:-21600}     # 6h default
    local NEED_IDLE=3              # 3 consecutive clear polls = 3 min
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
    # run_ppl <adapter_rel> <tag> <seed>
    local adapter_rel="$1" tag="$2" seed="$3"
    if [[ -f "${PPL_HOST}/${tag}.json" ]]; then
        log "  ${tag} already done"; return 0
    fi
    log "  ${tag} — PPL (4 corpora, ~5 min)"
    timeout --signal=KILL 1800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter_rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed ${seed} --output /app/HMA_Project/${PPL_OUT}/${tag}.json" \
        > "${PPL_HOST}/${tag}.log" 2>&1
    log "  ${tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 15
}

rerun_seed() {
    local seed="$1"
    if [[ -z "${ORIG_CKPT[$seed]:-}" ]]; then
        log "  seed ${seed} is not a known damaged A_4DFP4 seed; skipping"; return 0
    fi
    local tag="a_4dfp4_rerun_s${seed}"
    local log_file="${OUT_HOST}/${tag}.log"

    log ""
    log "--- ${tag} (original WikiText ${ORIG_WT[$seed]} > 16.65 = DAMAGED) ---"
    log "    replicating ${ORIG_DIR[$seed]} config: checkpoint_every=${ORIG_CKPT[$seed]}"
    log "    eval_samples=${ORIG_EVALN[$seed]}  mem_abort_gb=${ORIG_ABORT[$seed]}"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode fp4 \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples ${ORIG_EVALN[$seed]} \
         --eval_max_new_tokens 256 \
         --checkpoint_every ${ORIG_CKPT[$seed]} \
         --mem_abort_gb ${ORIG_ABORT[$seed]} \
         --output_dir ${OUT}" \
        > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT"
        docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 15

    local ap=""
    for cand in "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter; skipping PPL"; return 0
    fi
    run_ppl "${ap#${ROOT_HOST}/}" "ppl_axis4__A_4DFP4_rerun__seed${seed}" "${seed}"
}

log "===== Axis 4 — re-run damaged A_4DFP4 seeds: ${SEEDS[*]} ====="
log "      waiting for the axis-3 chain and the seed-789 retry to finish"
wait_gpu 64800 || { log "abort"; exit 1; }

# Free observation first: score the older B_4DFP8 seed-789 adapter, which was
# trained under the same config as the axis-3 seed-789 run but is a different
# run. Cheap, and it reads run-to-run reproducibility straight off arm B.
if [[ -d "${ROOT_HOST}/${B789_OLD_ADAPTER}" ]]; then
    log ""
    log "--- B_4DFP8 seed 789, 2026-08-25 run (duplicate of the axis-3 run) ---"
    run_ppl "${B789_OLD_ADAPTER}" "ppl_axis4__B_4DFP8_dup__seed789" 789
else
    log "  B_4DFP8 seed-789 2026-08-25 adapter missing; skipping duplicate PPL"
fi

for seed in "${SEEDS[@]}"; do
    wait_gpu 64800 || { log "abort"; exit 1; }
    rerun_seed "$seed"
done

log ""
log "===== axis 4 done ====="
log "  Compare each re-run's WikiText PPL against tau = 16.65 and against the"
log "  original value. A seed that returns below tau has flipped, which"
log "  establishes that damage is a property of the run, not the seed, and the"
log "  §5.3 sentence must say 'runs' rather than 'seeds'."
log "  Then re-run: python3 scripts/analyze_axis3.py"
