#!/usr/bin/env bash
# 70B scale check — does the 4-D FP4 instability found at 3B persist at 70B?
#
# At 3B, 4-D FP4 routing damaged 4 of 8 runs on held-out WikiText while 4-D FP8
# damaged 0 of 8 (one-tailed Fisher p = 0.039). This asks whether the same
# failure appears at 70B, which is what §3's scale claim rests on.
#
# Only the FP4 arm needs training. The FP8 and Standard adapters already exist:
#   FP8  results/70b_accuracy_e2m1/accuracy__naive_fp4__...__seed42__20260821_081519_adapter
#        method=naive_fp4 body=e2m1 pack_4d=fp8 seed=42 n_train=2000 ep=1 lr=5e-5
#   STD  results/70b_accuracy/accuracy__standard__...__seed42__20260818_121120_adapter
#        method=standard, identical n_train/epochs/lr/seed/max_seq_len. Its
#        body_encoding and fp8_ratio differ on paper but are inert for
#        method=standard, which performs no activation compression at all.
# That makes a three-point comparison possible rather than a bare FP4-vs-FP8
# delta, and saves about 16 hours of training.
#
# Config fidelity. The FP4 run replicates the 2026-08-21 FP8 run field for
# field, with three deliberate exceptions, all inert for the training
# mathematics:
#   checkpoint_every  25 -> 10   finer trace; the 2026-08-26 hang struck at
#                                step 175 and 10-step granularity is needed to
#                                localise it if it recurs
#   mem_abort_gb    80.0 -> 70   leaves 10 GB under the 80 GiB cgroup. Safe:
#                                the original run peaked at 55.2 GB reserved
#                                (56.4 GB by the training log), so the lower
#                                ceiling still clears the real peak by 14 GB
#   output_dir                   results/70b_axis, so the new run does not mix
#                                into the directory holding the originals
# eval_max_new_tokens stays at the original 256. Raising it to 512 would roughly
# double an eval that already runs 3.85 h, and would leave the FP4 accuracy
# incomparable to the FP8 run it is meant to be read against.
#
# Timeout. The measured total for this configuration is 8.28 h (train 4.43 h +
# eval 3.85 h), so the usual 8 h cap would kill the run during eval. It is
# raised to 12 h here. The adapter is written before eval starts
# (run_experiment.py:537 precedes the evaluate() call at :564), so even a
# timeout during eval leaves a scorable adapter behind — only the result JSON
# would be lost.
#
# Judgement. 70B has one seed per configuration, so no threshold can be built
# the way it was at 3B (Standard mean + 3 x FP8 sd needs a spread that does not
# exist here). The read is the relative gap instead:
#     delta = ppl(4D FP4) - ppl(4D FP8),  expressed as a percentage
#     3B damaged runs:  +4% to +15%
#     3B safe runs:     -1.5% to +1.3%
# The inference is asymmetric and this is fixed before the data arrives:
#   delta in the damaged band  ->  the phenomenon persists at scale
#   delta in the safe band     ->  NOT evidence that 70B is safe. At the 3B
#                                  damage rate of 4/8, a single run has a 50%
#                                  chance of coming back clean, so one safe
#                                  outcome carries a likelihood ratio of only
#                                  2:1 and must be reported as uninformative
#   anything between           ->  no call
#
# Usage:  scripts/run_70b_axis_4dfp4.sh
# Runtime: ~8.3 h training, then ~1-2 h per perplexity run over three adapters.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/70b_axis
PPL_OUT=results/ppl_70b
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

FP8_ADAPTER=results/70b_accuracy_e2m1/accuracy__naive_fp4__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260821_081519_adapter
STD_ADAPTER=results/70b_accuracy/accuracy__standard__Meta-Llama-3.1-70B-Instruct-bnb-4bit__nf4__rmsbf16__seed42__20260818_121120_adapter

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
# The container creates these as root and the host shell (uid 4051) then cannot
# open its own redirect targets inside them. chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 \
    /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT} 2>/dev/null
mkdir -p "${OUT_HOST}" "${PPL_HOST}" "${ROOT_HOST}/logs"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Require several consecutive clear polls. A single pgrep miss lands in the gap
# between one chain's training exit and its own follow-up eval, which is how an
# earlier chain started early.
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
    # run_ppl <adapter_rel> <tag>
    local adapter_rel="$1" tag="$2"
    if [[ -f "${PPL_HOST}/${tag}.json" ]]; then
        log "  ${tag} already done"; return 0
    fi
    if [[ ! -d "${ROOT_HOST}/${adapter_rel}" ]]; then
        log "  ${tag} adapter missing (${adapter_rel}); skipping"; return 0
    fi
    wait_gpu 43200 || return 1
    log ""
    log "  ${tag} — PPL over 4 corpora on 70B (~1-2 h)"
    timeout --signal=KILL 14400 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter_rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed 42 --output /app/HMA_Project/${PPL_OUT}/${tag}.json" \
        > "${PPL_HOST}/${tag}.log" 2>&1
    log "  ${tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 15
}

# ------------------------------------------------------------------
# Step 1 — train the 4-D FP4 arm (the only configuration not already on disk)
# ------------------------------------------------------------------
FP4_JSON=${OUT}/naive_e2m1_4dfp4_s42.json
log "===== 70B scale check — 4-D FP4 vs existing FP8 and Standard ====="
log "      waiting for the axis-4 chain to finish"
wait_gpu 43200 || { log "abort"; exit 1; }

if [[ -f "${ROOT_HOST}/${FP4_JSON}" ]]; then
    log "  ${FP4_JSON} already exists; skipping training"
else
    log ""
    log "--- 70B naive_fp4 e2m1 pack_4d=fp4 seed 42 (~8.3 h: 4.4 h train + 3.9 h eval) ---"
    timeout --signal=KILL 43200 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 70B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode fp4 \
         --group_size 128 --task gsm8k --seed 42 \
         --n_train 2000 --epochs 1 --lr 5e-5 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 200 --eval_max_new_tokens 256 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output ${FP4_JSON}" \
        > "${ROOT_HOST}/logs/70b_4dfp4_s42.log" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  70B FP4 TIMEOUT — the adapter is saved before eval, so perplexity"
        log "  can still run; only the result JSON is lost"
        docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
    fi
    log "  70B FP4 exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 15
fi

# ------------------------------------------------------------------
# Step 2 — perplexity on all three adapters, FP4 first so the essential
#          delta lands even if the later runs are interrupted
# ------------------------------------------------------------------
FP4_ADAPTER=""
for cand in "${OUT_HOST}"/naive_e2m1_4dfp4_s42_adapter \
            "${OUT_HOST}"/naive_e2m1_4dfp4_s42_adapter_partial \
            "${OUT_HOST}"/accuracy__naive_fp4__*_adapter; do
    [[ -d "$cand" ]] && FP4_ADAPTER="$cand"
done

if [[ -n "$FP4_ADAPTER" ]]; then
    run_ppl "${FP4_ADAPTER#${ROOT_HOST}/}" "ppl_70b__A_4DFP4__seed42"
else
    log "  no 70B FP4 adapter found; cannot score the new arm"
fi
run_ppl "${FP8_ADAPTER}" "ppl_70b__B_4DFP8__seed42"
run_ppl "${STD_ADAPTER}" "ppl_70b__C_STD__seed42"

log ""
log "===== 70B scale check done ====="
log "  Read delta = ppl(FP4) - ppl(FP8) as a percentage and place it against"
log "  the 3B bands: damaged +4%..+15%, safe -1.5%..+1.3%. A result in the safe"
log "  band is NOT evidence that 70B is safe — one run has a 50% chance of"
log "  coming back clean at the 3B damage rate, so report it as uninformative."
