#!/usr/bin/env bash
# Axis 5 — is the 4-D FP4 damage a function of training length, not scale?
#
# The 70B run came back clean (WikiText delta +0.18%, inside the 3B safe band),
# but it is not comparable to 3B on training length:
#     3B   7473 samples x 2 epochs = 3736 optimiser steps
#     70B  2000 samples x 1 epoch  =  500 optimiser steps   (7.5x fewer)
# If damage accumulates over training, 500 steps may simply be too short for it
# to appear, in which case the clean 70B result says nothing about scale and a
# second 70B seed would not resolve it either. This axis tests that directly by
# running 3B at 500 steps.
#
# Why the pre-registered threshold cannot be used here. tau = 16.65 was built as
# C_STD@3736 mean + 3 * B_4DFP8@3736 sd, i.e. inside the 3736-step regime. A
# 500-step model is far less adapted, and WikiText degradation is a by-product
# of GSM8K adaptation, so its baseline perplexity sits somewhere else entirely.
# Carrying tau across would most likely put all eight runs under it and return a
# 0/8 that reflects undertraining rather than absence of damage. This is the
# same regime-transfer error that the 70B relative bands raise, so the judgement
# here is threshold-free and stays inside the 500-step regime.
#
# Judgement, fixed before the data arrives. At 3736 steps the real signal was
# dispersion, not threshold crossing:
#     A@3736  15.78 - 18.43   spread 2.65   bimodal, gap 0.83 = 4.39 sd
#     B@3736  15.89 - 16.30   spread 0.41   unimodal
#     F = var(A)/var(B) = 33.66, df=(7,7), p = 0.000072
# The same variance ratio is the primary read at 500 steps. Critical values were
# computed here from the regularised incomplete beta (no scipy on this host):
#     B n=3 -> df=(7,2), F.05 = 19.35      B n=5 -> df=(7,4), F.05 = 6.09
#     B n=4 -> df=(7,3), F.05 =  8.89      B n=6 -> df=(7,5), F.05 = 4.88
# Four FP8 runs are used rather than three: it costs one extra run (~31 min) and
# more than halves the critical value, which shrinks the inconclusive band from
# 5-19.4 down to 5-8.9. Going further to five buys much less.
#
#     F > 8.89                      -> A@500 is significantly more dispersed.
#                                      Damage is present at 500 steps, so
#                                      training length is not the explanation
#                                      and a second 70B seed is worth running.
#     F < 5                         -> the two arms disperse alike. Damage
#                                      accumulates with training; the clean 70B
#                                      result is about length, not scale, and a
#                                      second 70B seed would not help.
#     5 <= F <= 8.89                -> inconclusive; add two more FP8 runs
#                                      (df=(7,5), F.05 = 4.88) and re-read.
#
# Secondary read, not a test but easy to see: the range ratio. At 3736 it was
# A 2.65 / B 0.41 = 6.5x. A similar ratio at 500 steps means damage is present;
# 1-2x means it is not.
#
# Seeds match axis 2 exactly (42, 123, 456, 789, 1011, 2024, 3033, 4042) so only
# training length changes. Note that axis 4 established damage to be a property
# of the run rather than the seed, so per-seed pairing across the two lengths is
# NOT interpretable; the comparison that counts is distribution against
# distribution.
#
# eval_samples: the config layer rejects 0 ("mode='accuracy' requires
# eval_samples > 0"), so 1 is used. Accuracy cannot separate these arms anyway
# and perplexity is the judgement metric; a single sample keeps the eval to a
# few seconds instead of 18 minutes.
#
# Order: the four FP8 reference runs go first. Without the reference no A-arm
# number is interpretable, so an interrupted chain is far more useful this way
# round than the reverse.
#
# Runtime: 500 steps ~= 22.6 min at the measured 2.71 s/step, plus ~8 min of
# perplexity, so ~31 min per run and ~6.2 h for all twelve.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/axis5_short
PPL_OUT=results/ppl_axis5
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

FP4_SEEDS=(42 123 456 789 1011 2024 3033 4042)
FP8_SEEDS=(42 123 456 789)

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
# Hazard type 1: the container creates these as root and the host shell then
# cannot open its own redirect targets. chown BEFORE any host-side write.
docker exec ${CONTAINER} chown -R 4051:4051 \
    /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT} 2>/dev/null
mkdir -p "${OUT_HOST}" "${PPL_HOST}"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 4: timeout kills the docker exec client, not the container-side
# process. Make sure nothing of ours outlives the script, however it ends.
cleanup() {
    docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    log "cleanup: container-side jobs killed if any remained"
}
trap cleanup EXIT INT TERM

# Hazard type 3: order queued chains instead of racing them.
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

run_one() {
    # run_one <arm: A_4DFP4|B_4DFP8> <pack_4d_mode> <seed>
    local arm="$1" mode="$2" seed="$3"
    local tag="${arm}_s${seed}"
    local ppl_tag="ppl_axis5__${arm}__seed${seed}"

    if [[ -f "${PPL_HOST}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} already done"; return 0
    fi

    wait_gpu 43200 || return 1
    log ""
    log "--- ${tag}  (500 steps, pack_4d_mode=${mode}) ---"
    timeout --signal=KILL 7200 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode ${mode} \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 2000 --epochs 1 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 1 --eval_max_new_tokens 256 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output_dir ${OUT}" \
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

    local ap=""
    for cand in "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter; skipping PPL"; return 0
    fi
    # Both arms train the same seeds into the same directory, so move the
    # adapter under an arm-tagged name before the other arm reuses that seed.
    local ap_tagged="${OUT_HOST}/${tag}_adapter"
    if [[ "$ap" != "$ap_tagged" ]]; then
        mv "$ap" "$ap_tagged" && ap="$ap_tagged"
        for j in "${OUT_HOST}"/accuracy__naive_fp4__*seed${seed}__*.json; do
            [[ -f "$j" ]] && mv "$j" "${OUT_HOST}/${tag}.json"
        done
    fi

    log "  ${ppl_tag} — PPL (4 corpora, ~8 min)"
    timeout --signal=KILL 3600 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${ap#${ROOT_HOST}/} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed ${seed} --output /app/HMA_Project/${PPL_OUT}/${ppl_tag}.json" \
        > "${PPL_HOST}/${ppl_tag}.log" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${ppl_tag} TIMEOUT — killing the container-side process"
        docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
        sleep 10
    fi
    log "  ${ppl_tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 10
}

log "===== Axis 5 — 3B at 500 steps: is damage a function of training length? ====="
log "      FP8 reference first (${#FP8_SEEDS[@]} runs), then FP4 (${#FP4_SEEDS[@]} runs)"

for s in "${FP8_SEEDS[@]}"; do run_one B_4DFP8 fp8 "$s"; done
for s in "${FP4_SEEDS[@]}"; do run_one A_4DFP4 fp4 "$s"; done

log ""
log "===== axis 5 done ====="
log "  Read F = var(A@500)/var(B@500) on WikiText, df=(7,3):"
log "    F > 8.89        damage present at 500 steps; length is not the"
log "                    explanation; a second 70B seed is worth running"
log "    F < 5           damage accumulates with training; the clean 70B"
log "                    result is about length, not scale"
log "    5 <= F <= 8.89  inconclusive; add two more FP8 runs and re-read"
log "  Secondary: range ratio. 3736 gave A 2.65 / B 0.41 = 6.5x."
