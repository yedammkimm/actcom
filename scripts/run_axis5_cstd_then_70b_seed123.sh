#!/usr/bin/env bash
# Two experiments, strictly sequential on the one GPU.
#
#   1. C_STD @ 500 steps x 4 seeds   (~1.9 h)  — the missing cell of axis 5
#   2. 70B 4-D FP4, seed 123         (~8.3 h)  — second seed for the scale claim
#
# ---------------------------------------------------------------------------
# 1. C_STD @ 500 — is the 500-step instability caused by compression at all?
#
# Axis 5 ran both 4-bit arms at 500 steps and both came back wildly dispersed
# on out-of-distribution text, with one extreme run each:
#     A_4DFP4@500  15.54 15.91 15.93 15.95 16.99 18.09 19.56 42.09  (WikiText)
#     B_4DFP8@500  15.38 15.73 18.03 28.36
# Per-chunk trajectories confirm these are real: seed 1011 reads 36.6 after its
# first 25 WikiText chunks and never comes down, and its GovReport running value
# starts at 169.7 against a normal ~10.8. No chunk was dropped as non-finite in
# any of the twelve runs, and no adapter carries NaN.
#
# So the extremes are genuine degradation, not a measurement accident. What the
# design cannot say is whether compression caused them, because B_4DFP8 — the
# arm that is supposed to be the stable reference — is itself dispersed at 500
# steps (sd 6.10 against 0.158 at 3736 steps). With the reference broken, the
# pre-registered F test is not interpretable: F = 2.19 is produced entirely by
# the two extreme observations, and drops to 1.06 when one is removed from each
# arm. Brown-Forsythe on medians gives W = 0.0055 against a critical 4.96, i.e.
# no detectable difference in dispersion, while the same test at 3736 steps
# gives W = 9.64 against 4.60.
#
# C_STD carries no activation compression at all, so it isolates the cause.
# Judgement fixed before the data arrives, on BOTH corpora — WikiText alone is
# too weak a read here given how large the GovReport excursions were:
#
#   C_STD@500 tight        WikiText ~15.5-16.0 AND GovReport ~11-13
#     -> the instability is caused by 4-bit activation compression, and FP8
#        4-D routing does NOT protect against it at short training. That is a
#        new result and §5's recommendation needs a qualifier about training
#        length.
#   C_STD@500 dispersed    any seed with GovReport > 100, or WikiText spread
#                          comparable to the 4-bit arms
#     -> the 500-step regime is pathological independently of compression.
#        This route is closed: do NOT go on to an intermediate step count.
#   mixed                  WikiText tight but GovReport excursions, or vice versa
#     -> report descriptively, close the route, no further runs.
#
# Config matches the axis-5 arms exactly except for --method standard, which
# makes body_encoding and pack_4d_mode inert (no packing context is entered).
#
# ---------------------------------------------------------------------------
# 2. 70B 4-D FP4 seed 123 — a second seed for the scale claim
#
# This runs regardless of what C_STD@500 shows. Seed 42 came back clean
# (WikiText delta +0.178% against FP8, inside the 3B safe band), but at the 3B
# damage rate of 4/8 a single clean run carries a likelihood ratio of only 2:1.
# A second seed either doubles that to 4:1, or — with probability about a half
# if the phenomenon persists at scale — returns damaged, which would settle the
# scale claim outright.
#
# Config replicates the 2026-08-21 70B FP8 run field for field except
# pack_4d_mode (fp8 -> fp4) and seed (42 -> 123), matching the seed-42 FP4 run
# already on disk. eval_max_new_tokens stays at the original 256 so accuracy
# remains comparable across the 70B arms. The measured total for this
# configuration is 8.3 h (4.6 h train + 3.8 h eval), so the cap is 12 h.
#
# ---------------------------------------------------------------------------
# Hazards handled (handoff doc 4.0): chown before any host-side write (type 1),
# this file is never edited while running — copy it instead (type 2), the shared
# flock orders queued chains (type 3), and every timeout path plus an EXIT trap
# kills the container-side process that `timeout` leaves behind (type 4).

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project

C_OUT=results/axis5_short
C_PPL=results/ppl_axis5
B_OUT=results/70b_axis
B_PPL=results/ppl_70b

CSTD_SEEDS=(42 123 456 789)
SEED70B=123

for d in ${C_OUT} ${C_PPL} ${B_OUT} ${B_PPL}; do
    docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${d}
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${d} 2>/dev/null
    mkdir -p "${ROOT_HOST}/${d}"
done
mkdir -p "${ROOT_HOST}/logs"

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

cleanup() {
    docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
    docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
    log "cleanup: container-side jobs killed if any remained"
}
trap cleanup EXIT INT TERM

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
    # run_ppl <adapter_rel> <ppl_dir> <tag> <seed> <timeout_s>
    local adapter_rel="$1" dir="$2" tag="$3" seed="$4" tmo="$5"
    if [[ -f "${ROOT_HOST}/${dir}/${tag}.json" ]]; then
        log "  ${tag} already done"; return 0
    fi
    log "  ${tag} — PPL over 4 corpora"
    timeout --signal=KILL "${tmo}" docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${adapter_rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed ${seed} --output /app/HMA_Project/${dir}/${tag}.json" \
        > "${ROOT_HOST}/${dir}/${tag}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT — killing the container-side process (hazard type 4)"
        docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
        sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${dir} 2>/dev/null
    sleep 10
}

# ---------------------------------------------------------------------------
# 1. C_STD @ 500 steps
# ---------------------------------------------------------------------------
log "===== C_STD @ 500 steps — the missing cell of axis 5 ====="

run_cstd() {
    local seed="$1"
    local tag="C_STD_s${seed}"
    local ppl_tag="ppl_axis5__C_STD__seed${seed}"
    if [[ -f "${ROOT_HOST}/${C_PPL}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} already done"; return 0
    fi
    wait_gpu 43200 || return 1
    log ""
    log "--- ${tag}  (500 steps, method=standard, no activation compression) ---"
    timeout --signal=KILL 7200 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method standard --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 2000 --epochs 1 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 1 --eval_max_new_tokens 256 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output_dir ${C_OUT}" \
        > "${ROOT_HOST}/${C_OUT}/${tag}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT — killing the container-side process"
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
        sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${C_OUT} 2>/dev/null
    sleep 10

    local ap=""
    for cand in "${ROOT_HOST}/${C_OUT}"/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                "${ROOT_HOST}/${C_OUT}"/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter; skipping PPL"; return 0
    fi
    local ap_tagged="${ROOT_HOST}/${C_OUT}/${tag}_adapter"
    if [[ "$ap" != "$ap_tagged" ]]; then
        mv "$ap" "$ap_tagged" && ap="$ap_tagged"
        for j in "${ROOT_HOST}/${C_OUT}"/accuracy__standard__*seed${seed}__*.json; do
            [[ -f "$j" ]] && mv "$j" "${ROOT_HOST}/${C_OUT}/${tag}.json"
        done
    fi
    run_ppl "${ap#${ROOT_HOST}/}" "${C_PPL}" "${ppl_tag}" "${seed}" 3600
}

for s in "${CSTD_SEEDS[@]}"; do run_cstd "$s"; done

log ""
log "===== C_STD @ 500 done ====="
log "  Read WikiText AND GovReport together:"
log "    tight (WT ~15.5-16.0, Gov ~11-13)  -> compression causes the 500-step"
log "       instability and FP8 does not protect at short training"
log "    any seed with Gov > 100, or WT spread like the 4-bit arms"
log "       -> the regime is pathological; close this route, no 1868-step run"

# ---------------------------------------------------------------------------
# 2. 70B 4-D FP4, seed 123 — runs regardless of the above
# ---------------------------------------------------------------------------
log ""
log "===== 70B 4-D FP4 seed ${SEED70B} — second seed for the scale claim ====="

FP4_JSON=${B_OUT}/naive_e2m1_4dfp4_s${SEED70B}.json
if [[ -f "${ROOT_HOST}/${FP4_JSON}" ]]; then
    log "  ${FP4_JSON} already exists; skipping training"
else
    wait_gpu 43200 || { log "abort"; exit 1; }
    log ""
    log "--- 70B naive_fp4 e2m1 pack_4d=fp4 seed ${SEED70B} (~8.3 h) ---"
    timeout --signal=KILL 43200 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 70B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode fp4 \
         --group_size 128 --task gsm8k --seed ${SEED70B} \
         --n_train 2000 --epochs 1 --lr 5e-5 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 200 --eval_max_new_tokens 256 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output ${FP4_JSON}" \
        > "${ROOT_HOST}/logs/70b_4dfp4_s${SEED70B}.log" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  70B FP4 s${SEED70B} TIMEOUT — the adapter is written before eval,"
        log "  so perplexity can still run; killing the container-side process"
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null
        sleep 10
    fi
    log "  70B FP4 s${SEED70B} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${B_OUT} 2>/dev/null
    sleep 15
fi

FP4_ADAPTER=""
for cand in "${ROOT_HOST}/${B_OUT}"/naive_e2m1_4dfp4_s${SEED70B}_adapter \
            "${ROOT_HOST}/${B_OUT}"/naive_e2m1_4dfp4_s${SEED70B}_adapter_partial; do
    [[ -d "$cand" ]] && FP4_ADAPTER="$cand"
done
if [[ -n "$FP4_ADAPTER" ]]; then
    wait_gpu 43200 || { log "abort"; exit 1; }
    run_ppl "${FP4_ADAPTER#${ROOT_HOST}/}" "${B_PPL}" "ppl_70b__A_4DFP4__seed${SEED70B}" "${SEED70B}" 28800
else
    log "  no 70B FP4 seed ${SEED70B} adapter found; cannot score it"
fi

log ""
log "===== both experiments done ====="
log "  70B read: delta = ppl(FP4 s123) - ppl(FP8 s42) on WikiText, as a percentage,"
log "  against the 3B bands (damaged +4%..+15%, safe -1.5%..+1.3%). Two clean 70B"
log "  runs raise the likelihood ratio to 4:1; one damaged run settles the scale"
log "  claim. A clean pair still is not proof of safety at scale."
