#!/usr/bin/env bash
# Axis 7 — can four bits be made safe by changing the scale axis alone?
#
# Every arm so far that left 4-D head views at four bits was damaged: blockwise
# INT4 (arm D, Q/K cosine 0.371) went 3/3, blockwise E2M1 (arm A, cosine 0.587)
# went 4/8. The two safe arms both promote those tensors to FP8 (cosine 0.992).
# pack_4d_mode='chan_int4' keeps four bits and moves the scale axis to the
# channel instead, which measures cosine 0.910 on the production quantiser —
# about 80% of the way from A to B.
#
#   Q/K cos    arm                          WikiText              damaged
#   0.371      D  int4 body, 4-D blockwise  17.26 17.69 33.75      3/3
#   0.587      A  e2m1 body, 4-D blockwise  15.78 .. 18.43         4/8
#   0.910      THIS RUN                                            ?
#   0.992      B  e2m1 body, 4-D FP8        15.89 .. 16.30         0/8
#   0.992      E  int4 body, 4-D FP8        15.93 .. 16.47         0/3
#
# Body encoding is E2M1, matching arm A. That makes A the comparison: same body,
# same bit width, scale axis the only difference, which is exactly the question.
# Arm D would also be a clean contrast but it is n=3 against A's n=8, and the
# dose-response curve is drawn with an E2M1 body at both of its established
# points, so mixing an INT4 body into the middle of it would muddle the axis.
# Note that axis 6's finding — body encoding is free once head views are at FP8 —
# does NOT transfer here: these head views are at four bits, not eight, so the
# body effect may well return.
#
# Memory is not the claim. Scale overhead is 16/(B*L) against 16/group_size for
# blockwise, and at this project's sequence lengths (harmonic mean 152) that is
# 0.105 against 0.125 bits per element; a smoke run measured 4.1206 bits against
# 4.1250 for all-blockwise. The 0.447 bits that FP8 costs at 70B is recovered by
# using four bits at all, not by the choice of scale axis. What this run tests is
# whether that is safe to do.
#
# Judgement, fixed before the data arrives, on WikiText-2 against the
# pre-registered tau = 16.65:
#   0/3    -> the scale axis alone makes four bits safe; no FP8 dtype needed,
#             which lowers the hardware requirement
#   2-3/3  -> cosine 0.910 is not enough; the case for FP8 is strengthened
#   1/3    -> intermediate; report descriptively, do not claim either
# Three seeds cannot estimate a rate, but the neighbouring cells are saturated
# (A is 4/8, B is 0/8), so the direction will be visible.
#
# Early signal within about twenty minutes, from grad_norm_trace: A runs at 1.18
# with 77% of steps above the clip threshold, B and E at 0.61 with essentially
# none. A value near 0.7 would put this arm on the safe side.
#
# Config replicates arm A (results/naive4bit_e2m1) except pack_4d_mode. Two
# fields differ from those particular runs — checkpoint_every 50 -> 10 and
# mem_abort_gb 100 -> 70 — because a finer gradient trace is wanted and 70 GB
# leaves headroom under the 80 GiB cgroup. Both are inert for the training
# mathematics, and arm A is itself already split on them: its seeds 42/123/456
# used 50/100 while its axis-2 seeds used 10/70, and all eight are pooled.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/axis7_chan_int4
PPL_OUT=results/ppl_axis7
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

SEEDS=(42 123 456)

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 \
    /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT} 2>/dev/null
mkdir -p "${OUT_HOST}" "${PPL_HOST}"

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

run_seed() {
    local seed="$1"
    local tag="F_CHANINT4_s${seed}"
    local ppl_tag="ppl_axis7__F_CHANINT4__seed${seed}"
    if [[ -f "${PPL_HOST}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} already done"; return 0
    fi

    wait_gpu 43200 || return 1
    log ""
    log "--- ${tag}  (body_encoding=e2m1, pack_4d_mode=chan_int4) ---"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode chan_int4 \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 500 --eval_max_new_tokens 256 \
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

log "===== Axis 7 — per-channel INT4 on 4-D head views (Q/K cos 0.910) ====="
for s in "${SEEDS[@]}"; do run_seed "$s"; done

log ""
log "===== axis 7 done ====="
log "  Count WikiText-2 above the pre-registered tau = 16.65:"
log "    0/3   -> the scale axis alone makes four bits safe; no FP8 dtype needed"
log "    2-3/3 -> cosine 0.910 is not enough; the case for FP8 is strengthened"
log "    1/3   -> intermediate; report descriptively"
log "  Compare grad_norm_trace against A (1.18, clip 77%) and B/E (0.61, clip ~0%)."
