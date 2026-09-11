#!/usr/bin/env bash
# Axis 6 — the empty cell of the 2x2: INT4 body with 4-D FP8 routing.
#
#                   4D FP4          4D FP8
#   body INT4    D  3/3 damaged     THIS RUN
#   body E2M1    A  4/8 damaged     B  0/8 damaged
#
# The paper recommends two things at once — use E2M1 for the body, and promote
# 4-D head-view tensors to FP8 — without ever testing whether both are needed.
# This cell decides that.
#
# Prediction from the gradient probe. The measured cosine gap between INT4 and
# E2M1 is +0.226 on Q/K and below 0.01 on residual, MLP, V and O. The two body
# encodings therefore differ essentially only on Q/K, and Q/K are exactly the
# 4-D tensors that pack_4d_mode governs. With pack_4d_mode=fp8 those tensors are
# stored at 8 bits regardless of body_encoding, so the encodings should become
# equivalent. Training gradients agree: A (E2M1 body, 4-D FP4) runs at grad norm
# 1.21 while D (INT4 body, 4-D FP4) runs at 586, and the only compressed tensors
# that differ between them are Q/K.
#
# Judgement, fixed before the data arrives. Damage is counted on WikiText-2
# against the pre-registered tau = 16.65, the same rule as every other arm:
#   0/3 damaged  ->  4-D routing alone is sufficient. The E2M1 recommendation is
#                    not needed on top of it and should be withdrawn, which
#                    simplifies the method to a single intervention.
#   3/3 damaged  ->  body encoding and 4-D routing are independently required.
#                    Keep the current two-part recommendation.
#   1-2/3        ->  intermediate; report descriptively and do not claim either.
# Three seeds is enough to see the direction because the two neighbouring cells
# are saturated (D is 3/3, B is 0/8); it is not enough to estimate a rate.
#
# Early signal, available within minutes rather than hours: grad_norm_trace. If
# this cell behaves like B the trace sits near 0.61 with a clip rate near zero;
# if it behaves like D it sits in the hundreds with 100% of steps above the clip
# threshold. That distinguishes the two hypotheses long before perplexity.
#
# Config replicates the 2026-08-31 D_INT4 runs field for field — eval_samples
# 500, checkpoint_every 10, mem_abort_gb 70 — with pack_4d_mode as the single
# change (fp4 -> fp8). Only output_dir differs, so results land beside the
# originals rather than mixing with them. The D_INT4 runs were made at commit
# 1a91dcf; every training-path file (pack_hooks.py, run_experiment.py,
# configs/base.py, quantize.py, dtype_policy.py) is byte-identical at HEAD
# (77e9837), so the comparison is exact.
#
# Runtime: the D_INT4 runs took 2.24 h and 5.04 h of training plus 0.75-1.60 h of
# eval; budgeting ~3.5 h per seed gives ~10.5 h for three, plus ~8 min of
# perplexity each.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/axis6_int4_fp8
PPL_OUT=results/ppl_axis6
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}

SEEDS=(42 123 456)

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

run_seed() {
    local seed="$1"
    local tag="E_INT4FP8_s${seed}"
    local ppl_tag="ppl_axis6__E_INT4FP8__seed${seed}"
    if [[ -f "${PPL_HOST}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} already done"; return 0
    fi

    wait_gpu 43200 || return 1
    log ""
    log "--- ${tag}  (body_encoding=int4, pack_4d_mode=fp8) ---"
    log "    replicating naive4bit_int4 config: eval_samples=500 checkpoint_every=10 mem_abort_gb=70"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding int4 --pack_4d_mode fp8 \
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

log "===== Axis 6 — INT4 body with 4-D FP8: is the E2M1 recommendation needed? ====="
for s in "${SEEDS[@]}"; do run_seed "$s"; done

log ""
log "===== axis 6 done ====="
log "  Count WikiText-2 perplexity above the pre-registered tau = 16.65:"
log "    0/3  -> 4-D routing alone suffices; withdraw the E2M1 recommendation"
log "    3/3  -> both interventions are independently required; keep it"
log "    1-2  -> intermediate; report descriptively"
log "  Also compare grad_norm_trace against B (~0.61, clip ~0%) and D (~586, clip 100%)."
