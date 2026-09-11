#!/usr/bin/env bash
# Axis 7 extension to n=8, then the 70B memory sweep under per-channel INT4.
#
# ---------------------------------------------------------------------------
# 1. F_CHANINT4 seeds 789, 1011, 2024, 3033, 4042      (~17.5 h)
#
# The recommendation has moved from "promote 4-D head views to FP8" to "give
# them a per-channel scale and keep four bits", so the configuration being
# recommended is now the one with the weakest evidence: F is at n=3 while both
# the arm it replaces (B, 0/8) and the arm it is contrasted against (A, 4/8)
# are at n=8. Three clean runs do not exclude a damage rate that would matter:
#   true rate 50%  ->  P(0/3) = 0.125
#   true rate 25%  ->  P(0/3) = 0.42
# A 25% rate cannot be ruled out at n=3 and would be unacceptable for a headline
# recommendation. Extending to n=8 puts F on the same footing as A and B, and
# doing it now rather than after the write-up avoids having to rewrite the
# method and results sections if a damaged run appears.
#
# Same seeds as A and B so the three arms share a seed set. Config identical to
# the first three F runs (body e2m1, pack_4d_mode chan_int4, n_train 7473,
# epochs 2, lr 2e-4, eval_samples 500, checkpoint_every 10, mem_abort_gb 70).
#
# Judgement stays the pre-registered WikiText-2 rule, tau = 16.65, now over
# eight runs:
#   0/8  ->  matches B exactly; the recommendation stands on equal evidence
#   1/8  ->  still far below A's 4/8; report the rate honestly
#   >=2  ->  the per-channel claim weakens and FP8 returns as the safe default
#
# ---------------------------------------------------------------------------
# 2. 70B memory sweep at pack_4d_mode=chan_int4, L in {1024, 2048, 3072}  (~2 h)
#
# The memory table currently reports a configuration the paper no longer
# recommends. Under 4-D FP8 the 70B run measures byte_bits 4.5693 and peak
# reserved 63.23 / 76.85 / 90.95 GB at the three sequence lengths. Per-channel
# INT4 should bring byte_bits to about 4.12, since the 4-D fraction (11.11% of
# kept elements at 70B) drops from eight bits to four while the scale overhead
# barely moves. That is roughly 0.45 bits per element, and it should push the
# L=3072 figure down by several GB and move the 128 GB crossover further out.
#
# Config replicates results/70b_mem_sweep_e2m1 field for field: mode=memory,
# mem_batch_size 1, mem_steps 7, mem_warmup 3, gc off, optimizer adamw, and the
# per-length mem_abort_gb the originals used (100 / 100 / 115). Those ceilings
# were set for the FP8 runs; the per-channel runs should sit below them, and if
# one does abort that is itself the measurement.
#
# Note the byte_bits figures are now computed from scale counts accumulated at
# pack time rather than assumed to be 16/group_size. For an all-blockwise
# configuration the two agree to tail-group rounding, so the existing FP8
# numbers remain comparable.

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/axis7_chan_int4
PPL_OUT=results/ppl_axis7
MEM_OUT=results/70b_mem_chan_int4
OUT_HOST=${ROOT_HOST}/${OUT}
PPL_HOST=${ROOT_HOST}/${PPL_OUT}
MEM_HOST=${ROOT_HOST}/${MEM_OUT}

SEEDS=(789 1011 2024 3033 4042)

for d in ${OUT} ${PPL_OUT} ${MEM_OUT}; do
    docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${d}
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${d} 2>/dev/null
    mkdir -p "${ROOT_HOST}/${d}"
done

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
    local MAX_WAIT=${1:-86400}
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
    wait_gpu 86400 || return 1
    log ""
    log "--- ${tag}  (body=e2m1, pack_4d=chan_int4) ---"
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
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null; sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 10

    local ap=""
    for cand in "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                "${OUT_HOST}"/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    [[ -z "$ap" ]] && { log "  ${tag} no adapter; skipping PPL"; return 0; }
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
    (( rc == 137 || rc == 124 )) && docker exec ${CONTAINER} pkill -f "eval_perplexity.py" 2>/dev/null
    log "  ${ppl_tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 10
}

run_mem() {
    local L="$1" abort="$2"
    local tag="mem_70b_chan_int4_L${L}"
    if [[ -f "${MEM_HOST}/${tag}.json" ]]; then
        log "  ${tag} already done"; return 0
    fi
    wait_gpu 86400 || return 1
    log ""
    log "--- ${tag}  (70B, pack_4d=chan_int4, mem_abort_gb=${abort}) ---"
    timeout --signal=KILL 10800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode memory --method naive_fp4 --model 70B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode chan_int4 \
         --group_size 128 --task gsm8k --seed 42 \
         --mem_seq_len ${L} --mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
         --optimizer adamw --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --mem_abort_gb ${abort} \
         --output ${MEM_OUT}/${tag}.json" \
        > "${MEM_HOST}/${tag}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT — killing the container-side process"
        docker exec ${CONTAINER} pkill -f "run_experiment.py" 2>/dev/null; sleep 10
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${MEM_OUT} 2>/dev/null
    sleep 10
}

log "===== Part 1 — F_CHANINT4 to n=8 (seeds ${SEEDS[*]}) ====="
for s in "${SEEDS[@]}"; do run_seed "$s"; done

log ""
log "===== Part 2 — 70B memory sweep under chan_int4 ====="
run_mem 1024 100
run_mem 2048 100
run_mem 3072 115

log ""
log "===== done ====="
log "  F: count WikiText-2 above tau = 16.65 over eight runs."
log "  70B: compare byte_bits and peak_reserved_gb against the FP8 sweep"
log "       (4.5693 bits; 63.23 / 76.85 / 90.95 GB at L=1024/2048/3072)."
