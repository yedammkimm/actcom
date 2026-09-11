#!/usr/bin/env bash
# Axis 3 — B_4DFP8 seed-stability confirmation (mirror of axis 2)
#
# Adds 5 more seeds to the existing n=3 for B_4DFP8 (body=e2m1, pack_4d=fp8).
# Purpose: convert "0/3 damaged" into "0/8 damaged" so §5.3 has symmetric
# power against A_4DFP4's n=8. Under a true 50% instability rate, 0/3 has
# p=1/8=12.5% (weak); 0/8 has p=(1/2)^8=0.4% (very strong).
#
# Pre-registered damage threshold (locked BEFORE seeing axis-2 data, reused):
#   Standard mean + 3 * B(4D FP8) sd  =  16.08 + 3*0.19  =  16.65
#   Any arm's seed with WikiText PPL > 16.65 counts as damaged.
#
# Existing 3 B_4DFP8 seeds (all safe):
#   seed 42:  WikiText 15.93  safe
#   seed 123: WikiText 16.00  safe
#   seed 456: WikiText 16.28  safe
# Currently damaged 0/3.
#
# Judgement after n=8:
#   0/8  →  binomial(0, 8, 0.50) p=0.004  strong "stable" (paper claim)
#   1/8  →  binomial(≤1, 8, 0.50) p=0.035  still significantly < A
#   ≥2/8 →  weakens 4D FP8 stability claim; report honestly
#
# The 5 new seeds mirror axis 2 A: 789, 1011, 2024, 3033, 4042.
# Only diff vs run_axis2_a4dfp4_seeds.sh is --pack_4d_mode fp8 (was fp4).

set -u
CONTAINER=hma-container
OUT=results/axis3_b4dfp8
PPL_OUT=results/ppl_axis3
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}
PPL_HOST=/home/yedam/HMA/HMA_Project/${PPL_OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST} ${PPL_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=3600   # 1h cap — nothing else queued
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/(batch_eval|eval_perplexity))" > /dev/null 2>&1; do
        sleep 60
        waited=$((waited + 60))
        if (( waited > MAX_WAIT )); then
            log "  TIMEOUT waiting for GPU"; return 1
        fi
        if (( waited % 600 == 0 )); then
            log "  still waiting for GPU (${waited}s)"
        fi
    done
    log "  GPU free"
}

run_train_and_ppl() {
    local seed="$1"
    local tag="b_4dfp8_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"

    log ""
    log "--- ${tag} training (seed=${seed}, ~3.5h) ---"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode fp8 \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 200 \
         --checkpoint_every 10 --mem_abort_gb 70 \
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
    for cand in ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter; skipping PPL"; return 0
    fi

    local rel_ap="${ap#/home/yedam/HMA/HMA_Project/}"
    local ppl_tag="ppl_axis3__B_4DFP8__seed${seed}"
    local ppl_json="/app/HMA_Project/${PPL_OUT}/${ppl_tag}.json"
    local ppl_log="${PPL_HOST}/${ppl_tag}.log"
    if [[ -f "${PPL_HOST}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} already done"; return 0
    fi

    log ""
    log "  ${ppl_tag} — PPL (4 corpora, ~5 min)"
    timeout --signal=KILL 1500 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${rel_ap} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --seed ${seed} --output ${ppl_json}" \
        > "${ppl_log}" 2>&1
    log "  ${ppl_tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 15
}

log "===== Axis 3 — B_4DFP8 five new seeds — waiting for GPU ====="
wait_gpu || { log "abort"; exit 1; }

log ""
log "===== axis 3 begin — 5 seeds × (3.5h train + 5min PPL) = ~17.5h ====="

for seed in 789 1011 2024 3033 4042; do
    run_train_and_ppl $seed
done

log ""
log "===== axis 3 done. Aggregate B_4DFP8 WikiText PPL across n=8 vs threshold 16.65. ====="
