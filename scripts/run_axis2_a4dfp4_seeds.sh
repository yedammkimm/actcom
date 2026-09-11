#!/usr/bin/env bash
# Axis 2 — A_4DFP4 seed-instability frequency test
#
# Adds 5 more seeds to the existing n=3 for A_4DFP4 (body=e2m1, pack_4d=fp4).
# Purpose: convert the "1/3 damaged" observation into a proper frequency
# estimate by extending to n=8.
#
# Pre-registered damage threshold (locked BEFORE seeing any data):
#   Standard mean + 3 * B(4D FP8) sd  =  16.08 + 3*0.19  =  16.65
#   A seed with WikiText PPL > 16.65 counts as damaged.
#
# Existing 3 seeds:
#   seed 42:  WikiText 15.78  safe
#   seed 123: WikiText 18.43  damaged
#   seed 456: WikiText 16.96  damaged
# Currently damaged 2/3.
#
# Judgement after n=8:
#   ≥5 damaged  → ~60%+ damage rate (strong)
#   3-4         → ~40% (medium)
#   1-2         → rare-but-happens (weak but honest)
#   0 additions → seed-42-batch was noise; axis 2 rejected

set -u
CONTAINER=hma-container
OUT=results/axis2_seeds
PPL_OUT=results/ppl_axis2
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}
PPL_HOST=/home/yedam/HMA/HMA_Project/${PPL_OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST} ${PPL_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=32400   # 9h cap while INT4 chain finishes
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
    local tag="a_4dfp4_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"

    log ""
    log "--- ${tag} training (seed=${seed}, ~3.5h) ---"
    # eval_samples 200 to save 30min per run — accuracy discrimination is weak
    # anyway; the PPL step is the judgment signal.
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding e2m1 --pack_4d_mode fp4 \
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

    # Find adapter (full or partial)
    local ap=""
    for cand in ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter; skipping PPL"; return 0
    fi

    local rel_ap="${ap#/home/yedam/HMA/HMA_Project/}"
    local ppl_tag="ppl_axis2__A_4DFP4__seed${seed}"
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

log "===== Axis 2 — A_4DFP4 five new seeds — waiting for INT4 chain first ====="
wait_gpu || { log "abort"; exit 1; }

log ""
log "===== axis 2 begin — 5 seeds × (3.5h train + 5min PPL) = ~17.5h ====="

# The specific seeds picked by the user (spread out to avoid clustering)
for seed in 789 1011 2024 3033 4042; do
    run_train_and_ppl $seed
done

log ""
log "===== axis 2 done. Aggregate WikiText PPL across all 8 seeds vs threshold 16.65. ====="
