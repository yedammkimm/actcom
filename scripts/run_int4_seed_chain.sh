#!/usr/bin/env bash
# INT4 seeds 123 + 456 — training + PPL (4 corpora) — §5 threshold row sd
#
# For each seed:
#   1. Train (naive_fp4, body=int4, pack_4d=fp4, ~3.5h)
#   2. PPL on 4 corpora (~5 min): GSM8K test, WikiText-2, narrativeqa, govreport
#
# All the safeguards are wired in:
#   * timeout --signal=KILL 28800  (8h wall cap per run)
#   * mem_abort_gb=70              (10 GB headroom in the 80 GB cgroup)
#   * checkpoint_every=10          (fine trace for post-mortem)
#   * partial adapter save on NaNError (rescue divergent SR-style runs)
#   * eval watchdog inside evaluate() aborts + saves partial JSON on OOM
#
# If shinya reboots mid-run, checkpoint sidecar shows exactly where we stopped
# and the adapter (full or partial) can be re-scored with batch_eval + PPL later.

set -u
CONTAINER=hma-container
OUT=results/naive4bit_int4
PPL_OUT=results/ppl_eval
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}
PPL_HOST=/home/yedam/HMA/HMA_Project/${PPL_OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT} /app/HMA_Project/${PPL_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST} ${PPL_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

run_train_and_ppl() {
    local seed="$1"
    local tag="naive_4dfp4_int4_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"

    log ""
    log "--- ${tag} training (seed=${seed}, ~3.5h) ---"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding int4 --pack_4d_mode fp4 \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 500 \
         --checkpoint_every 10 --mem_abort_gb 70 \
         --output_dir ${OUT}" \
        > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT (>8h)"
        docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 15

    # Locate the freshly-saved adapter (either full or partial after NaNError)
    local ap=""
    for cand in ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter \
                ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed${seed}__*_adapter_partial; do
        [[ -d "$cand" ]] && ap="$cand"
    done
    if [[ -z "$ap" ]]; then
        log "  ${tag} no adapter to score; skipping PPL"
        return 0
    fi
    local rel_ap="${ap#/home/yedam/HMA/HMA_Project/}"
    local ppl_tag="ppl__D_INT4__seed${seed}"
    local ppl_json="/app/HMA_Project/${PPL_OUT}/${ppl_tag}.json"
    local ppl_log="${PPL_HOST}/${ppl_tag}.log"
    if [[ -f "${PPL_HOST}/${ppl_tag}.json" ]]; then
        log "  ${ppl_tag} PPL already done, skipping"
        return 0
    fi

    log ""
    log "  ${ppl_tag} — PPL over 4 corpora (~5 min)"
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

log "===== INT4 seeds 123 + 456 — training + 4-corpus PPL — §5 sd row ====="
run_train_and_ppl 123
run_train_and_ppl 456

log ""
log "===== all done ====="
log "  Next: aggregate 3-seed sd across 4 corpora; §5 verdict."
