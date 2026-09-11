#!/usr/bin/env bash
# GACT det + GACT SR training on 3B seed 42 (2026-08-31).
#
# Purpose: complete the §5 cos-ppl curve with two more datapoints.
#   Q/K cos     body                 ppl(target)
#    0.99       e2m1 + 4D fp8         16.07  (measured)
#    0.585      e2m1 + 4D fp4         17.06  (measured)
#    0.50       gact_affine_det       ?
#    0.36       int4 (running)        ?
#   -0.01       gact_affine (SR)      ?  ← may diverge
#
# Uses --eval_samples 200 (SE ~3.5pp, enough to see "collapsed vs not") to save
# ~30 min per run. Adapter is saved BEFORE training loss NaNs (new partial-save
# path in run_experiment.py), so a divergent SR still yields an adapter for PPL.

set -u
CONTAINER=hma-container
OUT=results/gact_body
LOG_HOST=/home/yedam/HMA/HMA_Project/${OUT}

docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=32400   # 9h cap (INT4 + PPL follow-up + optional 123/456)
    while docker exec ${CONTAINER} pgrep -f "python (run_experiment|scripts/eval_perplexity|gradient_error_probe)" > /dev/null 2>&1; do
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

run_gact() {
    local body="$1"     # gact_affine_det or gact_affine
    local seed="$2"
    local tag="gact_${body}_s${seed}"
    local log_file="${LOG_HOST}/${tag}.log"

    log ""
    log "--- ${tag} (body_encoding=${body}) ---"
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --mode accuracy --method naive_fp4 --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --body_encoding ${body} --pack_4d_mode fp4 \
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
        log "  ${tag} TIMEOUT (>8h); killing"
        docker exec ${CONTAINER} pkill -9 -f "run_experiment" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    sleep 30
}

eval_ppl_for_adapter() {
    local ap_host="$1"
    local out_tag="$2"
    local ap_docker=$(echo "$ap_host" | sed 's|/home/yedam/HMA/HMA_Project|/app/HMA_Project|')
    local tag="ppl__${out_tag}"
    local log_file="/home/yedam/HMA/HMA_Project/results/ppl_eval/${tag}.log"
    local out_json="/app/HMA_Project/results/ppl_eval/${tag}.json"
    if [[ -f "/home/yedam/HMA/HMA_Project/results/ppl_eval/${tag}.json" ]]; then
        log "  SKIP ${tag}"; return 0
    fi
    log "starting ${tag}"
    timeout --signal=KILL 900 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${ap_docker} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --seed 42 --output ${out_json}" \
        > "${log_file}" 2>&1
    log "  ${tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/ppl_eval 2>/dev/null
}

log "===== GACT det + GACT SR body training (§5 threshold curve completion) ====="
wait_gpu || { log "abort"; exit 1; }

# Determinsitc first — safer, sets expectation for SR.
run_gact gact_affine_det 42
# SR may diverge; adapter save is best-effort in run_experiment.py.
run_gact gact_affine 42

log ""
log "===== PPL follow-up on any GACT adapters produced ====="

# Full-training adapter (only if training completed)
for ap in ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__*_adapter; do
    [[ -d "$ap" ]] || continue
    # Derive body encoding from the parent JSON
    parent_json="${ap%_adapter}.json"
    if [[ -f "$parent_json" ]]; then
        be=$(python3 -c "import json; print(json.load(open('$parent_json')).get('config',{}).get('body_encoding'))" 2>/dev/null)
    else
        be="unknown"
    fi
    eval_ppl_for_adapter "$ap" "E_GACT__${be}__seed42"
done

# Partial adapter (rescued from a diverged run)
for ap in ${LOG_HOST}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__*_adapter_partial; do
    [[ -d "$ap" ]] || continue
    parent_json="${ap%_adapter_partial}.json"
    if [[ -f "$parent_json" ]]; then
        be=$(python3 -c "import json; print(json.load(open('$parent_json')).get('config',{}).get('body_encoding'))" 2>/dev/null)
    else
        be="unknown"
    fi
    eval_ppl_for_adapter "$ap" "E_GACT__${be}__partial__seed42"
done

log ""
log "===== done. §5 threshold curve should now have 5 points ====="
