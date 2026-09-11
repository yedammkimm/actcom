#!/usr/bin/env bash
# Post-INT4-training: run held-out PPL on the newest INT4 adapter as soon as
# training finishes. Idempotent: skips if the PPL JSON already exists.

set -u
CONTAINER=hma-container
INT4_OUT=results/naive4bit_int4
PPL_OUT=results/ppl_eval
LOG_HOST=/home/yedam/HMA/HMA_Project/${INT4_OUT}
mkdir -p /home/yedam/HMA/HMA_Project/${PPL_OUT}

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

wait_gpu() {
    local waited=0
    local MAX_WAIT=21600   # 6h cap
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

log "===== INT4 PPL follow-up: waiting for training to finish ====="
wait_gpu || { log "abort"; exit 1; }

# Grab all INT4 adapters currently present (seed 42 first, later 123/456 if run).
for ap in /home/yedam/HMA/HMA_Project/${INT4_OUT}/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed*_adapter; do
    [[ -d "$ap" ]] || continue
    # Extract seed from adapter name
    seed=$(basename "$ap" | sed -E 's/.*__seed([0-9]+)__.*/\1/')
    tag="ppl__D_INT4__seed${seed}"
    log_file="/home/yedam/HMA/HMA_Project/${PPL_OUT}/${tag}.log"
    out_json="/app/HMA_Project/${PPL_OUT}/${tag}.json"
    if [[ -f "/home/yedam/HMA/HMA_Project/${PPL_OUT}/${tag}.json" ]]; then
        log "  SKIP ${tag} (already done)"; continue
    fi
    log "starting ${tag}  adapter=$(basename "$ap")"
    # Adapter path relative to /app/HMA_Project (docker view)
    ap_docker=$(echo "$ap" | sed 's|/home/yedam/HMA/HMA_Project|/app/HMA_Project|')
    timeout --signal=KILL 900 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${ap_docker} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
         --seed ${seed} \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT"
        docker exec ${CONTAINER} pkill -9 -f "eval_perplexity" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${PPL_OUT} 2>/dev/null
    sleep 10
done

log "===== INT4 PPL follow-up done ====="
