#!/usr/bin/env bash
# Recovery: eval the INT4 seed 42 adapter that survived the shinya reboot.
# 500-sample GSM8K eval + WikiText/GSM8K PPL, ~20 min total.

set -u
CONTAINER=hma-container
ADAPTER=results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260831_082311_adapter
OUT_JSON="/app/HMA_Project/${ADAPTER}_reeval_n500_mnt256.json"
LOG_HOST=/home/yedam/HMA/HMA_Project/logs
mkdir -p ${LOG_HOST}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Step 1: accuracy via batch_eval (matches training-time eval protocol)
log "=== step 1/2: batch_eval (500 samples, mnt=256) ==="
timeout --signal=KILL 3600 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python scripts/batch_eval.py \
     --adapter_path ${ADAPTER} \
     --weight_quant nf4 --bf16_rmsnorm \
     --n_samples 500 --max_new_tokens 256 --seed 42 \
     --output ${OUT_JSON}" \
    > "${LOG_HOST}/int4_batch_eval_recovery.log" 2>&1
RC=$?
log "  batch_eval exit=${RC}"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null

sleep 15

# Step 2: PPL (WikiText + GSM8K test)
log ""
log "=== step 2/2: eval_perplexity (WikiText-2 + GSM8K test) ==="
PPL_JSON="/app/HMA_Project/results/ppl_eval/ppl__D_INT4__seed42.json"
timeout --signal=KILL 900 docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python scripts/eval_perplexity.py \
     --adapter_path ${ADAPTER} \
     --weight_quant nf4 --bf16_rmsnorm \
     --max_len 512 --gsm8k_samples 500 --wikitext_chunks 200 \
     --seed 42 --output ${PPL_JSON}" \
    > "${LOG_HOST}/int4_ppl_recovery.log" 2>&1
log "  eval_perplexity exit=$?"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results 2>/dev/null

log ""
log "===== done ====="
