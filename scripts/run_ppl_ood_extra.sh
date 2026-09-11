#!/usr/bin/env bash
# Extra OOD corpora sweep — narrativeqa + govreport — 9 adapters.
# Skips gsm8k/wikitext2 (already measured); only adds new corpora at 200 chunks
# each. Same protocol as WikiText-2 for apples-to-apples cross-corpus compare.

set -u
CONTAINER=hma-container
OUT_DIR=/home/yedam/HMA/HMA_Project/results/ppl_ood_extra
mkdir -p ${OUT_DIR}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/results/ppl_ood_extra

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# (adapter_relpath | label | seed)
ADAPTERS=(
    "results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260831_082311_adapter|D_INT4|42"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260825_193617_adapter|A_4DFP4|42"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter|A_4DFP4|123"
    "results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260826_104720_adapter|A_4DFP4|456"
    "results/pilot_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_164235_adapter|B_4DFP8|42"
    "results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260822_010922_adapter|B_4DFP8|123"
    "results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260822_075818_adapter|B_4DFP8|456"
    "results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter|C_STD|42"
    "results/pilot_pack4d_fp8_matrix/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260816_140828_adapter|C_STD|456"
)

log "===== OOD extra sweep — narrativeqa + govreport, ${#ADAPTERS[@]} adapters ====="

for entry in "${ADAPTERS[@]}"; do
    IFS='|' read -r rel label seed <<< "$entry"
    if [[ ! -d "/home/yedam/HMA/HMA_Project/${rel}" ]]; then
        log "  SKIP ${label}/${seed}: adapter missing"
        continue
    fi
    tag="ppl_ood__${label}__seed${seed}"
    log_file="${OUT_DIR}/${tag}.log"
    out_json="/app/HMA_Project/results/ppl_ood_extra/${tag}.json"
    if [[ -f "${OUT_DIR}/${tag}.json" ]]; then
        log "  SKIP ${tag} (done)"
        continue
    fi
    log "starting ${tag}"
    timeout --signal=KILL 900 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/eval_perplexity.py \
         --adapter_path ${rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --max_len 512 --seed ${seed} \
         --skip_gsm8k --skip_wikitext \
         --extra_corpora narrativeqa,govreport --extra_chunks 200 \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${tag} TIMEOUT"
        docker exec ${CONTAINER} pkill -9 -f "eval_perplexity" 2>/dev/null
    fi
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/ppl_ood_extra 2>/dev/null
    sleep 10
done

log ""
log "===== sweep done. Judgment first, then decide next experiments. ====="
