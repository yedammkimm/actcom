#!/usr/bin/env bash
# Uniform re-eval of all 3B §5 arms with stop_strings + mnt=512.
# Existing training-time evals used no stop_strings + mnt=256 which capped
# every arm at ~100% truncation and made in-arm comparisons brittle. This
# sweep aligns the eval protocol so the §5 threshold table stands on
# matched conditions.

set -u
CONTAINER=hma-container
OUT_DIR=/home/yedam/HMA/HMA_Project/results/reeval_stop_mnt512
mkdir -p ${OUT_DIR}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/results/reeval_stop_mnt512

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# adapter_relpath  label  seed
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

log "===== reeval sweep begin (${#ADAPTERS[@]} adapters, stop+mnt=512, ~60 min) ====="

for entry in "${ADAPTERS[@]}"; do
    IFS='|' read -r rel label seed <<< "$entry"
    if [[ ! -d "/home/yedam/HMA/HMA_Project/${rel}" ]]; then
        log "  SKIP ${label}/${seed}: adapter missing"
        continue
    fi
    tag="reeval__${label}__seed${seed}__stop_mnt512"
    log_file="${OUT_DIR}/${tag}.log"
    out_json="/app/HMA_Project/results/reeval_stop_mnt512/${tag}.json"
    if [[ -f "${OUT_DIR}/${tag}.json" ]]; then
        log "  SKIP ${tag} (done)"
        continue
    fi
    log "starting ${tag}"
    timeout --signal=KILL 3600 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python scripts/batch_eval.py \
         --adapter_path ${rel} \
         --weight_quant nf4 --bf16_rmsnorm \
         --n_samples 500 --max_new_tokens 512 --seed ${seed} \
         --output ${out_json}" \
        > "${log_file}" 2>&1
    rc=$?
    log "  ${tag} exit=${rc}"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/results/reeval_stop_mnt512 2>/dev/null
    sleep 15
done

log ""
log "===== sweep done ====="
