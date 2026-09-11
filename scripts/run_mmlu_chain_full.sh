#!/usr/bin/env bash
# MMLU 5-shot on the full 22-adapter cohort (follow-on to the 9-adapter chain).
#
# Scope. The pre-registered primary comparison is the Spearman correlation
# between held-out WikiText-2 perplexity and MMLU across arm A's eight runs,
# with the base model as the anchor. That is nine adapters, and it is what
# discharges the registration. Arms B, C and D were part of the wider design
# but not of the primary comparison; they are appended afterwards only if this
# run completes cleanly. Skipping the primary metric because the four cheaper
# benchmarks already agreed would be selective reporting, so it is not an
# option, but narrowing it to the registered comparison is.
#
# Protocol. The published 5-shot setup, all 14,042 test items, few-shot drawn
# from each subject's own dev rows. Nothing is subsampled. Fixed batch 16 with
# length sorting: token-budget batching was measured slower at equal accuracy
# and removed, and larger fixed batches gain nothing because the GPU is already
# saturated at roughly 1,160 tokens/s on this model.
#
# Memory. The guard now reclaims before it judges (see memory_guard). Stage 2
# died fifteen times on fragmentation that a reclaim would have released, so a
# task that reaches the soft threshold here should log reclaims and continue
# rather than fail. If a cell does fail, it fails alone and the chain steps on.
#
# Resume. One JSON per adapter; an existing file is skipped. Re-running resumes.
set -u

CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/mc_mmlu
OUT_HOST=${ROOT_HOST}/${OUT}
RUNDIR=${OUT}/_run
RUNDIR_HOST=${OUT_HOST}/_run

# hazard 2: bash re-reads a running script from disk, so hand off to a snapshot
if [[ "${1:-}" != "--go" ]]; then
    docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${RUNDIR} /app/HMA_Project/${OUT}/logs
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null   # hazard 1
    mkdir -p "${RUNDIR_HOST}" "${OUT_HOST}/logs"
    cp "${ROOT_HOST}/scripts/run_mmlu_chain_full.sh" "${RUNDIR_HOST}/chain_full.sh"
    cp "${ROOT_HOST}/scripts/eval_multichoice.py" "${RUNDIR_HOST}/eval_multichoice.py"
    echo "snapshot -> ${RUNDIR_HOST}/  (scripts/ is free to edit from here on)"
    exec bash "${RUNDIR_HOST}/chain_full.sh" --go
fi

HARNESS=/app/HMA_Project/${RUNDIR}/eval_multichoice.py
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

cleanup() {                                              # hazard 4
    docker exec ${CONTAINER} pkill -f "eval_multichoice" 2>/dev/null && \
        log "cleanup: container-side eval killed"
    return 0
}
trap cleanup EXIT INT TERM

exec 9>/tmp/hma_gpu.lock                                 # hazard 3
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

ADAPTERS=(
"base|"
"A_seed42|results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260825_193617_adapter"
"A_seed123|results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter"
"A_seed456|results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260826_104720_adapter"
"A_seed789|results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260831_224859_adapter"
"A_seed1011|results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed1011__20260901_045636_adapter"
"A_seed2024|results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed2024__20260901_083847_adapter"
"A_seed3033|results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed3033__20260901_114401_adapter"
"A_seed4042|results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260901_144026_adapter"
"D_seed42|results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260831_082311_adapter"
"D_seed123|results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260831_194822_adapter"
"D_seed456|results/naive4bit_int4/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260831_230549_adapter"
"C_seed42|results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter"
"C_seed456|results/pilot_pack4d_fp8_matrix/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260816_140828_adapter"
"B_seed42|results/pilot_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_164235_adapter"
"B_seed123|results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260822_010922_adapter"
"B_seed456|results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed456__20260822_075818_adapter"
"B_seed789|results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260901_202010_adapter"
"B_seed1011|results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed1011__20260901_230446_adapter"
"B_seed2024|results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed2024__20260902_015809_adapter"
"B_seed3033|results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed3033__20260902_045035_adapter"
"B_seed4042|results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260902_074256_adapter"
)

log "MMLU 5-shot, 14042 items, ${#ADAPTERS[@]} adapters, batch 16"
n=0
for entry in "${ADAPTERS[@]}"; do
    tag=${entry%%|*}; adapter=${entry#*|}; n=$((n+1))
    json_host="${OUT_HOST}/mmlu__${tag}.json"
    if [[ -f "${json_host}" ]]; then log "  skip ${tag}  (json present)"; continue; fi
    ap=""; [[ -n "${adapter}" ]] && ap="--adapter_path ${adapter}"
    t0=$SECONDS
    timeout --signal=KILL 28800 docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python ${HARNESS} \
         --base_model meta-llama/Llama-3.2-3B-Instruct \
         --weight_quant nf4 --bf16_rmsnorm ${ap} \
         --tasks mmlu --batch_size 16 \
         --output ${OUT}/mmlu__${tag}.json" \
        > "${OUT_HOST}/logs/mmlu__${tag}.log" 2>&1
    rc=$?; el=$(( SECONDS - t0 ))
    if (( rc == 137 || rc == 124 )); then
        log "  TIMEOUT ${tag} after ${el}s -- reaping container process"
        docker exec ${CONTAINER} pkill -f "eval_multichoice" 2>/dev/null; sleep 10
    fi
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    acc=$(grep -o "acc_sum=[0-9.]*" "${OUT_HOST}/logs/mmlu__${tag}.log" | tail -1)
    peak=$(grep -o "peak=[0-9.]* GB" "${OUT_HOST}/logs/mmlu__${tag}.log" | tail -1)
    rec=$(grep -o "reclaim #[0-9]*" "${OUT_HOST}/logs/mmlu__${tag}.log" | tail -1)
    log "  ${n}/${#ADAPTERS[@]} ${tag}  exit=${rc}  ${el}s  ${acc:-acc=?}  ${peak:-peak=?}  ${rec:-no reclaim}"
done
log "===== done: $(ls ${OUT_HOST}/mmlu__*.json 2>/dev/null | wc -l) / ${#ADAPTERS[@]} cells ====="
