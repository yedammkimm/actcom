#!/usr/bin/env bash
# Downstream multiple-choice capability, 22 adapters x 6 benchmarks.
#
# The question. In-distribution metrics do not separate these arms: GSM8K
# accuracy and GSM8K perplexity are flat across A, B, C and D, while held-out
# WikiText perplexity spreads from 15.78 to 18.43 within arm A alone. If that
# spread is real capability rather than measurement noise, it should show up on
# benchmarks the adapters were never trained on. If it does not, held-out
# perplexity is no more predictive than the in-distribution metrics, and the
# paper has to stop treating it as a capability proxy.
#
# Ordering. Wall-clock is not the binding constraint here -- the memory cgroup
# is -- so nothing is subsampled and MMLU keeps its published 5-shot protocol
# on all 14,042 items. What is optimised instead is the order, so that stopping
# at any point leaves a complete and interpretable subset:
#
#   stage 1   ARC-C, ARC-E, PIQA, WinoGrande     cheap, four tasks, ~7 h
#   stage 2   HellaSwag                          the large 0-shot task
#   stage 3   MMLU 5-shot                        the primary metric, ~57 h
#
# and within every stage the adapters run base -> A(8) -> D(3) -> C(2) -> B(8),
# so arm A -- the one whose perplexity spread motivates the experiment -- is
# complete first, with the base model already measured as an anchor.
#
# Resume. One JSON per (adapter, task); an existing file is skipped. The server
# has died twice during this project, so a crash costs one cell, not one
# adapter and not one stage. Re-running this script resumes.
#
# PRE-REGISTRATION, fixed before the first cell ran.
#
#   primary metric     MMLU accuracy, all 14,042 items, 5-shot, acc_norm.
#                      Every MMLU choice is a single token (" A" / " B" / " C"
#                      / " D"), so the per-token mean equals the sum and
#                      acc_norm and acc coincide; both are reported anyway.
#   primary comparison Spearman correlation between WikiText-2 perplexity and
#                      MMLU accuracy over the eight runs of arm A.
#   secondary          mean MMLU difference between the four damaged and the
#                      four safe runs of arm A.
#   confirmatory       the other five tasks. No threshold is applied to them
#                      and no verdict is taken from them; they are described.
#   positive control   whether arm D separates from arm C. If the arm that is
#                      damaged on every other metric -- gradient norm 586,
#                      100% of steps clipped, 3/3 above tau, in-distribution
#                      perplexity +3% -- does not separate on MMLU, then MMLU
#                      cannot detect this kind of damage, and a null result
#                      inside arm A is a limit of the metric rather than
#                      evidence that no damage exists. Without the positive
#                      control the null is uninterpretable.
#
# MMLU is named the primary metric in advance because choosing, after the fact,
# whichever of six benchmarks separates would be selection. This project has
# already met that trap once, with NarrativeQA and GovReport, and handled it by
# declaring them corroborating rather than independent tests.
#
# Batching. Fixed count, 16 on MMLU and 64 on the short tasks, with length
# sorting. Token-budget batching was implemented and measured slower (1.43 vs
# 1.68 items/s on MMLU at a 150,000-token budget, same accuracy, same item
# hash) and has been removed. The harness raises at 60 GB reserved, leaving
# 20 GB under the container's 80 GiB limit, so a task that grows out of budget
# fails alone and the chain steps past it.

set -u

CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/HMA_Project
OUT=results/mc_downstream
OUT_HOST=${ROOT_HOST}/${OUT}
RUNDIR=${OUT}/_run
RUNDIR_HOST=${OUT_HOST}/_run

# ---------------------------------------------------------------- snapshot
# Hazard type 2: bash re-reads a script from disk as it runs, so editing this
# file mid-chain corrupts the running job. Copy this script and the harness
# into the run directory and hand off to the copy; the originals are then free
# to edit while a four-day chain is in flight.
if [[ "${1:-}" != "--go" ]]; then
    docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${RUNDIR} \
                                      /app/HMA_Project/${OUT}/logs
    # Hazard type 1: the container creates these as root and the host shell
    # then cannot open its own redirect targets. chown BEFORE any host write.
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null
    mkdir -p "${RUNDIR_HOST}" "${OUT_HOST}/logs"
    cp "${ROOT_HOST}/scripts/run_mc_chain.sh"     "${RUNDIR_HOST}/chain.sh"
    cp "${ROOT_HOST}/scripts/eval_multichoice.py" "${RUNDIR_HOST}/eval_multichoice.py"
    echo "snapshot -> ${RUNDIR_HOST}/  (edit scripts/ freely from here on)"
    exec bash "${RUNDIR_HOST}/chain.sh" --go
fi

HARNESS=/app/HMA_Project/${RUNDIR}/eval_multichoice.py
log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# Hazard type 4: timeout kills the docker exec client, not the container-side
# job, so the container process has to be reaped explicitly.
cleanup() {
    docker exec ${CONTAINER} pkill -f "eval_multichoice" 2>/dev/null && \
        log "cleanup: container-side eval killed"
    return 0
}
trap cleanup EXIT INT TERM

# Hazard type 3: queue behind any other chain rather than racing it.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

# ---------------------------------------------------------------- adapters
# Selected from the full-length 3B cohort only (n_train 7473, 2 epochs); the
# 2000x1 short runs are a different training length and are not comparable.
# Where an arm has two perplexity measurements for one seed, the row cited in
# the paper's WikiText table is the one taken.
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

# task|batch|timeout_seconds
STAGE1=("arc_challenge|64|3600" "arc_easy|64|3600" "piqa|64|3600" "winogrande|64|3600")
STAGE2=("hellaswag|64|14400")
STAGE3=("mmlu|16|28800")

run_cell() {                                   # $1 tag  $2 adapter  $3 task|bs|to
    local tag=$1 adapter=$2 spec=$3
    local task=${spec%%|*} rest=${spec#*|}
    local bs=${rest%%|*} to=${rest#*|}
    local json_host="${OUT_HOST}/mc__${tag}__${task}.json"

    if [[ -f "${json_host}" ]]; then
        log "  skip ${tag} ${task}  (json present)"
        return 0
    fi
    local ap=""
    [[ -n "${adapter}" ]] && ap="--adapter_path ${adapter}"

    local t0=$SECONDS
    timeout --signal=KILL "${to}" docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python ${HARNESS} \
         --base_model meta-llama/Llama-3.2-3B-Instruct \
         --weight_quant nf4 --bf16_rmsnorm ${ap} \
         --tasks ${task} --batch_size ${bs} \
         --output ${OUT}/mc__${tag}__${task}.json" \
        > "${OUT_HOST}/logs/mc__${tag}__${task}.log" 2>&1
    local rc=$? el=$(( SECONDS - t0 ))

    if (( rc == 137 || rc == 124 )); then
        log "  TIMEOUT ${tag} ${task} after ${el}s -- reaping container process"
        docker exec ${CONTAINER} pkill -f "eval_multichoice" 2>/dev/null
        sleep 10
    fi
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${OUT} 2>/dev/null

    local acc peak
    acc=$(grep -o "acc_sum=[0-9.]*" "${OUT_HOST}/logs/mc__${tag}__${task}.log" | tail -1)
    peak=$(grep -o "peak=[0-9.]* GB" "${OUT_HOST}/logs/mc__${tag}__${task}.log" | tail -1)
    log "  ${tag} ${task}  exit=${rc}  ${el}s  ${acc:-acc=?}  ${peak:-peak=?}"
    return 0                                   # never abort the chain on one cell
}

run_stage() {                                  # $1 label, rest: task specs
    local label=$1; shift
    log ""
    log "===== ${label} ====="
    for entry in "${ADAPTERS[@]}"; do
        local tag=${entry%%|*} adapter=${entry#*|}
        for spec in "$@"; do
            run_cell "${tag}" "${adapter}" "${spec}"
        done
    done
    log "===== ${label} complete ====="
}

log "22 adapters, stages 1-3, resuming from ${OUT}/"
run_stage "stage 1 -- ARC-C / ARC-E / PIQA / WinoGrande" "${STAGE1[@]}"
run_stage "stage 2 -- HellaSwag"                          "${STAGE2[@]}"
run_stage "stage 3 -- MMLU 5-shot"                        "${STAGE3[@]}"

log ""
log "===== all stages done ====="
log "  cells: $(ls ${OUT_HOST}/mc__*.json 2>/dev/null | wc -l) / 132"
