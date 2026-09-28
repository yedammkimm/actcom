#!/usr/bin/env bash
# Determinism probe: does the same-seed divergence of Section 5.4 disappear
# under PyTorch's deterministic mode?
#
# Question. Two runs of one configuration at seed 123 agree bit-for-bit in
# loss for optimizer steps 1 and 2 and separate at step 3 (Table 10:
# 1.295640 / 1.641697 / 1.472919 against 1.473335). The paper attributes the
# split to non-deterministic reduction order in CUDA kernels and says the
# attribution is untested. This chain tests it: the same configuration is
# launched twice under default kernels and twice under
# torch.use_deterministic_algorithms(True) with a fixed cuBLAS workspace, for
# 20 optimizer steps on the paper's own data order (n_train 7473, 2 epochs, the
# permutation of np.random.RandomState(seed); only the step count is limited,
# so the losses at steps 1 and 2 must reproduce Table 10). Every cell is one
# process, so no CUDA allocator state is shared between the two members of a
# pair. Each record stores every micro-step loss, a sha256 of every LoRA
# gradient before clipping and of every LoRA parameter after the update, at
# every optimizer step.
#
# Prior observations that shaped the design (synthetic tensors, this GPU,
# 2026-09-28, before this file was committed; the same ops are re-measured by
# scripts/audit/determinism_ops_probe.py as stage 0 and stored):
#   - under default kernels the flash SDPA backward returns different bits on
#     every repeat (30/30, one bf16 ulp); matmul, the LoRA chain, cross-entropy,
#     rms_norm, silu, bitsandbytes NF4 and PagedAdamW8bit, and clip_grad_norm_
#     repeated bit-for-bit 30/30;
#   - under deterministic mode the flash backward repeats bit-for-bit and is
#     not refused by torch 2.12; the mem-efficient and cuDNN backends are
#     refused ("No available kernel"), the default dispatch for the model's
#     shapes is FLASH;
#   - the three pairs in the paper differ in more than the run: the seed-123
#     and seed-789 pairs straddle a driver update (580.159.03 to 580.173.02)
#     and every pair straddles a code commit; the seed-4042 pair separates at
#     step 3 under one driver.
#
# Prediction, fixed before the first cell runs:
#   The arm-A pair under default kernels separates before or at optimizer
#   step 3, as in Table 10. Under deterministic mode with the flash backend
#   the A, B and C pairs agree in every micro-step loss, every gradient hash
#   and every parameter hash through all 20 steps.
#
# Reading rules, fixed before the first cell runs:
#   outcome 1  deterministic pairs identical on every ladder for all three
#              arms, default pair separates: over 20 steps the kernels are the
#              only source of the divergence. The paper's attribution stands
#              for the step at which the split appears; the claim is about 20
#              steps, not about 3,736, and says nothing about how the split
#              grows later. "Property of the run, not the seed" becomes "not
#              predictable from the seed".
#   outcome 2  a deterministic pair separates: a source deterministic mode does
#              not cover is present. The ladder (micro-step loss, gradient
#              hash, parameter hash) says whether it is in the forward, the
#              backward or the optimizer; the op-level record names the
#              candidates. The attribution is withdrawn, not replaced.
#   outcome 3  deterministic mode refuses an operation: the error text names a
#              nondeterminism source present in every run of the paper. The
#              cell is repeated with --deterministic_warn_only, and if pairs
#              still separate, with --sdpa_backend math for arms B and C (arm A
#              under MATH stores the (B,H,L,L) softmax as a rank-4 tensor and is
#              not the paper's arm A; that is stated wherever it is reported).
#   positive control fails  the default pair does not separate within 20
#              steps: the agreement of the deterministic pairs then carries no
#              information on its own. Stage 2 (below) is run and reported; if
#              the pair still does not separate, the probe reports that the
#              Table 10 split did not reproduce between back-to-back launches
#              and the attribution stays open.
#
# Stages. 0: op-level probe, default and deterministic, one process each.
# 1: positive control, arm A under default kernels, two launches, then the
# gate line from determinism_compare.py --gate is logged. 2: the remaining ten
# cells (A deterministic, B and C in both modes). 3, only if the positive
# control failed: the four arm-A cells again at 200 optimizer steps.
# 4: the four arm-A cells at seed 4042. Cells whose record exists are skipped,
# so the chain resumes after a crash.
#
# Provenance. Every cell must record env.git_dirty=false. The chain refuses to
# start on a dirty tree, and it commits each record (JSON and checkpoint
# sidecar; adapters and logs are ignored by git) before the next cell starts,
# so every launch sees a tree that is exactly one commit. Nothing tracked may
# be edited while the chain runs.
#
# Guards (each traces to an incident in this project): the chain copies itself
# to a run directory and executes the copy, so the file in the repository can
# be edited without corrupting the running shell; an exclusive flock on
# /tmp/hma_gpu.lock; a refusal to start while nvidia-smi or the container shows
# a compute process; timeout --signal=KILL with the container-side process
# killed by PID from a trap (never a host-side pkill -f); chown to 4051:4051
# after every container-side write and before any host-side write.
#
# Usage:  scripts/run_determinism_probe.sh            (STEPS=20 STEPS_LONG=200 by default)

set -u
CONTAINER=hma-container
ROOT_HOST=/home/yedam/HMA/actcom
ROOT_CONT=/app/actcom
OUT=results/determinism
OUT_HOST=${ROOT_HOST}/${OUT}
STEPS=${STEPS:-20}
STEPS_LONG=${STEPS_LONG:-200}
RUN_ROOT=${RUN_ROOT:-/tmp/hma_determinism_runs}

# Run from a snapshot copy, never from the file in the repository.
if [[ -z "${HMA_CHAIN_SNAPSHOT:-}" ]]; then
    RUN_DIR=${RUN_ROOT}/$(date +%Y%m%d_%H%M%S)
    mkdir -p "${RUN_DIR}"
    cp "$0" "${RUN_DIR}/run_determinism_probe.sh"
    HMA_CHAIN_SNAPSHOT=1 exec bash "${RUN_DIR}/run_determinism_probe.sh" "$@"
fi

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

mkdir -p "${OUT_HOST}"
docker exec ${CONTAINER} mkdir -p ${ROOT_CONT}/${OUT}
docker exec ${CONTAINER} chown -R 4051:4051 ${ROOT_CONT}/${OUT} 2>/dev/null

CURRENT_NAME=""
cleanup() {
    if [[ -n "${CURRENT_NAME}" ]]; then
        for pid in $(docker exec ${CONTAINER} pgrep -f "python .*${CURRENT_NAME}\.json" 2>/dev/null); do
            log "cleanup: killing container pid ${pid} (${CURRENT_NAME})"
            docker exec ${CONTAINER} kill -9 "${pid}" 2>/dev/null
        done
    fi
    docker exec ${CONTAINER} chown -R 4051:4051 ${ROOT_CONT}/${OUT} 2>/dev/null
}
trap cleanup EXIT INT TERM

# Serialise against any other chain; the lock releases when fd 9 closes.
exec 9>/tmp/hma_gpu.lock
log "waiting for the GPU lock (/tmp/hma_gpu.lock)"
flock -x 9
log "GPU lock acquired"

# A free lock is not a free GPU: refuse to start next to a running job.
busy_host=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
busy_cont=$(docker exec ${CONTAINER} pgrep -fc "python (run_experiment|scripts/)" 2>/dev/null || true)
if (( ${busy_host:-0} > 0 || ${busy_cont:-0} > 0 )); then
    log "GPU busy (host compute apps=${busy_host}, container python jobs=${busy_cont}); refusing to start"
    exit 1
fi

# The tree must be exactly one commit.
if [[ -n "$(git -C ${ROOT_HOST} status --porcelain)" ]]; then
    log "working tree is dirty; commit first:"; git -C ${ROOT_HOST} status --short | head -20
    exit 1
fi
HEAD=$(git -C ${ROOT_HOST} rev-parse HEAD)
log "===== determinism probe at commit ${HEAD} (STEPS=${STEPS}, STEPS_LONG=${STEPS_LONG}) ====="

commit_record() {
    # commit_record <name> <what>
    local name="$1" what="$2"
    git -C ${ROOT_HOST} add -- "${OUT}/${name}.json" 2>/dev/null
    [[ -f "${OUT_HOST}/${name}.checkpoints.jsonl" ]] && git -C ${ROOT_HOST} add -- "${OUT}/${name}.checkpoints.jsonl"
    if git -C ${ROOT_HOST} commit -q -m "determinism probe: ${name} (as run)

${what} written by scripts/run_determinism_probe.sh at commit ${HEAD:0:12}.
Committed by the chain before the next cell starts, so every launch sees a
clean tree and env.git_dirty stays false.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"; then
        log "  committed ${name}"
    else
        log "  COMMIT FAILED for ${name}; the next cell will record git_dirty=true"
    fi
}

run_ops_probe() {
    local mode="$1" name="ops_probe__${mode}"
    if [[ -f "${OUT_HOST}/${name}.json" ]]; then log "  ${name} exists; skipping"; return 0; fi
    local envflags=()
    [[ ${mode} == deterministic ]] && envflags=(-e CUBLAS_WORKSPACE_CONFIG=:4096:8)
    CURRENT_NAME=${name}
    log "--- ${name} ---"
    timeout --signal=KILL 1800 docker exec "${envflags[@]}" ${CONTAINER} bash -c \
        "cd ${ROOT_CONT} && python scripts/audit/determinism_ops_probe.py --mode ${mode} --output ${OUT}/${name}.json" \
        > "${OUT_HOST}/${name}.log" 2>&1
    log "  ${name} exit=$?"
    CURRENT_NAME=""
    docker exec ${CONTAINER} chown -R 4051:4051 ${ROOT_CONT}/${OUT} 2>/dev/null
    grep -v "_pytree\|register_constant" "${OUT_HOST}/${name}.log" | sed 's/^/    /' | head -30
    [[ -f "${OUT_HOST}/${name}.json" ]] && commit_record "${name}" "Op-level record (${mode} kernels)"
}

run_cell() {
    # run_cell <arm A|B|C> <default|deterministic> <rep> [steps] [seed]
    local arm="$1" mode="$2" rep="$3" steps="${4:-$STEPS}" seed="${5:-123}"
    local backend=default
    [[ ${mode} == deterministic ]] && backend=flash
    local name="${arm}__${mode}__${backend}"
    [[ ${seed} != 123 ]] && name="${name}__seed${seed}"
    [[ ${steps} != "${STEPS}" ]] && name="${name}__steps${steps}"
    name="${name}__rep${rep}"
    if [[ -f "${OUT_HOST}/${name}.json" ]]; then log "  ${name} exists; skipping"; return 0; fi

    local mflags
    case ${arm} in
        A) mflags="--method naive_fp4 --body_encoding e2m1 --pack_4d_mode fp4" ;;
        B) mflags="--method naive_fp4 --body_encoding e2m1 --pack_4d_mode fp8" ;;
        C) mflags="--method standard --body_encoding e2m1 --pack_4d_mode fp4" ;;
        *) log "unknown arm ${arm}"; return 1 ;;
    esac
    local dflags="" envflags=()
    if [[ ${mode} == deterministic ]]; then
        dflags="--deterministic --sdpa_backend flash"
        envflags=(-e CUBLAS_WORKSPACE_CONFIG=:4096:8)
    fi

    CURRENT_NAME=${name}
    log "--- ${name} (arm ${arm}, ${mode}, seed ${seed}, ${steps} optimizer steps) ---"
    # Table 17 settings, as in scripts/run_axis4_s123_locked.sh; only the step
    # count is limited and the evaluation skipped.
    timeout --signal=KILL 3600 docker exec "${envflags[@]}" ${CONTAINER} bash -c \
        "cd ${ROOT_CONT} && python run_experiment.py \
         --mode accuracy ${mflags} --model 3B \
         --weight_quant nf4 --bf16_rmsnorm \
         --group_size 128 --task gsm8k --seed ${seed} \
         --n_train 7473 --epochs 2 --lr 2e-4 \
         --batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
         --optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
         --lora_dropout 0.05 --eval_samples 500 --eval_max_new_tokens 256 \
         --checkpoint_every 1 --mem_abort_gb 100 \
         --max_opt_steps ${steps} --skip_eval --param_hash_every 1 --record_micro_losses \
         ${dflags} \
         --output ${OUT}/${name}.json" \
        > "${OUT_HOST}/${name}.log" 2>&1
    local rc=$?
    if (( rc == 137 || rc == 124 )); then
        log "  ${name} TIMEOUT (rc=${rc}); killing the container-side process by pid"
        for pid in $(docker exec ${CONTAINER} pgrep -f "python .*${name}\.json" 2>/dev/null); do
            docker exec ${CONTAINER} kill -9 "${pid}" 2>/dev/null
        done
    fi
    log "  ${name} exit=${rc}"
    CURRENT_NAME=""
    docker exec ${CONTAINER} chown -R 4051:4051 ${ROOT_CONT}/${OUT} 2>/dev/null
    sleep 5

    if [[ -f "${OUT_HOST}/${name}.json" ]]; then
        python3 - "${OUT_HOST}/${name}.json" <<'PY' | sed 's/^/    /'
import json, sys
d = json.load(open(sys.argv[1])); e = d['env']; r = d['results']
print(f"status={d['status']} git_commit={e['git_commit'][:12]} git_dirty={e['git_dirty']} "
      f"det={e.get('deterministic_algorithms')} cublas={e.get('cublas_workspace_config')} "
      f"sdpa_default={e.get('sdpa_default_dispatch')} steps_done={d['throughput'].get('steps_completed')}")
sl = r.get('step_losses') or []
print('step losses 1-3:', [repr(x) for x in sl[:3]])
print('param hash step 1:', (r.get('param_hash_trace') or [[None, '']])[0][1][:16], ' peak_reserved_gb:', d['memory'].get('peak_reserved_gb'))
if d.get('error'): print('error:', str(d['error'])[:300])
PY
        commit_record "${name}" "Run record and checkpoint sidecar"
    else
        log "  ${name}: no record written"
    fi
}

# ---- stage 0: op-level probe ------------------------------------------------
log "stage 0: op-level probe"
run_ops_probe default
run_ops_probe deterministic

# ---- stage 1: positive control ---------------------------------------------
log "stage 1: positive control, arm A under default kernels"
run_cell A default 1
run_cell A default 2
GATE=$(python3 ${ROOT_HOST}/scripts/audit/determinism_compare.py --gate \
        "${OUT_HOST}/A__default__default__rep1.json" "${OUT_HOST}/A__default__default__rep2.json" 2>&1)
log "positive control: ${GATE}"

# ---- stage 2: the remaining cells --------------------------------------------
log "stage 2: remaining cells"
run_cell A deterministic 1
run_cell A deterministic 2
run_cell B default 1
run_cell B default 2
run_cell B deterministic 1
run_cell B deterministic 2
run_cell C default 1
run_cell C default 2
run_cell C deterministic 1
run_cell C deterministic 2

# ---- stage 3: only if the positive control did not separate -----------------
if [[ "${GATE}" == *"first_diff_micro=none"* && "${GATE}" == *"first_diff_loss=none"* \
      && "${GATE}" == *"first_diff_grad=none"* && "${GATE}" == *"first_diff_param=none"* ]]; then
    log "stage 3: the default pair did not separate within ${STEPS} steps; arm A at ${STEPS_LONG} steps"
    run_cell A default 1 "${STEPS_LONG}"
    run_cell A default 2 "${STEPS_LONG}"
    run_cell A deterministic 1 "${STEPS_LONG}"
    run_cell A deterministic 2 "${STEPS_LONG}"
else
    log "stage 3 skipped: the default pair separated (${GATE})"
fi

# ---- stage 4: seed 4042, arm A -----------------------------------------------
log "stage 4: arm A at seed 4042"
run_cell A default 1 "${STEPS}" 4042
run_cell A default 2 "${STEPS}" 4042
run_cell A deterministic 1 "${STEPS}" 4042
run_cell A deterministic 2 "${STEPS}" 4042

# ---- collect deterministic-mode errors and warnings, write the summary -------
{
    echo "# Deterministic-mode errors and warnings, collected from the cell logs by scripts/run_determinism_probe.sh"
    echo "# commit ${HEAD}  written $(date -u +%FT%TZ)"
    echo "# pattern: 'does not have a deterministic implementation' | 'Deterministic behavior was enabled' | 'RuntimeError.*determin' | 'UserWarning.*determin' | 'warn_only'"
    for f in "${OUT_HOST}"/*__deterministic__*.log "${OUT_HOST}"/ops_probe__deterministic.log; do
        [[ -f "$f" ]] || continue
        echo "## $(basename "$f")"
        hits=$(grep -n -E "does not have a deterministic implementation|Deterministic behavior was enabled|RuntimeError.*determin|UserWarning.*determin|warn_only" "$f" || true)
        if [[ -n "${hits}" ]]; then echo "${hits}"; else echo "(no deterministic-mode error or warning in this log)"; fi
    done
} > "${OUT_HOST}/nondeterministic_ops.txt"
python3 ${ROOT_HOST}/scripts/audit/determinism_compare.py --dir "${OUT_HOST}" --write "${OUT_HOST}/summary.md"
log "===== done; summary at ${OUT}/summary.md (not committed by the chain: review, then commit with the manifest) ====="
