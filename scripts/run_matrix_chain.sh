#!/usr/bin/env bash
# Sequential chain for the paper's B=4/L=4096 Arm 4 memory matrix.
# Each config runs in its own process to avoid CUDA allocator fragmentation.

set -euo pipefail

CONTAINER=hma-container
OUT="results/verify/matrix"
docker exec ${CONTAINER} mkdir -p "/app/HMA_Project/${OUT}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Single-line command inside docker exec so no lines get parsed as separate
# shell commands (the earlier multi-line BASE variable had that bug).
run_one() {
    local method="$1"
    local extra="${2:-}"
    local tag="${method}${extra:+_gc}"
    log "starting ${tag}"
    docker exec ${CONTAINER} bash -c "cd /app/HMA_Project && python run_experiment.py --mode memory --method ${method} ${extra} --model 3B --weight_quant bf16 --bf16_rmsnorm --seed 42 --mem_batch_size 4 --mem_seq_len 4096 --mem_steps 100 --mem_warmup 10 --checkpoint_every 50 --output_dir ${OUT}" \
        > "/home/yedam/HMA/HMA_Project/${OUT}/${tag}_run.log" 2>&1
    log "  ${tag} done"
}

# naive_fp4 already finished (results/memory__naive_fp4__..._20260814_010035.json)
run_one uniform_fp8
run_one standard --gc

log "chain complete"
