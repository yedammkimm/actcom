#!/usr/bin/env bash
# Run one memory measurement per (method, B, L) as its own process so CUDA
# allocator fragmentation from an earlier config never contaminates the next.
#
# All runs use Arm 4 dtype policy (--bf16_rmsnorm), which is the paper's
# canonical baseline. Fresh process per config; 100 steps + 10 warmup matches
# the legacy benchmark_native_packing_vram (26.9 s/step at B=4, L=4096, ~50 min).
#
# Usage:
#   scripts/run_memory_matrix.sh [--quick]
#     --quick : only (standard, oamp) at (B=4, L=4096) — the paper headline

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="results/verify/matrix"
docker exec hma-container mkdir -p "/app/HMA_Project/$OUT"

METHODS=(standard oamp naive_fp4 uniform_fp8)
CONFIGS=(
    "2 512"
    "2 1024"
    "2 2048"
    "4 4096"
)

if [[ "${1:-}" == "--quick" ]]; then
    METHODS=(standard oamp)
    CONFIGS=("4 4096")
fi

for method in "${METHODS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        read -r B L <<< "$cfg"
        tag="${method}_B${B}_L${L}"
        echo ""
        echo "================================================================"
        echo "  ${tag}   (Arm 4 dtype, fresh process)"
        echo "================================================================"
        docker exec hma-container bash -c "cd /app/HMA_Project && python run_experiment.py \
            --mode memory --method ${method} --model 3B --weight_quant bf16 \
            --bf16_rmsnorm --seed 42 \
            --mem_batch_size ${B} --mem_seq_len ${L} \
            --mem_steps 100 --mem_warmup 10 \
            --checkpoint_every 50 \
            --output_dir ${OUT}" 2>&1 | grep -Ev "torch/utils/_pytree|deprecated|Loading weights|Materializing|HTTP Request" | tail -20
    done
done

echo ""
echo "=== matrix complete ==="
docker exec hma-container ls -1 "/app/HMA_Project/$OUT" | grep '\.json$' | sort
