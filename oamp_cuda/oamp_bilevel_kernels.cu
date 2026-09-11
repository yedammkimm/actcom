/*
 * Copyright (c) 2025 OAMP Research Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * oamp_bilevel_kernels.cu
 * ======================
 * OAMP Bi-Level Mixed-Precision CUDA Kernels
 *
 * Differences from oamp_kernels.cu
 * --------------------------------
 * Legacy   : Block-level (16 elements/group), Tri-Level (FP4/FP8-active/FP8-outlier)
 *            Threshold-based decision, threshold is a layer-global scalar
 *
 * Current  : Group-level (128 elements/group), Bi-Level (FP8 anchor 20% / FP4 body 80%)
 *            Max-abs based top-K routing, per-group quantization
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  QAT Path (STE simulation during training)                              │
 * │                                                                      │
 * │  [Kernel A] group_analyze_kernel                                     │
 * │    Grid=(num_groups), Block=256                                       │
 * │    Per-group: max-abs  →  importance[num_groups]                      │
 * │                                                                      │
 * │  [Python]  topk → fp8_mask[num_groups]                                │
 * │                                                                      │
 * │  [Kernel B] group_fused_quantize_kernel                               │
 * │    Grid=(num_groups), Block=256                                       │
 * │    Per-group: FP8(absmax/448) or FP4(absmax/7) → dequant → BF16      │
 * │    1 memory pass (vs 6 separate PyTorch ops)                         │
 * │                                                                      │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  Physical Pack Path (Gradient Checkpointing / Inference)             │
 * │                                                                      │
 * │  [Kernel C] fp8_byte_pack_kernel                                     │
 * │    BF16[N_fp8, D]  →  uint8[N_fp8, D]   (INT8 absmax/127)           │
 * │                                                                      │
 * │  [Kernel D] fp4_nibble_pack_kernel                                   │
 * │    BF16[N_fp4, D]  →  uint8[N_fp4, D/2] (2 nibbles per byte)        │
 * │                                                                      │
 * │  [Kernel E] fp8_byte_unpack_kernel                                   │
 * │  [Kernel F] fp4_nibble_unpack_kernel                                 │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * Nibble Format (FP4)
 * -------------------
 *   Quantize : q = round(val / scale) ∈ [-7, 7]
 *   Store    : stored = q + 8 ∈ [1, 15]  (0 = reserved / zero)
 *   Packing  : byte = (lo & 0xF) | ((hi & 0xF) << 4)
 *   Unpack   : q = (stored & 0xF) - 8,  val ≈ q * scale
 *
 * FP8 Physical Format (INT8-based, 1 byte per element)
 * -------------------------------------------------------
 *   scale    : absmax / 127
 *   Quantize : q = clamp(round(val / scale), -127, 127)  →  int8 stored as uint8
 *   Unpack   : val ≈ (int8)q * scale
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Common constants / inline helpers
// ──────────────────────────────────────────────────────────────────────────────

#define BLOCK_SIZE   256       // threads per block (L2 cache-line friendly)
#define FP8_SCALE_MAX 448.0f   // E4M3FN max
#define FP4_SCALE_MAX   7.0f   // signed 4-bit symmetric max
#define FP8_INT8_MAX  127.0f   // INT8 physical store max

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ nv_bfloat16 f32_to_bf16(float v)  { return __float2bfloat16(v); }


// ══════════════════════════════════════════════════════════════════════════════
// Kernel A : group_analyze_kernel
//
//  For each group (= one row, group_size elements):
//    Compute max-abs importance score.
//  Computed via shared memory reduction in a single pass.
//
//  Grid : (num_groups)     — one block per group
//  Block: BLOCK_SIZE threads
//  Smem : BLOCK_SIZE * sizeof(float)   (for absmax)
// ══════════════════════════════════════════════════════════════════════════════
__global__ void group_analyze_kernel(
    const nv_bfloat16* __restrict__ x,          // [num_groups, group_size]
    float*             __restrict__ absmax_out, // [num_groups]  max-abs importance
    const int group_size
) {
    // group index
    const int grp = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16* row = x + (long long)grp * group_size;

    // Phase 1: Per-thread partial max-abs
    float local_abs_max = 0.0f;

    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        float v = bf16_to_f32(__ldg(&row[i]));
        local_abs_max   = fmaxf(local_abs_max, fabsf(v));
    }

    // Phase 2: Shared Memory Tree Reduction
    __shared__ float smem_am[BLOCK_SIZE];

    smem_am[tid] = local_abs_max;
    __syncthreads();

    // warp-sized tree reduction (unrolled for power-of-2 BLOCK_SIZE)
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem_am[tid]  = fmaxf(smem_am[tid], smem_am[tid + s]);
        }
        __syncthreads();
    }

    // Phase 3: Thread 0 writes final result
    if (tid == 0) {
        absmax_out[grp] = smem_am[0];
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel B : group_fused_quantize_kernel
//
//  Applies FP8 or FP4 simulated quantization per group based on fp8_mask.
//  No intermediate tensors — single memory pass.
//
//  Performance vs PyTorch:
//    before : importance → topk → fp8_quantize(all) → fp4_quantize(all)
//             → where(mask, fp8, fp4)     ← 6 memory round trips
//    after  : analyze + fused_quantize    ← 2 memory round trips
//
//  Grid : (num_groups)
//  Block: BLOCK_SIZE threads
// ══════════════════════════════════════════════════════════════════════════════
__global__ void group_fused_quantize_kernel(
    const nv_bfloat16* __restrict__ x,       // [num_groups, group_size]  input
          nv_bfloat16* __restrict__ out,      // [num_groups, group_size]  output
    const bool*        __restrict__ fp8_mask, // [num_groups]  True=FP8, False=FP4
    const float*       __restrict__ absmax,   // [num_groups]  absmax per group
    const int group_size
) {
    const int grp = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16* row_in  = x   + (long long)grp * group_size;
          nv_bfloat16* row_out = out + (long long)grp * group_size;

    const bool is_fp8 = fp8_mask[grp];
    const float am    = absmax[grp];

    /* Scale computation:
     * FP8 : absmax / 448  (E4M3FN range scaling — non-uniform interval simulation)
     * FP4 : absmax / 7    (signed 4-bit symmetric)
     */
    const float scale     = is_fp8 ? (am / FP8_SCALE_MAX + 1e-8f)
                                   : (am / FP4_SCALE_MAX + 1e-8f);
    const float inv_scale = 1.0f / scale;

    // Per-element quantize + dequant
    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        float v  = bf16_to_f32(__ldg(&row_in[i]));
        float qv;
        if (is_fp8) {
            // E4M3FN simulation: clamp to [-448, 448], then round
            qv = __float2int_rn(v * inv_scale);
            qv = fmaxf(-FP8_SCALE_MAX, fminf(FP8_SCALE_MAX, qv));
        } else {
            // FP4 absmax symmetric: clamp to [-7, 7]
            qv = __float2int_rn(v * inv_scale);
            qv = fmaxf(-FP4_SCALE_MAX, fminf(FP4_SCALE_MAX, qv));
        }
        row_out[i] = f32_to_bf16(qv * scale);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel C : fp8_byte_pack_kernel  (Physical Packing)
//
//  Physically packs N rows to INT8 (absmax/127 scale).
//  BF16[N, D]  ->  uint8[N, D]   (1 byte per element)
//
//  Grid : (N)
//  Block: BLOCK_SIZE threads
// ══════════════════════════════════════════════════════════════════════════════
__global__ void fp8_byte_pack_kernel(
    const nv_bfloat16* __restrict__ x,      // [N_fp8, D]
          uint8_t*      __restrict__ packed, // [N_fp8, D]
    const float*        __restrict__ absmax, // [N_fp8]
    const int D
) {
    const int tok = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16* row_in  = x      + (long long)tok * D;
          uint8_t*     row_out = packed + (long long)tok * D;

    const float scale     = absmax[tok] / FP8_INT8_MAX + 1e-8f;
    const float inv_scale = 1.0f / scale;

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = bf16_to_f32(__ldg(&row_in[i]));
        int q   = __float2int_rn(v * inv_scale);
        q       = max(-127, min(127, q));
        // Store int8 as uint8 (bit-cast, preserves sign)
        row_out[i] = (uint8_t)((int8_t)q);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel D : fp4_nibble_pack_kernel  (Physical Packing)
//
//  Physically packs N rows to nibble-packed uint8.
//  BF16[N, D]  ->  uint8[N, D/2]  (2 nibbles per byte)
//
//  D must be even.
//
//  Grid : (N)
//  Block: BLOCK_SIZE threads
//  Each thread reads 2 elements (4 bytes) and stores 1 nibble-packed byte.
// ══════════════════════════════════════════════════════════════════════════════
__global__ void fp4_nibble_pack_kernel(
    const nv_bfloat16* __restrict__ x,      // [N_fp4, D]
          uint8_t*      __restrict__ packed, // [N_fp4, D/2]
    const float*        __restrict__ absmax, // [N_fp4]
    const int D
) {
    const int tok = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16* row_in  = x      + (long long)tok * D;
          uint8_t*     row_out = packed + (long long)tok * (D / 2);

    const float scale     = absmax[tok] / FP4_SCALE_MAX + 1e-8f;
    const float inv_scale = 1.0f / scale;

    // Output element count = D/2 (number of nibble pairs)
    const int D_half = D / 2;

    for (int i = tid; i < D_half; i += BLOCK_SIZE) {
        // Pack 2 BF16 elements into one byte
        float v_lo = bf16_to_f32(__ldg(&row_in[2 * i    ]));
        float v_hi = bf16_to_f32(__ldg(&row_in[2 * i + 1]));

        // Quantize to [-7, 7] then shift to [1, 15] unsigned offset
        int q_lo = __float2int_rn(v_lo * inv_scale);
        int q_hi = __float2int_rn(v_hi * inv_scale);

        q_lo = max(-7, min(7, q_lo)) + 8;  // [1, 15]
        q_hi = max(-7, min(7, q_hi)) + 8;  // [1, 15]

        // Nibble packing: lo -> lower 4 bits, hi -> upper 4 bits
        row_out[i] = (uint8_t)((q_lo & 0xF) | ((q_hi & 0xF) << 4));
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel E : fp8_byte_unpack_kernel  (Physical Unpacking)
//
//  uint8[N_fp8, D]  →  BF16[N_fp8, D]
// ══════════════════════════════════════════════════════════════════════════════
__global__ void fp8_byte_unpack_kernel(
    const uint8_t*     __restrict__ packed, // [N_fp8, D]
          nv_bfloat16* __restrict__ out,    // [N_fp8, D]
    const float*       __restrict__ absmax, // [N_fp8]
    const int D
) {
    const int tok = blockIdx.x;
    const int tid = threadIdx.x;

    const uint8_t*     row_in  = packed + (long long)tok * D;
          nv_bfloat16* row_out = out    + (long long)tok * D;

    const float scale = absmax[tok] / FP8_INT8_MAX + 1e-8f;

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        int8_t q = (int8_t)__ldg(&row_in[i]);
        row_out[i] = f32_to_bf16((float)q * scale);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel F : fp4_nibble_unpack_kernel  (Physical Unpacking)
//
//  uint8[N_fp4, D/2]  →  BF16[N_fp4, D]
// ══════════════════════════════════════════════════════════════════════════════
__global__ void fp4_nibble_unpack_kernel(
    const uint8_t*     __restrict__ packed, // [N_fp4, D/2]
          nv_bfloat16* __restrict__ out,    // [N_fp4, D]
    const float*       __restrict__ absmax, // [N_fp4]
    const int D
) {
    const int tok = blockIdx.x;
    const int tid = threadIdx.x;

    const uint8_t*     row_in  = packed + (long long)tok * (D / 2);
          nv_bfloat16* row_out = out    + (long long)tok * D;

    const float scale  = absmax[tok] / FP4_SCALE_MAX + 1e-8f;
    const int D_half   = D / 2;

    for (int i = tid; i < D_half; i += BLOCK_SIZE) {
        uint8_t byte = __ldg(&row_in[i]);

        // Nibble extraction
        int q_lo = (byte & 0x0F);          // lower 4 bits
        int q_hi = (byte >> 4) & 0x0F;    // upper 4 bits

        // Remove unsigned offset: [1,15] -> [-7,7]
        float v_lo = (float)(q_lo - 8) * scale;
        float v_hi = (float)(q_hi - 8) * scale;

        row_out[2 * i    ] = f32_to_bf16(v_lo);
        row_out[2 * i + 1] = f32_to_bf16(v_hi);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Launcher functions (called from C++ bindings)
// ══════════════════════════════════════════════════════════════════════════════

extern "C" {

void launch_group_analyze(
    const nv_bfloat16* x,
    float*             absmax_out,
    int                num_groups,
    int                group_size,
    cudaStream_t       stream
) {
    group_analyze_kernel<<<num_groups, BLOCK_SIZE, 0, stream>>>(
        x, absmax_out, group_size
    );
}

void launch_group_fused_quantize(
    const nv_bfloat16* x,
          nv_bfloat16* out,
    const bool*        fp8_mask,
    const float*       absmax,
    int                num_groups,
    int                group_size,
    cudaStream_t       stream
) {
    group_fused_quantize_kernel<<<num_groups, BLOCK_SIZE, 0, stream>>>(
        x, out, fp8_mask, absmax, group_size
    );
}

void launch_fp8_byte_pack(
    const nv_bfloat16* x,
          uint8_t*     packed,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
) {
    fp8_byte_pack_kernel<<<N, BLOCK_SIZE, 0, stream>>>(x, packed, absmax, D);
}

void launch_fp4_nibble_pack(
    const nv_bfloat16* x,
          uint8_t*     packed,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
) {
    fp4_nibble_pack_kernel<<<N, BLOCK_SIZE, 0, stream>>>(x, packed, absmax, D);
}

void launch_fp8_byte_unpack(
    const uint8_t*     packed,
          nv_bfloat16* out,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
) {
    fp8_byte_unpack_kernel<<<N, BLOCK_SIZE, 0, stream>>>(packed, out, absmax, D);
}

void launch_fp4_nibble_unpack(
    const uint8_t*     packed,
          nv_bfloat16* out,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
) {
    fp4_nibble_unpack_kernel<<<N, BLOCK_SIZE, 0, stream>>>(packed, out, absmax, D);
}

} // extern "C"
