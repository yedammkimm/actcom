/*
 * Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.
 *
 * CUDA kernels for the max-abs anchor variant, exposed to Python through
 * oamp_bilevel_ops.cpp as the extension `oamp_bilevel` and used only by
 * oamp/cuda_ops.py. The training runs in the paper do not use them.
 *
 * Every kernel handles one group of `group_size` elements per block. The
 * straight-through path has two kernels: group_analyze_kernel writes the
 * max-abs of each group, Python marks the top groups as FP8 anchors, and
 * group_fused_quantize_kernel quantizes and dequantizes every group in one
 * pass, on the e4m3 range for anchors and on [-7, 7] for the rest. The
 * packing path stores anchor groups at one byte per element
 * (fp8_byte_pack_kernel) and body groups at two elements per byte
 * (fp4_nibble_pack_kernel), with matching unpack kernels.
 *
 * Storage formats. The one-byte code is q = clamp(round(x / scale), -127, 127)
 * with scale = absmax / 127, a signed 8-bit integer stored as uint8. The
 * nibble code is q = round(x / scale) in [-7, 7] with scale = absmax / 7,
 * stored as q + 8 in [1, 15] (0 is unused); two codes share a byte, the first
 * in the low nibble.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Common constants and inline helpers

#define BLOCK_SIZE   256       // threads per block (L2 cache-line friendly)
#define FP8_SCALE_MAX 448.0f   // E4M3FN max
#define FP4_SCALE_MAX   7.0f   // signed 4-bit symmetric max
#define FP8_INT8_MAX  127.0f   // INT8 physical store max

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ nv_bfloat16 f32_to_bf16(float v)  { return __float2bfloat16(v); }


// group_analyze_kernel: the max-abs of each group, one block per group, by a
// tree reduction in shared memory.
__global__ void group_analyze_kernel(
    const nv_bfloat16* __restrict__ x,          // [num_groups, group_size]
    float*             __restrict__ absmax_out, // [num_groups]  max-abs importance
    const int group_size
) {
    // group index
    const int grp = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16* row = x + (long long)grp * group_size;

    // per-thread partial max-abs
    float local_abs_max = 0.0f;

    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        float v = bf16_to_f32(__ldg(&row[i]));
        local_abs_max   = fmaxf(local_abs_max, fabsf(v));
    }

    // tree reduction in shared memory
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

    // thread 0 writes the result
    if (tid == 0) {
        absmax_out[grp] = smem_am[0];
    }
}


// group_fused_quantize_kernel: quantize and dequantize every group in one
// pass, integers on the e4m3 range for anchor groups and on [-7, 7] for the
// rest. The PyTorch path makes six passes over memory (importance, topk, two
// quantizations and a where); this path makes two, analyze and fused_quantize.
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

    // scale: absmax / 448 for an anchor group, absmax / 7 otherwise
    const float scale     = is_fp8 ? (am / FP8_SCALE_MAX + 1e-8f)
                                   : (am / FP4_SCALE_MAX + 1e-8f);
    const float inv_scale = 1.0f / scale;

    // Per-element quantize + dequant
    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        float v  = bf16_to_f32(__ldg(&row_in[i]));
        float qv;
        if (is_fp8) {
            // anchor: round, then clamp to [-448, 448]
            qv = __float2int_rn(v * inv_scale);
            qv = fmaxf(-FP8_SCALE_MAX, fminf(FP8_SCALE_MAX, qv));
        } else {
            // body: round, then clamp to [-7, 7]
            qv = __float2int_rn(v * inv_scale);
            qv = fmaxf(-FP4_SCALE_MAX, fminf(FP4_SCALE_MAX, qv));
        }
        row_out[i] = f32_to_bf16(qv * scale);
    }
}


// fp8_byte_pack_kernel: one row per block, BF16 [N, D] to one signed 8-bit code
// per element (scale = absmax / 127), stored as uint8 [N, D].
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


// fp4_nibble_pack_kernel: one row per block, BF16 [N, D] to uint8 [N, D/2].
// D must be even; each thread reads two elements and writes one byte.
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


// fp8_byte_unpack_kernel: uint8 [N, D] back to BF16 [N, D].
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


// fp4_nibble_unpack_kernel: uint8 [N, D/2] back to BF16 [N, D].
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


// Launchers called from the C++ bindings

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
