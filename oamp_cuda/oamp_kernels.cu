// Copyright (c) 2025 OAMP Research Team. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <stdint.h>
#include <stdio.h>

#define GROUP_SIZE 16

__device__ __forceinline__ float bfloat16_to_float(nv_bfloat16 val) { return __bfloat162float(val); }
__device__ __forceinline__ nv_bfloat16 float_to_bfloat16(float val) { return __float2bfloat16(val); }

// [Kernel 1] Analyze
__global__ void analyze_block_kernel(
    const float4* __restrict__ inputs,
    uint8_t* __restrict__ meta_out,
    int32_t* __restrict__ sizes_out,
    nv_bfloat16* __restrict__ scales_out,
    float threshold,
    const int num_groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    // Load 16 BF16 elements as two float4 (128-bit aligned)
    float4 v0 = inputs[gid * 2];
    float4 v1 = inputs[gid * 2 + 1];
    nv_bfloat16* raw0 = (nv_bfloat16*)&v0;
    nv_bfloat16* raw1 = (nv_bfloat16*)&v1;

    float max_val = -1e9f;
    float min_val = 1e9f;
    float abs_max = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = bfloat16_to_float(raw0[i]);
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
        if (fabsf(val) > abs_max) abs_max = fabsf(val);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = bfloat16_to_float(raw1[i]);
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
        if (fabsf(val) > abs_max) abs_max = fabsf(val);
    }

    float range = max_val - min_val;
    
    // Outlier detection: 448.0 = E4M3FN max representable value
    bool is_outlier = (abs_max > 448.0f); 
    bool is_active = (range >= threshold);

    float scale;
    if (is_outlier || is_active) {
        // Active/Outlier mode: quantize to INT8 range [-127, 127]
        meta_out[gid] = is_outlier ? 2 : 1;
        sizes_out[gid] = 16;  // 16 bytes (1 byte per element)
        scale = abs_max / 127.0f + 1e-6f; 
    } else {
        // Sleepy mode: quantize to 4-bit range [-7, 7]
        meta_out[gid] = 0;
        sizes_out[gid] = 8;   // 8 bytes (0.5 byte per element)
        scale = abs_max / 7.0f + 1e-6f; 
    }

    scales_out[gid] = float_to_bfloat16(scale);
}

// [Kernel 2] Pack
__global__ void hybrid_pack_kernel(
    const float4* __restrict__ inputs,
    uint8_t* __restrict__ packed_buffer,
    const uint8_t* __restrict__ meta_in,
    const int32_t* __restrict__ offsets_in,
    const nv_bfloat16* __restrict__ scales_in,
    const int num_groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    int my_offset = offsets_in[gid];
    uint8_t meta_type = meta_in[gid];
    float scale = bfloat16_to_float(scales_in[gid]);
    float inv_scale = 1.0f / scale;

    float4 v0 = inputs[gid * 2];
    float4 v1 = inputs[gid * 2 + 1];
    nv_bfloat16* raw0 = (nv_bfloat16*)&v0;
    nv_bfloat16* raw1 = (nv_bfloat16*)&v1;

    if (meta_type > 0) { 
        // FP8 Packing: scale based on INT8 range (absmax/127)
        for (int i = 0; i < 8; ++i) {
            float val = bfloat16_to_float(raw0[i]);
            int8_t q = (int8_t)(val * inv_scale); 
            packed_buffer[my_offset + i] = (uint8_t)q;
        }
        for (int i = 0; i < 8; ++i) {
            float val = bfloat16_to_float(raw1[i]);
            int8_t q = (int8_t)(val * inv_scale);
            packed_buffer[my_offset + 8 + i] = (uint8_t)q;
        }
    } else {
        // FP4 Packing: two 4-bit values per byte (nibble-packed)
        for (int i = 0; i < 4; ++i) {
             float val_l = bfloat16_to_float(raw0[2*i]);
             float val_h = bfloat16_to_float(raw0[2*i+1]);
             int q_l = (int)((val_l * inv_scale) + 8.5f); q_l = (q_l < 0) ? 0 : ((q_l > 15) ? 15 : q_l);
             int q_h = (int)((val_h * inv_scale) + 8.5f); q_h = (q_h < 0) ? 0 : ((q_h > 15) ? 15 : q_h);
             packed_buffer[my_offset + i] = (uint8_t)(q_l | (q_h << 4));
        }
        for (int i = 0; i < 4; ++i) {
             float val_l = bfloat16_to_float(raw1[2*i]);
             float val_h = bfloat16_to_float(raw1[2*i+1]);
             int q_l = (int)((val_l * inv_scale) + 8.5f); q_l = (q_l < 0) ? 0 : ((q_l > 15) ? 15 : q_l);
             int q_h = (int)((val_h * inv_scale) + 8.5f); q_h = (q_h < 0) ? 0 : ((q_h > 15) ? 15 : q_h);
             packed_buffer[my_offset + 4 + i] = (uint8_t)(q_l | (q_h << 4));
        }
    }
}

// [Kernel 3] Unpack
__global__ void hybrid_unpack_kernel(
    const uint8_t* __restrict__ packed_buffer,
    const uint8_t* __restrict__ meta_in,
    const int32_t* __restrict__ offsets_in,
    const nv_bfloat16* __restrict__ scales_in,
    float4* __restrict__ output,
    const int num_groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    int my_offset = offsets_in[gid];
    uint8_t meta_type = meta_in[gid];
    float scale = bfloat16_to_float(scales_in[gid]);
    
    float4 v0, v1;
    nv_bfloat16* out0 = (nv_bfloat16*)&v0;
    nv_bfloat16* out1 = (nv_bfloat16*)&v1;

    if (meta_type > 0) { 
        // FP8 Unpacking: int8 → float → BF16
        for (int i = 0; i < 8; ++i) {
            int8_t q = (int8_t)packed_buffer[my_offset + i];
            out0[i] = float_to_bfloat16((float)q * scale);
        }
        for (int i = 0; i < 8; ++i) {
            int8_t q = (int8_t)packed_buffer[my_offset + 8 + i];
            out1[i] = float_to_bfloat16((float)q * scale);
        }
    } else {
        // FP4 Unpacking: extract nibble pairs from packed bytes
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = packed_buffer[my_offset + i];
            int q_l = byte & 0x0F; int q_h = (byte >> 4) & 0x0F;
            out0[2*i] = float_to_bfloat16((float(q_l) - 8.0f) * scale);
            out0[2*i+1] = float_to_bfloat16((float(q_h) - 8.0f) * scale);
        }
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = packed_buffer[my_offset + 4 + i];
            int q_l = byte & 0x0F; int q_h = (byte >> 4) & 0x0F;
            out1[2*i] = float_to_bfloat16((float(q_l) - 8.0f) * scale);
            out1[2*i+1] = float_to_bfloat16((float(q_h) - 8.0f) * scale);
        }
    }
    output[gid * 2] = v0;
    output[gid * 2 + 1] = v1;
}

// Launcher: orchestrates Analyze → Scan → Pack pipeline
void launch_hybrid_compress(
    const nv_bfloat16* inputs,
    uint8_t* packed_buffer,
    uint8_t* meta,
    int32_t* sizes,
    int32_t* offsets,
    nv_bfloat16* scales,
    float threshold,
    int num_elements,
    int* total_compressed_size, // Output: total compressed bytes (host pointer)
    cudaStream_t stream
) {
    int num_groups = num_elements / GROUP_SIZE;
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;

    // 1. Analyze
    analyze_block_kernel<<<grid_size, block_size, 0, stream>>>(
        (const float4*)inputs, meta, sizes, scales, threshold, num_groups
    );

    // 2. Scan (Prefix Sum)
    thrust::device_ptr<int32_t> d_sizes(sizes);
    thrust::device_ptr<int32_t> d_offsets(offsets);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_sizes, d_sizes + num_groups, d_offsets);

    // 3. Calculate total compressed size (requires host-device sync)
    //    total = offset[last] + size[last]
    if (total_compressed_size != nullptr) {
        int last_offset = 0;
        int last_size = 0;
        int last_idx = num_groups - 1;

        cudaMemcpyAsync(&last_offset, offsets + last_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_size, sizes + last_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);

        // Synchronize to read the values on host
        cudaStreamSynchronize(stream);
        
        *total_compressed_size = last_offset + last_size;
    }

    // 4. Pack
    hybrid_pack_kernel<<<grid_size, block_size, 0, stream>>>(
        (const float4*)inputs, packed_buffer, meta, offsets, scales, num_groups
    );
}

void launch_hybrid_decompress(
    const uint8_t* packed_buffer,
    const uint8_t* meta,
    const int32_t* offsets,
    const nv_bfloat16* scales,
    nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream
) {
    int num_groups = num_elements / GROUP_SIZE;
    int block_size = 256;
    int grid_size = (num_groups + block_size - 1) / block_size;

    hybrid_unpack_kernel<<<grid_size, block_size, 0, stream>>>(
        packed_buffer, meta, offsets, scales, (float4*)output, num_groups
    );
}