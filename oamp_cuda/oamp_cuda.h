// Copyright (c) 2025 OAMP Research Team. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

void launch_pack_4bit(
    const nv_bfloat16* inputs,
    uint8_t* packed_out,
    nv_bfloat16* scales_out,
    int num_elements,
    cudaStream_t stream
);

void launch_unpack_4bit(
    const uint8_t* packed_in,
    const nv_bfloat16* scales_in,
    nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream
);