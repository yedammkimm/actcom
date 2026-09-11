// Copyright (c) 2025 OAMP Research Team. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

extern void launch_hybrid_compress(
    const nv_bfloat16* inputs, uint8_t* packed_buffer, uint8_t* meta, int32_t* sizes, int32_t* offsets,
    nv_bfloat16* scales, float threshold, int num_elements, int* total_size, cudaStream_t stream);

extern void launch_hybrid_decompress(
    const uint8_t* packed_buffer, const uint8_t* meta, const int32_t* offsets, const nv_bfloat16* scales,
    nv_bfloat16* output, int num_elements, cudaStream_t stream);

// [Hybrid Pack]
std::vector<torch::Tensor> oamp_pack(torch::Tensor x, float threshold) {
    int num_elements = x.numel();
    int num_groups = num_elements / 16;
    
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());

    torch::Tensor meta = torch::empty({num_groups}, options_u8);
    torch::Tensor sizes = torch::empty({num_groups}, options_i32);
    torch::Tensor offsets = torch::empty({num_groups}, options_i32);
    torch::Tensor scales = torch::empty({num_groups}, options_bf16);

    // 1. Allocate temporary buffer (max possible size)
    torch::Tensor temp_packed = torch::empty({num_elements}, options_u8);

    // 2. Run compression and compute actual compressed size
    int total_bytes = 0;
    launch_hybrid_compress(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        temp_packed.data_ptr<uint8_t>(),
        meta.data_ptr<uint8_t>(),
        sizes.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        reinterpret_cast<nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
        threshold,
        num_elements,
        &total_bytes, // Receives actual compressed size
        at::cuda::getCurrentCUDAStream()
    );

    // 3. Create exact-size tensor (shrink to actual compressed size)
    // Note: .slice() alone doesn't free memory — copy to new tensor
    torch::Tensor packed_exact = torch::empty({total_bytes}, options_u8);
    packed_exact.copy_(temp_packed.slice(0, 0, total_bytes));

    // 4. Return packed data + metadata
    // temp_packed is freed when scope ends (ref count drops to zero)
    return {packed_exact, meta, offsets, scales};
}

// [Hybrid Unpack]
torch::Tensor oamp_unpack(torch::Tensor packed, torch::Tensor meta, torch::Tensor offsets, torch::Tensor scales, std::vector<int64_t> shape) {
    int64_t num_elements = 1;
    for (auto s : shape) num_elements *= s;
    
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device());
    torch::Tensor output = torch::empty({num_elements}, options_bf16);

    launch_hybrid_decompress(
        packed.data_ptr<uint8_t>(),
        meta.data_ptr<uint8_t>(),
        offsets.data_ptr<int32_t>(),
        reinterpret_cast<const nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        (int)num_elements,
        at::cuda::getCurrentCUDAStream()
    );

    return output.view(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack", &oamp_pack, "OAMP Hybrid Pack");
    m.def("unpack", &oamp_unpack, "OAMP Hybrid Unpack");
}