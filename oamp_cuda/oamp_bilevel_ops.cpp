/*
 * Copyright (c) 2025 OAMP Research Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * oamp_bilevel_ops.cpp
 * ===================
 * PyTorch C++ bindings — exposes oamp_bilevel CUDA kernels to Python.
 *
 * Module name: oamp_bilevel
 *
 * Public Functions
 * ---------
 *  [QAT Path]
 *   analyze(x, group_size)                → absmax (per-group importance)
 *   fused_quantize(x, fp8_mask, absmax, group_size) → quantized_x
 *
 *  [Physical Pack Path]
 *   fp8_pack  (x_fp8, absmax_fp8)    → packed uint8  [N, D]
 *   fp4_pack  (x_fp4, absmax_fp4)    → packed uint8  [N, D/2]
 *   fp8_unpack(packed, absmax, D)    → BF16  [N, D]
 *   fp4_unpack(packed, absmax, D)    → BF16  [N, D]
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// ── Launcher declarations (defined in oamp_bilevel_kernels.cu) ───────────────────

extern "C" {

void launch_group_analyze(
    const nv_bfloat16* x,
    float*             absmax_out,
    int                num_groups,
    int                group_size,
    cudaStream_t       stream
);

void launch_group_fused_quantize(
    const nv_bfloat16* x,
          nv_bfloat16* out,
    const bool*        fp8_mask,
    const float*       absmax,
    int                num_groups,
    int                group_size,
    cudaStream_t       stream
);

void launch_fp8_byte_pack(
    const nv_bfloat16* x,
          uint8_t*     packed,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
);

void launch_fp4_nibble_pack(
    const nv_bfloat16* x,
          uint8_t*     packed,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
);

void launch_fp8_byte_unpack(
    const uint8_t*     packed,
          nv_bfloat16* out,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
);

void launch_fp4_nibble_unpack(
    const uint8_t*     packed,
          nv_bfloat16* out,
    const float*       absmax,
    int                N,
    int                D,
    cudaStream_t       stream
);

} // extern "C"


// ══════════════════════════════════════════════════════════════════════════════
// [QAT Path]  analyze
//
//  Input  : BF16 tensor x [num_groups, group_size] (pre-reshaped)
//  Output : float32 [num_groups] — per-group max-abs importance
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor group_analyze(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),               "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.is_contiguous(),         "x must be contiguous");
    TORCH_CHECK(x.dim() == 2,             "x must be 2D [num_groups, group_size]");

    int num_groups = x.size(0);
    int group_size = x.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor absmax_out = torch::empty({num_groups}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_group_analyze(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        absmax_out.data_ptr<float>(),
        num_groups,
        group_size,
        stream
    );

    return absmax_out;
}


// ══════════════════════════════════════════════════════════════════════════════
// [QAT Path]  fused_quantize
//
//  Input
//    x        : BF16  [num_groups, group_size]
//    fp8_mask : bool  [num_groups]  True = FP8 anchor group
//    absmax   : float32 [num_groups]  per-group max-abs
//
//  Output : BF16 [num_groups, group_size] — quantized+dequantized
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor group_fused_quantize(
    torch::Tensor x,
    torch::Tensor fp8_mask,
    torch::Tensor absmax
) {
    TORCH_CHECK(x.is_cuda() && fp8_mask.is_cuda() && absmax.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(fp8_mask.scalar_type() == torch::kBool, "fp8_mask must be bool");
    TORCH_CHECK(absmax.scalar_type() == torch::kFloat,  "absmax must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [num_groups, group_size]");

    x        = x.contiguous();
    fp8_mask = fp8_mask.contiguous();
    absmax   = absmax.contiguous();

    int num_groups = x.size(0);
    int group_size = x.size(1);

    torch::Tensor out = torch::empty_like(x);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_group_fused_quantize(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<      nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        fp8_mask.data_ptr<bool>(),
        absmax.data_ptr<float>(),
        num_groups,
        group_size,
        stream
    );

    return out;
}


// ══════════════════════════════════════════════════════════════════════════════
// [Physical Pack]  fp8_pack
//
//  Input  : BF16  [N, D] — FP8 anchor tokens (already gathered)
//          absmax float32 [N]
//  Output : uint8 [N, D]  (1 byte per element: INT8 encoding)
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor fp8_pack(torch::Tensor x, torch::Tensor absmax) {
    TORCH_CHECK(x.is_cuda() && absmax.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(absmax.scalar_type() == torch::kFloat, "absmax must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, D]");

    x      = x.contiguous();
    absmax = absmax.contiguous();

    int N = x.size(0);
    int D = x.size(1);

    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    torch::Tensor packed = torch::empty({N, D}, opts_u8);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fp8_byte_pack(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        packed.data_ptr<uint8_t>(),
        absmax.data_ptr<float>(),
        N, D, stream
    );

    return packed;
}


// ══════════════════════════════════════════════════════════════════════════════
// [Physical Pack]  fp4_pack
//
//  Input  : BF16  [N, D] — FP4 body tokens
//          absmax float32 [N]
//  Output : uint8 [N, D/2]  (nibble packed: 2 elements per byte)
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor fp4_pack(torch::Tensor x, torch::Tensor absmax) {
    TORCH_CHECK(x.is_cuda() && absmax.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(absmax.scalar_type() == torch::kFloat, "absmax must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, D]");
    TORCH_CHECK(x.size(1) % 2 == 0, "D must be even for nibble packing");

    x      = x.contiguous();
    absmax = absmax.contiguous();

    int N = x.size(0);
    int D = x.size(1);

    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    torch::Tensor packed = torch::empty({N, D / 2}, opts_u8);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fp4_nibble_pack(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        packed.data_ptr<uint8_t>(),
        absmax.data_ptr<float>(),
        N, D, stream
    );

    return packed;
}


// ══════════════════════════════════════════════════════════════════════════════
// [Physical Unpack]  fp8_unpack
//
//  Input  : uint8  [N, D]
//          absmax float32 [N]
//          D      int (hidden_dim, for validation)
//  Output : BF16   [N, D]
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor fp8_unpack(torch::Tensor packed, torch::Tensor absmax, int64_t D) {
    TORCH_CHECK(packed.is_cuda() && absmax.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8,  "packed must be uint8");
    TORCH_CHECK(absmax.scalar_type() == torch::kFloat,  "absmax must be float32");
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) == D, "packed shape mismatch");

    packed = packed.contiguous();
    absmax = absmax.contiguous();

    int N = packed.size(0);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device());
    torch::Tensor out = torch::empty({N, D}, opts_bf16);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fp8_byte_unpack(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        absmax.data_ptr<float>(),
        N, (int)D, stream
    );

    return out;
}


// ══════════════════════════════════════════════════════════════════════════════
// [Physical Unpack]  fp4_unpack
//
//  Input  : uint8  [N, D/2]
//          absmax float32 [N]
//          D      int
//  Output : BF16   [N, D]
// ══════════════════════════════════════════════════════════════════════════════
torch::Tensor fp4_unpack(torch::Tensor packed, torch::Tensor absmax, int64_t D) {
    TORCH_CHECK(packed.is_cuda() && absmax.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8,  "packed must be uint8");
    TORCH_CHECK(absmax.scalar_type() == torch::kFloat,  "absmax must be float32");
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) == D / 2,
                "packed shape mismatch (expected [N, D/2])");

    packed = packed.contiguous();
    absmax = absmax.contiguous();

    int N = packed.size(0);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device());
    torch::Tensor out = torch::empty({N, D}, opts_bf16);

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fp4_nibble_unpack(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        absmax.data_ptr<float>(),
        N, (int)D, stream
    );

    return out;
}


// ══════════════════════════════════════════════════════════════════════════════
// PYBIND11 Registration
// ══════════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "OAMP Bi-Level CUDA Kernels";

    // QAT Path
    m.def("analyze",
          &group_analyze,
          "Compute per-group max-abs importance score.\n"
          "  x: BF16[num_groups, group_size] → float32[num_groups]");

    m.def("fused_quantize",
          &group_fused_quantize,
          "Fused per-group bi-level quantize+dequant. 1 memory pass (vs 6 in PyTorch).\n"
          "  x: BF16[num_groups, gs], fp8_mask: bool[num_groups], absmax: float32[num_groups]\n"
          "  → BF16[num_groups, gs]");

    // Physical Pack/Unpack
    m.def("fp8_pack",   &fp8_pack,
          "Pack FP8 anchor tokens to INT8 bytes.\n"
          "  x: BF16[N,D], absmax: float[N] → uint8[N,D]");

    m.def("fp4_pack",   &fp4_pack,
          "Pack FP4 body tokens to nibble-packed bytes.\n"
          "  x: BF16[N,D], absmax: float[N] → uint8[N,D/2]");

    m.def("fp8_unpack", &fp8_unpack,
          "Unpack INT8 bytes to BF16.\n"
          "  packed: uint8[N,D], absmax: float[N], D: int → BF16[N,D]");

    m.def("fp4_unpack", &fp4_unpack,
          "Unpack nibble-packed bytes to BF16.\n"
          "  packed: uint8[N,D/2], absmax: float[N], D: int → BF16[N,D]");
}
