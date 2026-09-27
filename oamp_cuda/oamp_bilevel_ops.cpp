/*
 * Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.
 *
 * PyTorch bindings for oamp_bilevel_kernels.cu, built as the extension
 * `oamp_bilevel` by setup.py in this directory. Each function checks dtype, device and
 * layout, allocates the output and launches the kernel on the current stream:
 * analyze and fused_quantize for the straight-through path, fp8_pack,
 * fp4_pack, fp8_unpack and fp4_unpack for the packing path. Only
 * oamp/cuda_ops.py calls these.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Launchers defined in oamp_bilevel_kernels.cu

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


// analyze: BF16 [num_groups, group_size] to float32 [num_groups], the max-abs
// of each group.
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


// fused_quantize: x BF16 [num_groups, group_size], fp8_mask bool [num_groups]
// (true = anchor group), absmax float32 [num_groups]; returns the quantized
// and dequantized BF16 tensor of the same shape.
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


// fp8_pack: anchor rows BF16 [N, D] with absmax float32 [N] to uint8 [N, D],
// one byte per element.
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


// fp4_pack: body rows BF16 [N, D] with absmax float32 [N] to uint8 [N, D/2],
// two elements per byte.
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


// fp8_unpack: uint8 [N, D] with absmax float32 [N] back to BF16 [N, D]; D is
// passed for validation.
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


// fp4_unpack: uint8 [N, D/2] with absmax float32 [N] back to BF16 [N, D].
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


// Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "OAMP Bi-Level CUDA Kernels";

    // straight-through path
    m.def("analyze",
          &group_analyze,
          "Compute per-group max-abs importance score.\n"
          "  x: BF16[num_groups, group_size] -> float32[num_groups]");

    m.def("fused_quantize",
          &group_fused_quantize,
          "Fused per-group quantize and dequantize in one pass over memory.
"
          "  x: BF16[num_groups, gs], fp8_mask: bool[num_groups], absmax: float32[num_groups]\n"
          "  -> BF16[num_groups, gs]");

    // packing path
    m.def("fp8_pack",   &fp8_pack,
          "Pack FP8 anchor tokens to INT8 bytes.\n"
          "  x: BF16[N,D], absmax: float[N] -> uint8[N,D]");

    m.def("fp4_pack",   &fp4_pack,
          "Pack FP4 body tokens to nibble-packed bytes.\n"
          "  x: BF16[N,D], absmax: float[N] -> uint8[N,D/2]");

    m.def("fp8_unpack", &fp8_unpack,
          "Unpack INT8 bytes to BF16.\n"
          "  packed: uint8[N,D], absmax: float[N], D: int -> BF16[N,D]");

    m.def("fp4_unpack", &fp4_unpack,
          "Unpack nibble-packed bytes to BF16.\n"
          "  packed: uint8[N,D/2], absmax: float[N], D: int -> BF16[N,D]");
}
