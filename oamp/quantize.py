# Copyright (c) 2025 OAMP Research Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
oamp.quantize
=============
FP4 / FP8 E4M3FN quantization kernels (pure PyTorch, with STE).

Two granularity modes:
  - Per-token  (legacy): single scale per token (dim=-1)
  - Per-group  (default): scale per GROUP_SIZE block of elements

Per-group quantization (GROUP_SIZE=128) provides finer-grained scaling,
reducing quantization error especially for activations with non-uniform
magnitude distributions across the hidden dimension.

Design Principles  (cf. Paper §3.2)
-----------------
* FP4  : 4-bit symmetric absmax, per-group scaling (signed → [-7, 7])
* FP8  : Native ``torch.float8_e4m3fn``, per-group absmax scaling (max 448)
* STE  : Straight-Through Estimator — forward emits quantized values,
         backward passes gradients through unchanged.

Public API
----------
fp4_quantize(x, group_size=128) → Tensor   (FP4 per-group, STE)
fp8_quantize(x, group_size=128) → Tensor   (FP8 per-group, STE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FP4PerGroupSTE",
    "FP8PerGroupSTE",
    "fp4_quantize",
    "fp8_quantize",
    "GROUP_SIZE",
]

# Default group size for per-group quantization (matches all experiments)
GROUP_SIZE: int = 128


# ──────────────────────────────────────────────────────────────────────────────
# FP4 — Per-group 4-bit absmax symmetric quantization + STE (default)
# ──────────────────────────────────────────────────────────────────────────────

class FP4PerGroupSTE(torch.autograd.Function):
    """
    Per-group absmax symmetric 4-bit quantization with STE.

    Each group of ``group_size`` elements gets its own scale factor:
      scale = absmax(group) / 7.0
      x_q   = round(x / scale).clamp(-7, 7)
      x_out = x_q * scale  (dequantize, same dtype as input)

    Handles non-divisible hidden dimensions via zero-padding.
    Backward: gradient passes through unchanged (STE).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int = 128) -> torch.Tensor:
        x_f = x.float()
        D = x_f.shape[-1]
        if D % group_size != 0:
            pad_size = group_size - (D % group_size)
            x_f = F.pad(x_f, (0, pad_size))
            D_padded = x_f.shape[-1]
        else:
            pad_size = 0
            D_padded = D
        flat_shape = x_f.shape[:-1]
        x_grouped = x_f.reshape(*flat_shape, D_padded // group_size, group_size)
        absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        x_q = (x_grouped / scale).round().clamp(-7, 7)
        x_out = (x_q * scale).reshape(*flat_shape, D_padded)
        if pad_size > 0:
            x_out = x_out[..., :D]
        return x_out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class FP8PerGroupSTE(torch.autograd.Function):
    """
    Per-group FP8 E4M3FN quantization with Straight-Through Estimator.

    Each group of ``group_size`` elements gets its own scale factor:
      scale  = absmax(group) / 448
      x_fp8  = cast_to_E4M3FN(x / scale)
      x_out  = x_fp8 * scale  (dequantize, same dtype as input)

    Handles non-divisible hidden dimensions via zero-padding.
    Backward: gradient passes through unchanged (STE).
    """

    FP8_MAX: float = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int = 128) -> torch.Tensor:
        x_f = x.float()
        D = x_f.shape[-1]
        if D % group_size != 0:
            pad_size = group_size - (D % group_size)
            x_f = F.pad(x_f, (0, pad_size))
            D_padded = x_f.shape[-1]
        else:
            pad_size = 0
            D_padded = D
        flat_shape = x_f.shape[:-1]
        x_grouped = x_f.reshape(*flat_shape, D_padded // group_size, group_size)
        absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / FP8PerGroupSTE.FP8_MAX
        x_scaled = (x_grouped / scale).to(torch.bfloat16)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        x_out = (x_fp8.to(torch.bfloat16).float() * scale).reshape(*flat_shape, D_padded)
        if pad_size > 0:
            x_out = x_out[..., :D]
        return x_out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fp4_quantize(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Per-group FP4 quantization with STE (default). Input: any shape with last dim D."""
    return FP4PerGroupSTE.apply(x, group_size)


def fp8_quantize(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Per-group FP8 E4M3FN quantization with STE (default). Input: any shape with last dim D."""
    return FP8PerGroupSTE.apply(x, group_size)



