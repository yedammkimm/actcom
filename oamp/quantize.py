# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.

"""Per-group 4-bit and FP8 quantizers with a straight-through gradient.

Both work on groups of `group_size` elements along the last axis with one
absmax scale per group; a partial trailing group is zero-padded. The 4-bit
quantizer rounds x / scale onto the symmetric integer grid -7..7 (the code
calls this FP4). The FP8 quantizer casts x / scale to torch.float8_e4m3fn,
whose largest value is 448. Both return dequantized values in the input
dtype, and both pass the gradient through unchanged.

Only oamp.bilevel and oamp.cuda_ops use these; the pack hooks have their own
quantizers.
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


# 4-bit: per-group absmax symmetric quantization with a straight-through gradient

class FP4PerGroupSTE(torch.autograd.Function):
    """The 4-bit quantizer as an autograd.Function.

    Per group, scale = absmax / 7, q = clamp(round(x / scale), -7, 7), and the
    output is q * scale in the input dtype. The backward pass returns the
    incoming gradient unchanged.
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
    """The FP8 (e4m3fn) quantizer as an autograd.Function.

    Per group, scale = absmax / 448, x / scale is cast to float8_e4m3fn, and the
    output is the cast value times the scale, in the input dtype. The backward
    pass returns the incoming gradient unchanged.
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
    """Per-group 4-bit quantize-dequantize of `x` (any shape, groups along the
    last dim) with a straight-through gradient.
    """
    return FP4PerGroupSTE.apply(x, group_size)


def fp8_quantize(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Per-group FP8 quantize-dequantize of `x` (any shape, groups along the last
    dim) with a straight-through gradient.
    """
    return FP8PerGroupSTE.apply(x, group_size)



