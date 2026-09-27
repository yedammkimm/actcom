# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.

"""CUDA-backed version of the anchor-variant quantizer, with a PyTorch fallback.

The extension `oamp_bilevel`, built from oamp_cuda/ with
`python setup_bilevel.py build_ext --inplace`, provides a per-group max-abs
kernel and a fused quantize-dequantize kernel; cuda_available() reports
whether it loaded. Every function here falls back to oamp.bilevel and
oamp.quantize when it did not, so the same computation runs either way and
only the speed differs.

Two paths are exposed. bilevel_quantize_cuda is the straight-through
quantizer, a drop-in for group_bilevel_quantize. bilevel_pack and
bilevel_unpack store the anchor groups at one byte per element and the body
groups at two elements per byte in a BilevelPacked container, which is what
a checkpointing scheme would keep between forward and backward; the PyTorch
fallback of that path produces the same values but does not shrink memory.

The training runs in the paper use neither path; they compress saved
tensors through oamp.pack_hooks.
"""

from __future__ import annotations

import sys
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F

from .quantize import fp4_quantize, fp8_quantize, GROUP_SIZE

__all__ = [
    "cuda_available",
    "analyze",
    "fused_quantize",
    "bilevel_quantize_cuda",
    "bilevel_pack",
    "bilevel_unpack",
    "BilevelPacked",
]

# CUDA extension import

_BILEVEL_EXT = None
_CUDA_READY  = False

def _try_load_extension():
    global _BILEVEL_EXT, _CUDA_READY

    # Build artifacts reside in oamp_cuda/ directory
    ext_dir = os.path.join(os.path.dirname(__file__), "..", "oamp_cuda")
    ext_dir = os.path.abspath(ext_dir)
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)

    try:
        import oamp_bilevel  # type: ignore
        _BILEVEL_EXT = oamp_bilevel
        _CUDA_READY  = True
    except ImportError:
        _BILEVEL_EXT = None
        _CUDA_READY  = False

_try_load_extension()


def cuda_available() -> bool:
    """Return True if the oamp_bilevel extension is built and importable. When
    False, every function in this module runs its PyTorch fallback.
    """
    return _CUDA_READY


# Per-group analysis

def analyze(x: torch.Tensor, group_size: int = GROUP_SIZE):
    """Return the max-abs of each group of `x` (..., D) as a float32 tensor of
    shape (num_groups,), from a single kernel launch when the extension is
    available and from PyTorch otherwise.
    """
    D = x.shape[-1]
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        x = F.pad(x, (0, pad_size))
    x_flat = x.reshape(-1, group_size)

    if _CUDA_READY and x.is_cuda:
        return _BILEVEL_EXT.analyze(x_flat.contiguous())

    # PyTorch fallback
    return x_flat.float().abs().amax(dim=-1)


# Per-group fused quantize

class _FusedQuantizeSTE(torch.autograd.Function):
    """autograd.Function for the fused kernel: each group is quantized and
    dequantized as FP8 or 4-bit according to the group mask, and the backward
    pass returns the incoming gradient unchanged.
    """
    @staticmethod
    def forward(ctx, x_padded, fp8_group_mask, group_size):
        x_flat = x_padded.reshape(-1, group_size)

        if _CUDA_READY and x_padded.is_cuda:
            # Single CUDA kernel: 1 memory pass
            absmax = x_flat.float().abs().amax(dim=-1)
            result = _BILEVEL_EXT.fused_quantize(
                x_flat.contiguous(), fp8_group_mask.contiguous(),
                absmax.contiguous()
            )
            return result.reshape(x_padded.shape)

        # PyTorch fallback: 6 memory passes
        q_fp8 = fp8_quantize(x_padded, group_size)
        q_fp4 = fp4_quantize(x_padded, group_size)
        mask = fp8_group_mask.unsqueeze(-1).expand(-1, group_size)
        result = torch.where(mask,
                             q_fp8.reshape(-1, group_size),
                             q_fp4.reshape(-1, group_size))
        return result.reshape(x_padded.shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # STE: gradient flows to x only


def fused_quantize(
    x:              torch.Tensor,
    fp8_group_mask: torch.Tensor,
    group_size:     int = GROUP_SIZE,
) -> torch.Tensor:
    """Quantize and dequantize `x` (BF16, (..., D) with D a multiple of
    `group_size`; pad first) group by group, FP8 where `fp8_group_mask` is True
    and 4-bit elsewhere. Returns BF16 of the same shape, with a straight-through
    gradient.
    """
    return _FusedQuantizeSTE.apply(x, fp8_group_mask, group_size)


# Unified entry point

def bilevel_quantize_cuda(
    hidden_states: torch.Tensor,
    fp8_ratio: float = 0.20,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """The anchor-variant quantizer on the CUDA path: score groups by max-abs,
    mark the top `fp8_ratio` as anchors, and run the fused quantize. Takes a
    BF16 tensor (..., D) and returns one of the same shape.
    """
    D = hidden_states.shape[-1]

    # Pad if needed
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        x_padded = F.pad(hidden_states, (0, pad_size))
    else:
        pad_size = 0
        x_padded = hidden_states

    x_flat = x_padded.reshape(-1, group_size)
    num_groups = x_flat.shape[0]
    K = max(1, int(num_groups * fp8_ratio))

    if K >= num_groups:
        return fp8_quantize(hidden_states, group_size)

    # 1. per-group importance (max-abs)
    with torch.no_grad():
        importance = x_flat.float().abs().amax(dim=-1)
        _, topk_idx = importance.topk(K)
        fp8_mask = torch.zeros(num_groups, dtype=torch.bool,
                               device=hidden_states.device)
        fp8_mask.scatter_(0, topk_idx, True)

    # 2. fused quantize
    result = fused_quantize(x_padded, fp8_mask, group_size)

    if pad_size > 0:
        result = result[..., :D]

    return result.to(hidden_states.dtype)


# Physical pack / unpack

@dataclass
class BilevelPacked:
    """Physically packed anchor-variant activations.

        fp8_packed    uint8 [N_fp8, group_size]       anchor groups, one byte per element
        fp4_packed    uint8 [N_fp4, group_size / 2]   body groups, two elements per byte
        fp8_absmax    float32 [N_fp8]                 per-group scale of the anchor groups
        fp4_absmax    float32 [N_fp4]                 per-group scale of the body groups
        fp8_indices   int64 [N_fp8]                   which groups are anchors
        fp4_indices   int64 [N_fp4]                   which groups are body
        orig_shape    the (B, L, D) shape of the original tensor
        group_size    elements per group
    """
    fp8_packed:  torch.Tensor
    fp4_packed:  torch.Tensor
    fp8_absmax:  torch.Tensor
    fp4_absmax:  torch.Tensor
    fp8_indices: torch.Tensor
    fp4_indices: torch.Tensor
    orig_shape:  tuple
    group_size:  int = GROUP_SIZE

    def memory_bytes(self) -> int:
        """Bytes held by the packed payloads and scales."""
        return (
            self.fp8_packed.numel()   # 1 byte/elem
            + self.fp4_packed.numel() # 0.5 byte/elem (nibble-packed)
            + self.fp8_absmax.numel() * 4
            + self.fp4_absmax.numel() * 4
            + self.fp8_indices.numel() * 8
            + self.fp4_indices.numel() * 8
        )

    def savings_vs_bf16(self) -> float:
        """Percent saved against storing the original tensor in BF16."""
        B, L, D = self.orig_shape
        bf16_bytes = B * L * D * 2
        our_bytes  = self.memory_bytes()
        return (1.0 - our_bytes / bf16_bytes) * 100.0


def bilevel_pack(
    x: torch.Tensor,
    fp8_ratio: float = 0.20,
    group_size: int = GROUP_SIZE,
) -> BilevelPacked:
    """Pack a BF16 tensor (B, L, D) into a BilevelPacked container: score groups
    by max-abs, mark the top `fp8_ratio` as anchors, and store anchors at one
    byte per element and the body at two elements per byte. With the extension
    the packing is done by the fp8_pack and fp4_pack kernels and the memory is
    actually smaller; the PyTorch fallback produces the same values but keeps
    uint8 tensors of full size, so it is for checking results, not for saving
    memory.
    """
    assert x.dim() == 3, "x must be (B, L, D)"
    B, L, D = x.shape

    # Pad if needed
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        x_padded = F.pad(x, (0, pad_size))
    else:
        pad_size = 0
        x_padded = x

    x_flat = x_padded.reshape(-1, group_size)  # [num_groups, group_size]
    num_groups = x_flat.shape[0]
    K = max(1, int(num_groups * fp8_ratio))

    # 1. per-group max-abs importance
    with torch.no_grad():
        absmax = x_flat.float().abs().amax(dim=-1)  # [num_groups]
        _, topk_idx = absmax.topk(K)
        fp8_mask = torch.zeros(num_groups, dtype=torch.bool, device=x.device)
        fp8_mask.scatter_(0, topk_idx, True)

    fp8_idx = fp8_mask.nonzero(as_tuple=False).squeeze(1)
    fp4_idx = (~fp8_mask).nonzero(as_tuple=False).squeeze(1)

    x_fp8 = x_flat[fp8_idx]    # [N_fp8, group_size]
    x_fp4 = x_flat[fp4_idx]    # [N_fp4, group_size]
    am_fp8 = absmax[fp8_idx]
    am_fp4 = absmax[fp4_idx]

    # 2. physical packing
    if _CUDA_READY and x.is_cuda:
        fp8_packed = _BILEVEL_EXT.fp8_pack(x_fp8.contiguous(), am_fp8.contiguous())
        fp4_packed = _BILEVEL_EXT.fp4_pack(x_fp4.contiguous(), am_fp4.contiguous())
    else:
        fp8_packed = _pytorch_fp8_pack(x_fp8, am_fp8)
        fp4_packed = _pytorch_fp4_pack(x_fp4, am_fp4)

    return BilevelPacked(
        fp8_packed  = fp8_packed,
        fp4_packed  = fp4_packed,
        fp8_absmax  = am_fp8,
        fp4_absmax  = am_fp4,
        fp8_indices = fp8_idx,
        fp4_indices = fp4_idx,
        orig_shape  = (B, L, D),
        group_size  = group_size,
    )


def bilevel_unpack(packed: BilevelPacked) -> torch.Tensor:
    """Restore a BilevelPacked container to a BF16 tensor of the original
    (B, L, D) shape. The values carry the quantization error.
    """
    B, L, D = packed.orig_shape
    gs = packed.group_size
    device = packed.fp8_packed.device

    D_padded = D if D % gs == 0 else D + (gs - D % gs)
    num_groups = (B * L * D_padded) // gs

    # FP8 / FP4 unpack
    if _CUDA_READY and packed.fp8_packed.is_cuda:
        x_fp8 = _BILEVEL_EXT.fp8_unpack(packed.fp8_packed, packed.fp8_absmax, gs)
        x_fp4 = _BILEVEL_EXT.fp4_unpack(packed.fp4_packed, packed.fp4_absmax, gs)
    else:
        x_fp8 = _pytorch_fp8_unpack(packed.fp8_packed, packed.fp8_absmax)
        x_fp4 = _pytorch_fp4_unpack(packed.fp4_packed, packed.fp4_absmax, gs)

    # scatter back to the original group positions
    out_flat = torch.empty(num_groups, gs, dtype=torch.bfloat16, device=device)
    out_flat[packed.fp8_indices] = x_fp8
    out_flat[packed.fp4_indices] = x_fp4

    result = out_flat.reshape(B, L, D_padded)
    if D_padded != D:
        result = result[..., :D]
    return result


# PyTorch fallback packing: same values, no memory saving

def _pytorch_fp8_pack(x: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    """Fallback for the FP8 pack kernel: per-group BF16 to a signed 8-bit code
    stored in uint8. Same values, no size reduction.
    """
    N, D  = x.shape
    scale = (absmax / 127.0 + 1e-8).unsqueeze(1)   # [N, 1]
    q     = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q.view(torch.uint8)   # bit-cast to uint8


def _pytorch_fp4_pack(x: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    """Fallback for the 4-bit pack kernel: two 4-bit codes per byte, computed in
    PyTorch. Same values as the kernel.
    """
    N, D  = x.shape
    assert D % 2 == 0
    scale = (absmax / 7.0 + 1e-8).unsqueeze(1)   # [N, 1]

    q       = (x.float() / scale).round().clamp(-7, 7).to(torch.int32)
    q_off   = q + 8                               # [1, 15]
    q_lo    = q_off[:, 0::2] & 0xF               # N, D/2
    q_hi    = (q_off[:, 1::2] & 0xF) << 4
    packed  = (q_lo | q_hi).to(torch.uint8)
    return packed                                  # [N, D/2]


def _pytorch_fp8_unpack(packed: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    N    = packed.shape[0]
    q    = packed.view(torch.int8).float()        # uint8 bit-cast to int8, then float
    scale= (absmax / 127.0 + 1e-8).unsqueeze(1)
    return (q * scale).to(torch.bfloat16)


def _pytorch_fp4_unpack(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    D: int,
) -> torch.Tensor:
    N     = packed.shape[0]
    scale = (absmax / 7.0 + 1e-8).unsqueeze(1)   # [N, 1]

    lo = (packed & 0x0F).float() - 8.0            # [N, D/2]
    hi = ((packed >> 4) & 0x0F).float() - 8.0     # [N, D/2]

    # interleave lo, hi back into [N, D]
    out = torch.stack([lo, hi], dim=2).view(N, D)
    return (out * scale).to(torch.bfloat16)
