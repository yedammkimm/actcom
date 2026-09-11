# Copyright (c) 2025 OAMP Research Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
oamp.bilevel
============
Bi-Level Mixed-Precision Quantizer (model-agnostic, pure tensor operations).

Per-group Algorithm  (cf. Paper §3.2)
-------------------
  Input hidden_states  (..., D)
      |
      v
  Reshape into groups of 128 (pad if needed)
      |
      v
  compute max-abs per group                                   ... importance
      |
      v
  Top K% groups → fp8_quantize(per-group)  (FP8, 8-bit anchors)
  Rest          → fp4_quantize(per-group)  (FP4, 4-bit body)
      |
      v
  torch.where(mask, fp8_result, fp4_result)                   ... compose
      |
      v
  Output (..., D)  — STE-differentiable

Public API
----------
group_bilevel_quantize(hidden_states, fp8_ratio, group_size)  : per-group
BilevelQuantizer  : nn.Module, configurable quantizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import fp4_quantize, fp8_quantize, GROUP_SIZE
from .selector import select_anchor_groups

__all__ = [
    "BilevelQuantizer",
    "group_bilevel_quantize",
]


# ──────────────────────────────────────────────────────────────────────────────
# Per-group bi-level quantization 
# ──────────────────────────────────────────────────────────────────────────────

def group_bilevel_quantize(
    hidden_states: torch.Tensor,
    fp8_ratio: float = 0.20,
    group_size: int = GROUP_SIZE,
    routing: str = 'maxabs',
    generator: torch.Generator | None = None,
    fixed_mask: torch.Tensor | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Group-level bi-level mixed-precision quantization.

    Routes groups of ``group_size`` elements to FP8 or FP4.

    Parameters
    ----------
    hidden_states : Tensor  (..., D)
    fp8_ratio     : float   Fraction of groups quantized as FP8 anchors (default 0.20).
    group_size    : int     Group size (default 128).
    routing       : str     Group routing criterion: 'maxabs' (default) or 'random'.
    generator     : torch.Generator, optional
                    Dedicated RNG for 'random' routing. REQUIRED to avoid
                    perturbing the global torch RNG stream (which would
                    desynchronize e.g. LoRA dropout across methods).
    fixed_mask    : BoolTensor (num_hidden_groups,), optional
                    Precomputed per-hidden-dim group mask ('random_fixed'
                    routing). Tiled across token positions; overrides
                    ``routing``. num_hidden_groups = ceil(D / group_size).
    stats         : dict, optional
                    If given, 'fp8_groups' / 'total_groups' entries are
                    incremented in-place (tensor-friendly ints) for
                    bits/element verification logging.

    Returns
    -------
    quantized : Tensor  (..., D)  — STE-differentiable.
    """
    orig_shape = hidden_states.shape
    D = hidden_states.shape[-1]

    # Pad hidden dim if not divisible by group_size
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        hidden_states_padded = F.pad(hidden_states, (0, pad_size))
    else:
        pad_size = 0
        hidden_states_padded = hidden_states

    x_flat = hidden_states_padded.reshape(-1, group_size)
    num_groups = x_flat.shape[0]
    K = max(1, int(num_groups * fp8_ratio))

    if K >= num_groups:
        return fp8_quantize(hidden_states, group_size)

    # Step 1: Determine FP8 anchor mask (no gradient)
    with torch.no_grad():
        if fixed_mask is not None:
            # 'random_fixed': per-hidden-dim mask, tiled over token positions.
            groups_per_row = (D + group_size - 1) // group_size
            reps = num_groups // groups_per_row
            fp8_mask = fixed_mask.to(hidden_states.device).repeat(reps)
        else:
            if routing == 'random':
                # Use a dedicated generator so the global torch RNG stream
                # is untouched (keeps dropout sequences comparable).
                importance = torch.rand(num_groups, device=hidden_states.device,
                                        generator=generator)
            else:  # 'maxabs' (default)
                importance = x_flat.float().abs().amax(dim=-1)
            _, topk_idx = importance.topk(K)
            fp8_mask = torch.zeros(num_groups, dtype=torch.bool, device=hidden_states.device)
            fp8_mask.scatter_(0, topk_idx, True)

        if stats is not None:
            # Tensor accumulation (no .item() → no CPU sync per call)
            s = fp8_mask.sum()
            stats['fp8_groups'] = stats.get('fp8_groups', 0) + s
            stats['total_groups'] = stats.get('total_groups', 0) + num_groups

    # Step 2: Quantize all at both precisions (STE)
    q_fp8 = fp8_quantize(hidden_states_padded, group_size)
    q_fp4 = fp4_quantize(hidden_states_padded, group_size)

    # Step 3: Compose via group mask
    mask_expanded = fp8_mask.unsqueeze(-1).expand_as(x_flat)
    result_flat = torch.where(mask_expanded,
                              q_fp8.reshape(-1, group_size),
                              q_fp4.reshape(-1, group_size))

    result = result_flat.reshape(hidden_states_padded.shape)
    if pad_size > 0:
        result = result[..., :D]

    return result.to(hidden_states.dtype)


class BilevelQuantizer(nn.Module):
    """
    Bi-Level Mixed-Precision Quantizer.

    Inserts before any transformer layer to apply OAMP activation compression.

    Parameters
    ----------
    fp8_ratio : float
        Fraction of groups preserved as FP8 anchors (default 0.20 = 20%).
    group_size : int
        Group size for per-group quantization (default 128).
    enabled : bool
        If False, acts as identity (pass-through).

    Example
    -------
    >>> quantizer = BilevelQuantizer(fp8_ratio=0.20)
    >>> hidden = quantizer(hidden_states)   # apply before each layer

    Hook-based attachment:
    >>> hook = quantizer.register_pre_hook(layer)
    """

    def __init__(
        self,
        fp8_ratio: float = 0.20,
        group_size: int = GROUP_SIZE,
        enabled: bool = True,
    ):
        super().__init__()
        self.fp8_ratio = fp8_ratio
        self.group_size = group_size
        self.enabled = enabled
        self._stats = {"calls": 0, "fp8_groups": 0, "total_groups": 0}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden_states

        out = group_bilevel_quantize(
            hidden_states,
            fp8_ratio=self.fp8_ratio,
            group_size=self.group_size,
        )

        # Update statistics
        D = hidden_states.shape[-1]
        total_elements = hidden_states.numel()
        num_groups = total_elements // self.group_size
        K = max(1, int(num_groups * self.fp8_ratio))
        self._stats["calls"] += 1
        self._stats["fp8_groups"] += K
        self._stats["total_groups"] += num_groups

        return out

    def register_pre_hook(self, layer: nn.Module):
        """Register a forward pre-hook on ``layer`` that applies quantization."""
        quantizer = self

        def _hook(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                h = args[0]
                if isinstance(h, torch.Tensor) and h.dim() >= 2:
                    return (quantizer(h),) + args[1:]
            return args

        return layer.register_forward_pre_hook(_hook)

    def reset_stats(self):
        self._stats = {"calls": 0, "fp8_groups": 0, "total_groups": 0}

    @property
    def fp8_group_ratio(self) -> float:
        t = self._stats["total_groups"]
        return self._stats["fp8_groups"] / t if t > 0 else 0.0

    @property
    def bits_per_element(self) -> float:
        return self.fp8_ratio * 8 + (1 - self.fp8_ratio) * 4

    def extra_repr(self) -> str:
        return (
            f"fp8_ratio={self.fp8_ratio}, group_size={self.group_size}, "
            f"enabled={self.enabled}, "
            f"calls={self._stats['calls']}"
        )
