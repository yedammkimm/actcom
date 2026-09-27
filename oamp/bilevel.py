# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.

"""Max-abs anchor routing applied in the forward pass, with a straight-through gradient.

The hidden states are cut into groups of 128 elements along the last axis.
Each group is scored by its largest absolute value, the top `fp8_ratio` of
groups are quantized to FP8 and the rest to the symmetric 4-bit grid, and the
two results are composed with torch.where. The forward pass sees the
quantized values; the backward pass passes gradients through unchanged.

This is a standalone simulation of the anchor variant. The training runs in
the paper compress only the saved tensors, through oamp.pack_hooks, and do
not import this module.
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


# Per-group anchor routing

def group_bilevel_quantize(
    hidden_states: torch.Tensor,
    fp8_ratio: float = 0.20,
    group_size: int = GROUP_SIZE,
    routing: str = 'maxabs',
    generator: torch.Generator | None = None,
    fixed_mask: torch.Tensor | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """Quantize `hidden_states` (..., D) group by group and return a tensor of the
    same shape and dtype, with the straight-through gradient.

    `fp8_ratio` is the fraction of groups that become FP8 anchors and `group_size`
    the number of elements per group; a partial trailing group is zero-padded.
    `routing` picks the anchors by max-abs ('maxabs') or at random ('random').
    Random routing needs `generator`, a dedicated torch.Generator, so that the
    global RNG stream, which also drives LoRA dropout, is left untouched.
    `fixed_mask` is a precomputed boolean mask over the ceil(D / group_size)
    hidden-dimension groups; it is tiled over token positions and overrides
    `routing`. If `stats` is given, its 'fp8_groups' and 'total_groups' entries
    are incremented in place (as tensors, to avoid a host sync) for the
    bits-per-element bookkeeping.
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

    # 1. anchor mask, no gradient
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
            # accumulate as tensors; .item() would force a CPU sync on every call
            s = fp8_mask.sum()
            stats['fp8_groups'] = stats.get('fp8_groups', 0) + s
            stats['total_groups'] = stats.get('total_groups', 0) + num_groups

    # 2. quantize everything at both precisions (straight-through)
    q_fp8 = fp8_quantize(hidden_states_padded, group_size)
    q_fp4 = fp4_quantize(hidden_states_padded, group_size)

    # 3. compose with the group mask
    mask_expanded = fp8_mask.unsqueeze(-1).expand_as(x_flat)
    result_flat = torch.where(mask_expanded,
                              q_fp8.reshape(-1, group_size),
                              q_fp4.reshape(-1, group_size))

    result = result_flat.reshape(hidden_states_padded.shape)
    if pad_size > 0:
        result = result[..., :D]

    return result.to(hidden_states.dtype)


class BilevelQuantizer(nn.Module):
    """nn.Module wrapper around group_bilevel_quantize with max-abs routing.

    `fp8_ratio` and `group_size` are as in the function; with `enabled=False` the
    module is an identity. It can be called on a hidden-state tensor directly or
    attached to a layer as a forward pre-hook with register_pre_hook. It counts
    calls and anchor groups, which fp8_group_ratio and reset_stats expose.
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
        """Attach the quantizer to `layer` as a forward pre-hook on its first tensor
        argument and return the hook handle.
        """
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
