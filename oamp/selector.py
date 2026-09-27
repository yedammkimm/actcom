# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.

"""Group importance scores and anchor selection for the max-abs anchor variant.

Each group of `group_size` elements along the last axis is scored by its
largest absolute value, and the top `ratio` of groups are marked as FP8
anchors. Used by oamp.bilevel; the pack hooks carry their own copy of the
rule.
"""

import torch
import torch.nn.functional as F

from .quantize import GROUP_SIZE

__all__ = [
    "compute_group_importance",
    "select_anchor_groups",
]


# Per-group importance and selection

def compute_group_importance(
    hidden_states: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Return the max-abs of every group of `hidden_states` (..., D) as a float32
    tensor of shape (num_groups,), detached from the graph. A partial trailing
    group is zero-padded before scoring.
    """
    D = hidden_states.shape[-1]
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        hidden_states = F.pad(hidden_states, (0, pad_size))

    x_flat = hidden_states.reshape(-1, group_size)
    with torch.no_grad():
        importance = x_flat.float().abs().amax(dim=-1)
    return importance


def select_anchor_groups(
    hidden_states: torch.Tensor,
    ratio: float = 0.20,
    group_size: int = GROUP_SIZE,
) -> torch.BoolTensor:
    """Return a boolean mask of shape (num_groups,) that is True for the top
    `ratio` fraction of groups by max-abs, the FP8 anchors, and False for the
    4-bit body groups.
    """
    D = hidden_states.shape[-1]
    if D % group_size != 0:
        pad_size = group_size - (D % group_size)
        hidden_states = F.pad(hidden_states, (0, pad_size))

    x_flat = hidden_states.reshape(-1, group_size)
    num_groups = x_flat.shape[0]
    K = max(1, int(num_groups * ratio))

    if K >= num_groups:
        return torch.ones(num_groups, dtype=torch.bool, device=hidden_states.device)

    with torch.no_grad():
        importance = x_flat.float().abs().amax(dim=-1)
        _, topk_idx = importance.topk(K)
        mask = torch.zeros(num_groups, dtype=torch.bool, device=hidden_states.device)
        mask.scatter_(0, topk_idx, True)

    return mask
