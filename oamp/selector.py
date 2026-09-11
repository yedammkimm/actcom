# Copyright (c) 2025 OAMP Research Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
oamp.selector
=============
Importance scoring and anchor selection for bi-level quantization.

Per-group max-abs importance scoring: each group of GROUP_SIZE elements
is scored by its maximum absolute value: $I_j = \max_{i \in g_j} |h_i|$.
The top-K% groups are selected as FP8 anchors.

Public API
----------
compute_group_importance(hidden_states, group_size)  → Tensor       (num_groups,)
select_anchor_groups(hidden_states, ratio, group_size) → BoolTensor (num_groups,)
"""

import torch
import torch.nn.functional as F

from .quantize import GROUP_SIZE

__all__ = [
    "compute_group_importance",
    "select_anchor_groups",
]


# ──────────────────────────────────────────────────────────────────────────────
# Per-group importance & selection (default, matches experiments)
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_importance(
    hidden_states: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """
    Compute per-group max-abs as importance score.

    Reshapes hidden_states into groups of ``group_size`` elements and
    computes the maximum absolute value within each group. Handles
    non-divisible hidden dimensions via zero-padding.

    Parameters
    ----------
    hidden_states : Tensor  (..., D)
    group_size    : int     Group size (default 128).

    Returns
    -------
    importance : Tensor  (num_groups,)  — float32, no gradient attached.
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
    """
    Select the top ``ratio`` fraction of groups as FP8 anchors.

    Parameters
    ----------
    hidden_states : Tensor  (..., D)
    ratio         : float   Fraction of groups designated as FP8 anchors (default 0.20).
    group_size    : int     Group size (default 128).

    Returns
    -------
    fp8_mask : BoolTensor  (num_groups,)
               True  → FP8 anchor group
               False → FP4 body group
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
