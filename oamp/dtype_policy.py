"""Arm 4 dtype policy: bf16 norms / embed / lm_head / LoRA + BF16RMSNorm swap.

Spec v1 §3. Applied *after* ``get_peft_model`` and *before* param_ptrs
collection (INVARIANT-7).

Rationale (from 2026-08 audit):

  - ``prepare_model_for_kbit_training`` upcasts every non-Params4bit tensor
    (norms, embed, lm_head) to fp32. LoRA A/B are also initialized in fp32
    inside PEFT even when the base is bf16/NF4.
  - ``LlamaRMSNorm.forward`` casts ``hidden_states`` to fp32 to compute
    variance and saves the fp32 intermediate for backward. On 3B this is
    ``(B, L, 3072)`` * 4 tensors/layer * 28 layers ≈ 5.6 GB at B=4/L=4096
    (verified against the Table 11 fp32/bf16 sweep, 2026-08-13).

Verified in the 4-way GSM8K experiment (2026-08-12):

    Arm1 fp32              final_loss 0.870496   peak 6.167 GB
    Arm2 bf16 norms        0.872880              5.854
    Arm3 + LoRA bf16       0.871820              5.406
    Arm4 + BF16RMSNorm     0.871803              5.283   ← canonical
    NaN/Inf = 0 in every arm; Arm3->Arm4 final Δ = 1.6e-5.

INVARIANT-7 (spec §3.3): the order below cannot be re-ordered.

  1. base model load
  2. (NF4 only) prepare_model_for_kbit_training(use_gradient_checkpointing=False)
  3. get_peft_model                       — LoRA initialized here (fp32 by PEFT)
  4. cast norms / embed / lm_head / LoRA → bf16   ← this file
  5. RMSNorm swap                                 ← this file
  6. collect param_ptrs                           ← this file

Steps 4–6 are performed by :func:`apply_dtype_policy`. It returns
``(model, param_ptrs, dtype_report)`` so callers cannot get the order wrong.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .pack_hooks import collect_param_ptrs


class BF16RMSNorm(nn.Module):
    """RMSNorm without the fp32 upcast (CompAct-style)."""

    def __init__(self, weight: nn.Parameter, eps: float):
        super().__init__()
        # Reuse the original Parameter so identity is preserved (optimizer state,
        # PEFT freezing state, and gradient hooks all still point at the same object).
        self.weight = weight
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def _swap_rmsnorm_for_bf16(model) -> Tuple[int, set]:
    """Replace every ``*RMSNorm`` child module with :class:`BF16RMSNorm`.

    Uses substring match on the class name so Qwen / Mistral RMSNorm variants
    are also caught. Raises if zero swaps happen (spec §3.2).
    """
    swapped = 0
    seen_classes = set()
    for _, parent in model.named_modules():
        for attr, child in list(parent.named_children()):
            cls_name = type(child).__name__
            if 'RMSNorm' in cls_name and cls_name != 'BF16RMSNorm':
                seen_classes.add(cls_name)
                eps = getattr(child, 'variance_epsilon',
                              getattr(child, 'eps', 1e-6))
                new = BF16RMSNorm(child.weight, eps).to(child.weight.device)
                setattr(parent, attr, new)
                swapped += 1
    if swapped == 0:
        raise RuntimeError(
            "apply_dtype_policy: RMSNorm swap found 0 modules. "
            "The model has no *RMSNorm* class in its module tree. "
            "Check the model architecture or extend the class-name filter.")
    return swapped, seen_classes


def _cast_to_bf16(model, *, modules: bool = True, lora: bool = True) -> None:
    """Cast norms / embed / lm_head / LoRA parameters to bf16 in place.

    ``prepare_model_for_kbit_training`` upcasts these to fp32; PEFT
    initializes LoRA A/B in fp32. This reverses both.

    The two halves are gated separately because they are separate effects and
    the dtype decomposition needs to attribute memory to one or the other.
    ``modules`` covers norms / embeddings / lm_head, i.e. the
    ``prepare_model_for_kbit_training`` upcast; ``lora`` covers PEFT's
    ``autocast_adapter_dtype`` initialisation of A/B. Turning ``lora`` off
    alone reproduces the BF16-base configuration, where
    ``prepare_model_for_kbit_training`` never ran and only the adapters were
    left in fp32.
    """
    target = torch.bfloat16

    # Modules: norms + embeddings + LM head
    for name, module in (model.named_modules() if modules else ()):
        cls = type(module).__name__
        # Any *Norm layer (LayerNorm / RMSNorm / etc.) that carries a weight
        if 'Norm' in cls and getattr(module, 'weight', None) is not None:
            if module.weight.dtype != target:
                module.weight.data = module.weight.data.to(target)
            if getattr(module, 'bias', None) is not None and module.bias.dtype != target:
                module.bias.data = module.bias.data.to(target)
        elif isinstance(module, nn.Embedding):
            if module.weight.dtype != target:
                module.weight.data = module.weight.data.to(target)
        elif name.endswith('lm_head') and getattr(module, 'weight', None) is not None:
            if module.weight.dtype != target:
                module.weight.data = module.weight.data.to(target)

    # PEFT LoRA A/B parameters
    for name, param in (model.named_parameters() if lora else ()):
        if 'lora_' in name.lower() and param.dtype != target:
            param.data = param.data.to(target)


def _collect_dtype_report(model) -> dict:
    """Report the dtype of each policy-managed parameter category.

    Uses ``named_modules`` for embed / lm_head so tied-embedding models (e.g.
    Llama-3.2) where ``lm_head.weight`` is not a distinct parameter still get
    an entry. Returns lists so multi-dtype anomalies surface in the JSON.
    Expected after Arm 4: every list is ``['torch.bfloat16']``.
    """
    categories = {'norms': set(), 'embed': set(), 'lm_head': set(), 'lora': set()}
    # Params (LoRA + norms picked up via .weight below anyway)
    for name, param in model.named_parameters():
        n = name.lower()
        if 'lora_' in n:
            categories['lora'].add(str(param.dtype))
    # Modules — cover tied lm_head where its weight is shared with embed.
    for name, module in model.named_modules():
        cls = type(module).__name__
        w = getattr(module, 'weight', None)
        if w is None:
            continue
        if 'Norm' in cls:
            categories['norms'].add(str(w.dtype))
        if isinstance(module, nn.Embedding):
            categories['embed'].add(str(w.dtype))
        if name.endswith('lm_head'):
            categories['lm_head'].add(str(w.dtype))
    categories['lm_head_tied'] = _is_lm_head_tied(model)
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in categories.items()}


def _is_lm_head_tied(model) -> bool:
    """Detect tied embed<->lm_head weight (Llama-3.2 default)."""
    embed_w = lm_head_w = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and 'embed_tokens' in name:
            embed_w = getattr(module, 'weight', None)
        if name.endswith('lm_head') and getattr(module, 'weight', None) is not None:
            lm_head_w = module.weight
    if embed_w is None or lm_head_w is None:
        return False
    return embed_w.data_ptr() == lm_head_w.data_ptr()


def apply_dtype_policy(model, *,
                       weight_quant: str,
                       bf16_rmsnorm: bool = True,
                       bf16_params: bool = True,
                       bf16_lora: bool = True) -> Tuple[nn.Module, set, dict]:
    """Apply Arm 4 dtype policy in place and return (model, param_ptrs, dtype_report).

    Parameters
    ----------
    model : nn.Module
        A PEFT-wrapped model *after* ``get_peft_model`` (step 3 in INVARIANT-7).
        For NF4 runs, ``prepare_model_for_kbit_training`` must already have
        been called on the base before ``get_peft_model``.
    weight_quant : {'bf16', 'nf4'}
        Recorded on ``dtype_report['weight_quant']``. Does not gate the cast:
        BF16-base runs need the RMSNorm swap too (§3.4).
    bf16_rmsnorm : bool, default True
        Whether to swap ``*RMSNorm`` for :class:`BF16RMSNorm`. Off only for
        ablation of the RMSNorm fp32 upcast.
    bf16_params : bool, default True
        Whether to cast norms / embeddings / lm_head to bf16, reversing the
        ``prepare_model_for_kbit_training`` upcast. Off only for ablation.
    bf16_lora : bool, default True
        Whether to cast PEFT LoRA A/B to bf16, reversing ``autocast_adapter_dtype``.
        Off only for ablation. Leaving it off makes every adapter site save an
        fp32 copy of its input activation, which is the effect the dtype
        decomposition isolates.

    Returns
    -------
    model : nn.Module
        The same object, mutated in place.
    param_ptrs : set[int]
        Storage pointers for every parameter, collected *after* all casts
        (INVARIANT-7). Feed this to :class:`oamp.pack_hooks.PackHooks`.
    dtype_report : dict
        Contains ``weight_quant``, ``bf16_rmsnorm``, per-category dtype lists,
        and RMSNorm swap counts. Persist to the result JSON.
    """
    if weight_quant not in ('bf16', 'nf4'):
        raise ValueError(f"weight_quant must be 'bf16' or 'nf4', got {weight_quant!r}")

    tied_before = _is_lm_head_tied(model)

    # (4) cast
    _cast_to_bf16(model, modules=bf16_params, lora=bf16_lora)

    # If embed<->lm_head was tied and cast happened to break it (order-dependent
    # inside _cast_to_bf16), restore the tie so PEFT/HF don't train two copies.
    if tied_before and not _is_lm_head_tied(model):
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'tie_weights'):
            model.base_model.tie_weights()

    # (5) RMSNorm swap — applied for both bf16 and nf4 (§3.4)
    if bf16_rmsnorm:
        n_swapped, seen = _swap_rmsnorm_for_bf16(model)
    else:
        n_swapped, seen = 0, set()

    # (6) param_ptrs — after (4)+(5) so cast-induced new storage is picked up
    param_ptrs = collect_param_ptrs(model)

    dtype_report = {
        'weight_quant': weight_quant,
        'bf16_rmsnorm': bool(bf16_rmsnorm),
        'bf16_params': bool(bf16_params),
        'bf16_lora': bool(bf16_lora),
        'rmsnorm_swapped': n_swapped,
        'rmsnorm_classes_seen': sorted(seen),
        **_collect_dtype_report(model),
    }
    return model, param_ptrs, dtype_report
