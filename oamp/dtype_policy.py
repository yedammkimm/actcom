"""The bf16 policy applied to a PEFT model before training.

prepare_model_for_kbit_training upcasts every non-quantized tensor (norms,
embeddings, lm_head) to fp32, and PEFT initialises the LoRA A and B matrices
in fp32 as well. LlamaRMSNorm additionally casts its input to fp32 to
compute the variance and keeps that fp32 copy for backward; on the 3B model
that is (B, L, 3072) x 4 tensors per layer x 28 layers, about 5.6 GB at
B=4, L=4096. apply_dtype_policy reverses all of this: it casts norms,
embeddings, lm_head and the LoRA parameters to bf16 and swaps every *RMSNorm
module for BF16RMSNorm, which skips the upcast.

The four-way check that fixed the policy (100 identical steps, 3B, NF4
base, 2026-08-12):

    fp32 everywhere              final loss 0.870496   peak 6.167 GB
    bf16 norms, embed, lm_head   0.872880              5.854
    plus bf16 LoRA               0.871820              5.406
    plus BF16RMSNorm             0.871803              5.283

No NaN or Inf in any arm; the last two differ by 1.6e-5.

The order matters and the function enforces it: load the base model, run
prepare_model_for_kbit_training for NF4, call get_peft_model, then (4) cast,
(5) swap the norms and (6) collect the parameter storage pointers. Steps 4
to 6 are this file. The pointers are collected last because casting
reallocates storage, and apply_dtype_policy returns them together with the
model so a caller cannot get the order wrong.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .pack_hooks import collect_param_ptrs


class BF16RMSNorm(nn.Module):
    """RMSNorm that computes the variance in the input dtype instead of upcasting
    to fp32, as CompAct does.
    """

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
    """Replace every child module whose class name contains 'RMSNorm' with a
    BF16RMSNorm that reuses its weight. The substring match also catches the
    Qwen and Mistral variants. Raises if nothing was swapped, since that means
    the model's norm class was not recognised.
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
    """Cast norms, embeddings, lm_head and the LoRA parameters to bf16 in place.

    prepare_model_for_kbit_training upcasts the first group to fp32 and PEFT
    initialises the LoRA A and B matrices in fp32; this reverses both. The two
    halves are gated separately because they are separate effects and the dtype
    decomposition attributes memory to one or the other: `modules` covers
    norms, embeddings and lm_head, `lora` covers PEFT's autocast_adapter_dtype
    initialisation. Turning `lora` off alone reproduces the BF16-base
    configuration, where prepare_model_for_kbit_training never ran and only the
    adapters were left in fp32.
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
    """Report the dtype of each parameter category the policy manages.

    Uses named_modules for embed and lm_head so that tied-embedding models such
    as Llama-3.2, where lm_head.weight is not a separate parameter, still get an
    entry. Returns lists so a mixed-dtype category shows up in the JSON. After
    the policy every list should read ['torch.bfloat16'].
    """
    categories = {'norms': set(), 'embed': set(), 'lm_head': set(), 'lora': set()}
    # Params (LoRA + norms picked up via .weight below anyway)
    for name, param in model.named_parameters():
        n = name.lower()
        if 'lora_' in n:
            categories['lora'].add(str(param.dtype))
    # Modules: cover the tied lm_head, whose weight is shared with embed.
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
    """Return True when lm_head shares its weight with the input embedding, as
    Llama-3.2 does by default.
    """
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
    """Apply the policy in place and return (model, param_ptrs, dtype_report).

    `model` is a PEFT-wrapped model after get_peft_model; for NF4 runs,
    prepare_model_for_kbit_training must already have run on the base.
    `weight_quant` ('bf16' or 'nf4') is only recorded in the report; the
    RMSNorm swap applies to BF16-base runs as well. The three flags exist for
    ablation: `bf16_rmsnorm` swaps the norms, `bf16_params` casts norms,
    embeddings and lm_head, and `bf16_lora` casts the LoRA A and B matrices.
    Leaving `bf16_lora` off makes every adapter site keep an fp32 copy of its
    input activation, which is the effect the dtype decomposition in the paper
    isolates.

    `param_ptrs` is the set of storage pointers of every parameter, collected
    after the casts; pass it to PackHooks so parameters are never packed.
    `dtype_report` records weight_quant, the flags, the dtype of each parameter
    category and the swap counts, and belongs in the result JSON.
    """
    if weight_quant not in ('bf16', 'nf4'):
        raise ValueError(f"weight_quant must be 'bf16' or 'nf4', got {weight_quant!r}")

    tied_before = _is_lm_head_tied(model)

    # 4. cast
    _cast_to_bf16(model, modules=bf16_params, lora=bf16_lora)

    # If embed<->lm_head was tied and cast happened to break it (order-dependent
    # inside _cast_to_bf16), restore the tie so PEFT/HF don't train two copies.
    if tied_before and not _is_lm_head_tied(model):
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'tie_weights'):
            model.base_model.tie_weights()

    # 5. RMSNorm swap, for bf16 and nf4 bases alike
    if bf16_rmsnorm:
        n_swapped, seen = _swap_rmsnorm_for_bf16(model)
    else:
        n_swapped, seen = 0, set()

    # 6. parameter pointers, after the casts so the new storage is what gets recorded
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
