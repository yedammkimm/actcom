"""Common utilities for γ-based OAMP experiments.

Ensures FlashAttention gets used and vocab-head tensors are excluded from packing.
"""

import contextlib

import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


def flash_sdp_context(force: bool = True):
    """Return a context manager that restricts SDPA to FlashAttention.

    Strict: no MATH fallback. If FLASH is ineligible (e.g. non-null attn_mask),
    the underlying scaled_dot_product_attention call raises RuntimeError. That
    is the intended failure mode for audit runs.

    Set force=False to get nullcontext for production paths that need default
    dispatch (but log the selected backend separately).
    """
    if not force or sdpa_kernel is None:
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


def probe_backend_availability(seq_len: int = 64, dtype=torch.bfloat16):
    """Return which SDPA backends this device *can* use on a synthetic causal, no-mask call.

    NOTE: this measures *availability*, not what a real model forward actually selects.
    For that, force FLASH via flash_sdp_context(force=True) and let the call error out
    if the model's real inputs make FLASH ineligible.

    Returns one of: 'FLASH', 'EFFICIENT', 'CUDNN', 'MATH', 'UNKNOWN'.
    """
    if sdpa_kernel is None:
        return 'UNKNOWN'
    import torch.nn.functional as F
    q = torch.randn(1, 8, seq_len, 64, device='cuda', dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for name, backend in [
        ('FLASH', SDPBackend.FLASH_ATTENTION),
        ('EFFICIENT', SDPBackend.EFFICIENT_ATTENTION),
        ('CUDNN', SDPBackend.CUDNN_ATTENTION),
    ]:
        try:
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            return name
        except Exception:
            continue
    return 'MATH'


def normalize_attention_mask(attention_mask):
    """Return None if the mask is all-ones (trivial), else return as-is.

    Passing an all-ones mask to Transformers still forces sdpa fallback to MATH.
    Returning None lets Transformers set `is_causal=True` and select FLASH.

    Caveat (batch>1 + padding): with padding tokens the mask is not all-ones and
    must be preserved. The current callers all use batch=1 no-padding so this is
    fine, but any future multi-batch pipeline must handle padding-aware masks
    separately (e.g. via a boolean mask compatible with FLASH's future variants,
    or by grouping by length).
    """
    if attention_mask is None:
        return None
    if attention_mask.numel() == 0:
        return None
    if attention_mask.dtype == torch.bool:
        if bool(attention_mask.all()):
            return None
        return attention_mask
    if bool((attention_mask == 1).all()):
        return None
    return attention_mask
