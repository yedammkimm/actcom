"""Helpers that keep scaled-dot-product attention on the flash backend.

An all-ones attention mask is enough to push Transformers onto the MATH
backend, which materialises the (B, H, L, L) attention weights; these
helpers make sure that does not happen silently.
"""

import contextlib

import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


def flash_sdp_context(force: bool = True):
    """Return a context manager that restricts SDPA to the flash backend.

    Strict: there is no MATH fallback, so if flash is not eligible (a non-null
    attention mask, for instance) scaled_dot_product_attention raises a
    RuntimeError, which is the failure mode audit runs want. With force=False
    it returns a nullcontext for paths that need default dispatch; log the
    selected backend separately in that case.
    """
    if not force or sdpa_kernel is None:
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


def probe_backend_availability(seq_len: int = 64, dtype=torch.bfloat16):
    """Return which SDPA backend this device can use on a synthetic causal call
    without a mask: 'FLASH', 'EFFICIENT', 'CUDNN', 'MATH' or 'UNKNOWN'.

    This measures availability, not what a real model forward selects. For
    that, force flash with flash_sdp_context(force=True) and let the call fail
    if the model's real inputs make flash ineligible.
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
    """Return None when the mask is all ones, otherwise the mask unchanged.

    An all-ones mask still forces Transformers onto the MATH backend; returning
    None lets it set is_causal=True and select flash. With padding the mask is
    not all ones and must be kept. Every caller here uses batch size 1 without
    padding, so that is fine, but a batched pipeline would need a padding-aware
    mask of its own.
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
