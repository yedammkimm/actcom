# Copyright (c) 2025 OAMP Research Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
oamp.memory
===========
Theoretical activation memory savings calculator for bi-level quantization.

Formula  (cf. Paper §3.3)
-------
  BF16 baseline  : 2 bytes / element
  Bi-level       : fp8_ratio * 1 byte + (1 - fp8_ratio) * 0.5 byte
  Savings (%)    : 1 - bilevel / bf16

Example (15/85 default)
-----------------------
  15% * 1 + 85% * 0.5 = 0.575 byte  →  71.25% savings vs BF16 (2 bytes)

Public API
----------
compute_theoretical_savings(fp8_ratio, ...) → dict
print_memory_analysis(fp8_ratio)
"""

from __future__ import annotations

__all__ = [
    "compute_theoretical_savings",
    "print_memory_analysis",
]

# BF16 = 2 bytes, FP8 = 1 byte, FP4 = 0.5 byte
_BF16_BYTES: float = 2.0
_FP8_BYTES:  float = 1.0
_FP4_BYTES:  float = 0.5


def compute_theoretical_savings(
    fp8_ratio: float = 0.20,
    seq_len:   int   = 512,
    hidden_dim: int  = 3072,
    num_layers: int  = 28,
    batch_size: int  = 1,
) -> dict:
    """
    Compute theoretical activation memory savings under bi-level quantization.

    Parameters
    ----------
    fp8_ratio  : float  FP8 anchor ratio (default 0.15).
    seq_len    : int    Sequence length (default 512).
    hidden_dim : int    Hidden dimension (LLaMA-3.2-3B = 3072).
    num_layers : int    Number of transformer layers (LLaMA-3.2-3B = 28).
    batch_size : int    Batch size.

    Returns
    -------
    dict with keys:
        fp8_ratio              : float
        fp4_ratio              : float
        bits_per_element       : float  Average bits per element
        bytes_per_element      : float  Average bytes per element
        savings_pct            : float  Savings vs BF16 (%)
        bf16_total_mb          : float  BF16 total activation size (MB)
        bilevel_total_mb       : float  Bi-level total activation size (MB)
        saved_mb               : float  Memory saved (MB)
    """
    fp4_ratio = 1.0 - fp8_ratio

    elems_per_layer = batch_size * seq_len * hidden_dim

    bf16_bytes    = elems_per_layer * _BF16_BYTES
    bilevel_bytes = elems_per_layer * (fp8_ratio * _FP8_BYTES + fp4_ratio * _FP4_BYTES)

    savings_pct       = (1.0 - bilevel_bytes / bf16_bytes) * 100.0
    bits_per_element  = fp8_ratio * 8.0 + fp4_ratio * 4.0
    bytes_per_element = fp8_ratio * _FP8_BYTES + fp4_ratio * _FP4_BYTES

    bf16_total_mb    = bf16_bytes    * num_layers / (1024 ** 2)
    bilevel_total_mb = bilevel_bytes * num_layers / (1024 ** 2)

    return {
        "fp8_ratio":          fp8_ratio,
        "fp4_ratio":          fp4_ratio,
        "bits_per_element":   bits_per_element,
        "bytes_per_element":  bytes_per_element,
        "savings_pct":        savings_pct,
        "bf16_total_mb":      bf16_total_mb,
        "bilevel_total_mb":   bilevel_total_mb,
        "saved_mb":           bf16_total_mb - bilevel_total_mb,
    }


def print_memory_analysis(
    fp8_ratio: float = 0.20,
    seq_len:   int   = 512,
    hidden_dim: int  = 3072,
    num_layers: int  = 28,
    batch_size: int  = 1,
) -> None:
    """
    Pretty-print the activation memory savings analysis.

    Example
    -------
    >>> from oamp import print_memory_analysis
    >>> print_memory_analysis(fp8_ratio=0.15)
    """
    s = compute_theoretical_savings(
        fp8_ratio=fp8_ratio,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=batch_size,
    )

    print("=" * 55)
    print(f"  OAMP Bi-Level Memory Analysis")
    print("=" * 55)
    print(f"  FP8 anchor : {s['fp8_ratio']*100:.0f}%  ({s['fp8_ratio']*100:.0f}% x 1 byte)")
    print(f"  FP4 body   : {s['fp4_ratio']*100:.0f}%  ({s['fp4_ratio']*100:.0f}% x 0.5 byte)")
    print(f"  Avg bits   : {s['bits_per_element']:.2f} bits/element")
    print(f"  Avg bytes  : {s['bytes_per_element']:.4f} bytes/element")
    print("-" * 55)
    print(f"  BF16  baseline    : {s['bf16_total_mb']:.1f} MB  ({num_layers} layers)")
    print(f"  Bi-Level quantized: {s['bilevel_total_mb']:.1f} MB")
    print(f"  Saved             : {s['saved_mb']:.1f} MB")
    print(f"  Savings           : {s['savings_pct']:.1f}%  vs BF16")
    print("=" * 55)
