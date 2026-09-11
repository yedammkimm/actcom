"""Gradient-error probe: which tensors' compression corrupts gradients most?

Design:
  - Two checkpoints:
      A. step 0  = fresh Arm 4 model (LoRA B=0, γ shortcut baseline)
      B. step ~300 = pilot_sr_lr adapter (SR-trained, closest available
         to user-requested "step 500"; caveats logged)
  - Fixed batch: seed=0 randint, B=1 L=512 (matches audit conditions)
  - For each filter F ∈ {none, γ full, α_repro, MLP_only, attn_proj_only,
                         attn_4D_only, drop_4D}:
      * gate ctx.pack: compress tensor iff F(shape,dtype,role) is True
      * forward + backward, collect p.grad
      * report ||g||, ||g − g_true||, cos(g, g_true)
  - g_true (no compression) is computed ONCE per checkpoint and reused.
  - Additivity check: Σ_i (g_i − g_true) vs (g_γ_full − g_true).

γ + `maxabs` routing has NO stochastic element, so a single measurement per
filter is exact. If ever probing SR or random_mixed, the caller must pass
``sr_seed`` / ``mask_seed`` and (optionally) average across seeds.

Output: printed table + JSON at results/gradient_error_probe_<ts>.json.
"""

from __future__ import annotations

import os
import sys
import json
import time
import inspect
from collections import defaultdict
from typing import Callable, Dict, Optional

os.environ.setdefault("HF_HOME", "/app/hf_cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType,
                  prepare_model_for_kbit_training)

from oamp.pack_hooks import PackHooks, make_pack_hooks, collect_param_ptrs
from oamp.dtype_policy import apply_dtype_policy

# Model id can be overridden via env for cross-architecture probes
# (e.g. GRAD_PROBE_MODEL='Qwen/Qwen2.5-3B-Instruct' for the 2026-08-17 Qwen probe).
MODEL = os.environ.get('GRAD_PROBE_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')
CACHE = "/app/hf_cache"
DEV = torch.device("cuda:0")
B, L = 1, 512

# Trained adapter (non-SR γ, 300 steps, lr=2e-4, seed 42 — production match).
# Set to '' or nonexistent path to skip checkpoint B automatically.
ADAPTER = os.environ.get('GRAD_PROBE_ADAPTER', '')


# ----------------------------------------------------------------
# Arm 4 model loader
# ----------------------------------------------------------------

def load_arm4_fresh():
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir=CACHE, local_files_only=True,
        quantization_config=bnb).to(DEV)
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                      lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      bias='none')
    m = get_peft_model(m, lcfg)
    m, param_ptrs, _ = apply_dtype_policy(m, weight_quant='nf4', bf16_rmsnorm=True)
    m.train()
    return m, tok, param_ptrs


def load_arm4_trained(adapter_path: str):
    """Fresh Arm 4 + swap in a saved LoRA adapter state_dict."""
    m, tok, param_ptrs = load_arm4_fresh()
    if os.path.isdir(adapter_path):
        from peft.utils.save_and_load import set_peft_model_state_dict
        state = _read_adapter_state(adapter_path)
        info = set_peft_model_state_dict(m, state)
        print(f"[adapter] loaded {adapter_path}  info={info}", flush=True)
    else:
        print(f"[adapter] NOT FOUND: {adapter_path}  falling back to fresh", flush=True)
    return m, tok, param_ptrs


def _read_adapter_state(path):
    """PEFT saves adapter_model.safetensors or adapter_model.bin."""
    from safetensors.torch import load_file
    st = os.path.join(path, 'adapter_model.safetensors')
    if os.path.isfile(st):
        return load_file(st, device='cuda:0')
    bin_ = os.path.join(path, 'adapter_model.bin')
    if os.path.isfile(bin_):
        return torch.load(bin_, map_location='cuda:0')
    raise FileNotFoundError(f"no adapter file in {path}")


# ----------------------------------------------------------------
# Module attribution (audit script style)
# ----------------------------------------------------------------

_MODULE_BUCKETS = [
    ('q_proj', 'q_proj'), ('k_proj', 'k_proj'), ('v_proj', 'v_proj'),
    ('o_proj', 'o_proj'), ('gate_proj', 'gate_proj'),
    ('up_proj', 'up_proj'), ('down_proj', 'down_proj'),
    ('lm_head', 'lm_head'),
]


def _bucket_role(role: str) -> str:
    # residual is tagged at TWO points per decoder layer:
    #   - LlamaDecoderLayer.forward_pre → 'layers.N.residual_pre_attn'
    #     (== hidden_states entering the layer, before input_layernorm)
    #   - post_attention_layernorm.forward_pre → 'layers.N.residual_pre_mlp'
    #     (== hidden_states after residual_1 + attn_out, before MLP)
    # These are DIFFERENT tensors with different distributions and get
    # separate roles so [4] can compare Δcos between them.
    if role.endswith('.residual_pre_attn'):
        return 'residual_pre_attn'
    if role.endswith('.residual_pre_mlp'):
        return 'residual_pre_mlp'
    if role.endswith('.residual'):        # legacy tag (kept for backcompat)
        return 'residual_pre_attn'
    for needle, bucket in _MODULE_BUCKETS:
        if needle in role:
            io = 'in' if role.endswith('.in') else ('out' if role.endswith('.out') else '?')
            return f'{bucket}.{io}'
    return 'unknown'


def register_attribution(model, ptr_map: dict) -> list:
    """Linear.forward → tag input/output storage_ptrs.
    LlamaDecoderLayer.forward_pre  → 'residual_pre_attn' (layer entry).
    post_attention_layernorm.forward_pre → 'residual_pre_mlp' (MLP entry)."""
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            short = _short(name)

            def linhook(m, inputs, output, sn=short):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    ptr_map[inputs[0].untyped_storage().data_ptr()] = f'{sn}.in'
                if isinstance(output, torch.Tensor):
                    ptr_map[output.untyped_storage().data_ptr()] = f'{sn}.out'

            handles.append(mod.register_forward_hook(linhook))

        if 'LlamaDecoderLayer' in type(mod).__name__:
            short = _short(name)

            def declhook(m, inputs, sn=short):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    ptr_map[inputs[0].untyped_storage().data_ptr()] = f'{sn}.residual_pre_attn'

            handles.append(mod.register_forward_pre_hook(declhook))

        # Tag the tensor entering post_attention_layernorm as residual_pre_mlp.
        # In Llama this is the second residual add: residual_1 + attn_out.
        if 'post_attention_layernorm' in name and hasattr(mod, 'register_forward_pre_hook'):
            short = _short(name)

            def post_ln_hook(m, inputs, sn=short):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    ptr_map[inputs[0].untyped_storage().data_ptr()] = f'{sn}.residual_pre_mlp'

            handles.append(mod.register_forward_pre_hook(post_ln_hook))

    # Rotary + repeat_kv re-materialise Q/K/V head-view tensors with fresh
    # storage_ptrs, breaking the Linear-hook chain. Monkey-patch the two
    # helper functions so we can re-tag the output tensors while the probe
    # is active. Restore is exposed as a handle so the caller (main loop)
    # still cleans up in a for-loop over `handles`.
    _install_rope_repeatkv_tagging(model, ptr_map, handles)
    return handles


# How often the layer survived re-materialisation, versus how often the bare
# role had to be used. A depth-conditioned sweep is only readable if the second
# number is small and, more importantly, evenly spread over layers.
_tag_stats = defaultdict(int)


def _layer_of(role_raw: str):
    """Layer index from a tag such as 'layers.13.self_attn.q_proj.out'.

    Returns None when the tag never carried one, which is exactly the case a
    depth-conditioned filter must not silently treat as layer 0.
    """
    if role_raw.startswith('layers.'):
        head = role_raw.split('.', 2)
        if len(head) > 1 and head[1].isdigit():
            return int(head[1])
    return None


def _install_rope_repeatkv_tagging(model, ptr_map: dict, handles: list):
    """Wrap apply_rotary_pos_emb, repeat_kv, and scaled_dot_product_attention
    so Q/K/V head-view storage_ptrs get re-tagged after every re-materialisation
    the attention block performs. SDPA input is the final ground truth: whatever
    tensors reach ``F.scaled_dot_product_attention(q, k, v, ...)`` are, by
    definition, the Q/K/V that participate in attention."""
    try:
        from transformers.models.llama import modeling_llama as _ml
    except ImportError:
        _ml = None

    _orig_rope = getattr(_ml, 'apply_rotary_pos_emb', None) if _ml else None
    _orig_repeat_kv = getattr(_ml, 'repeat_kv', None) if _ml else None

    def _inherit(src, fallback):
        """Carry the layer-qualified tag forward instead of overwriting it.

        The Linear hook tags a projection output as
        'layers.N.self_attn.q_proj.out'. RoPE and SDPA used to replace that
        with a bare 'q_proj.out', which is all the role bucket needs but
        throws away the only thing a depth-conditioned filter can read. If the
        incoming tensor still carries its tag -- view and transpose preserve
        the storage pointer, contiguous() does not -- reuse it; otherwise fall
        back to the bare role, and count that as a miss.
        """
        if isinstance(src, torch.Tensor):
            got = ptr_map.get(src.untyped_storage().data_ptr())
            if got and got.endswith(fallback):
                _tag_stats['inherited'] += 1
                return got
        _tag_stats['fallback'] += 1
        return fallback

    if _orig_rope is not None:
        def tagged_rope(q, k, cos, sin, *a, **kw):
            rq = _inherit(q, 'q_proj.out')
            rk = _inherit(k, 'k_proj.out')
            q2, k2 = _orig_rope(q, k, cos, sin, *a, **kw)
            if isinstance(q2, torch.Tensor):
                ptr_map[q2.untyped_storage().data_ptr()] = rq
            if isinstance(k2, torch.Tensor):
                ptr_map[k2.untyped_storage().data_ptr()] = rk
            return q2, k2
        _ml.apply_rotary_pos_emb = tagged_rope

    if _orig_repeat_kv is not None:
        def tagged_repeat_kv(hidden_states, n_rep):
            out = _orig_repeat_kv(hidden_states, n_rep)
            if (isinstance(hidden_states, torch.Tensor)
                    and isinstance(out, torch.Tensor)):
                orig_role = ptr_map.get(
                    hidden_states.untyped_storage().data_ptr())
                if orig_role:
                    ptr_map[out.untyped_storage().data_ptr()] = orig_role
            return out
        _ml.repeat_kv = tagged_repeat_kv

    # SDPA input is the definitive Q/K/V location: whatever reaches
    # F.scaled_dot_product_attention(q, k, v, ...) IS the attention Q/K/V.
    # This covers cases where transformers 5.x skips repeat_kv (GQA-native SDPA)
    # or where a contiguous() call broke the earlier chain.
    _orig_sdpa = F.scaled_dot_product_attention

    def tagged_sdpa(q, k, v, *args, **kwargs):
        for t, role in ((q, 'q_proj.out'), (k, 'k_proj.out'), (v, 'v_proj.out')):
            if isinstance(t, torch.Tensor):
                ptr_map[t.untyped_storage().data_ptr()] = _inherit(t, role)
        return _orig_sdpa(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = tagged_sdpa

    class _PatchHandle:
        def remove(self):
            if _orig_rope is not None:
                _ml.apply_rotary_pos_emb = _orig_rope
            if _orig_repeat_kv is not None:
                _ml.repeat_kv = _orig_repeat_kv
            F.scaled_dot_product_attention = _orig_sdpa

    handles.append(_PatchHandle())


def _short(name: str) -> str:
    for prefix in ('base_model.model.model.', 'base_model.model.'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ----------------------------------------------------------------
# Q/K compression variants for experiment 2 (2026-08-20).
#
# 2 x 2 design: {blockwise, per-channel} x {FP4, INT4}. Only Q/K head-views
# are re-routed through these; everything else stays on the γ ctx original
# pack. Storage overhead for INT4 (int8 backing) is fine here since the
# probe only measures gradient corruption, not memory.
# ----------------------------------------------------------------

_INT4_MAX = 7   # signed 4-bit range [-8, 7]; asymmetric absmax scale uses 7
# Block size for every per-block scale in this file. It was a fixed constant
# until the 2026-09-08 sweep needed to vary it; the default is unchanged, so a
# run that does not set it is bit-identical to every earlier probe. Set with
# --group_size or GRAD_PROBE_GROUP_SIZE. _GACT_GROUP_SIZE below is a different
# quantity -- GACT's flat reshape uses 256 -- and is deliberately not tied to it.
_GROUP_SIZE = int(os.environ.get('GRAD_PROBE_GROUP_SIZE', 128))

# True E2M1 FP4 grid (torch.float4_e2m1fn_x2 codes, sans NaN/Inf, no signed zero).
# Positive levels: 8 values; negative levels: 7 (0 is unique). Max magnitude = 6.
_E2M1_LEVELS_POS = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MIDPOINTS  = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_MAX = 6.0


def _pack_int4_block(tensor: torch.Tensor):
    """INT4 with per-block scale, group_size=128, same axis as our FP4."""
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    x = tensor.reshape(-1)
    orig_numel = x.numel()
    pad = (_GROUP_SIZE - (orig_numel % _GROUP_SIZE)) % _GROUP_SIZE
    if pad > 0:
        x = F.pad(x, (0, pad))
    x = x.reshape(-1, _GROUP_SIZE)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).float()
    scale = absmax / _INT4_MAX                     # (num_groups, 1)
    q = (x.float() / scale).round().clamp(-8, _INT4_MAX).to(torch.int8)
    return ('int4_block', q, scale, orig_shape, orig_numel, orig_dtype)


def _unpack_int4_block(packed):
    _tag, q, scale, orig_shape, orig_numel, orig_dtype = packed
    x = q.float() * scale
    x = x.reshape(-1)[:orig_numel]
    return x.reshape(orig_shape).to(orig_dtype)


def _pack_int4_chan_4d(tensor: torch.Tensor):
    """HyC-LoRA condition: per-channel scale (reduce across BL, keep D-axis).

    Input: 4-D head-view (B, H, L, dh).
    Flatten to (B*L, H*dh) so channels = last dim, tokens = leading. Scale is
    computed over the token axis, one per channel.
    """
    assert tensor.dim() == 4, "_pack_int4_chan_4d expects (B, H, L, dh)"
    B, H, L, dh = tensor.shape
    orig_dtype = tensor.dtype
    x_ch = tensor.transpose(1, 2).contiguous().reshape(B * L, H * dh)   # (BL, D)
    absmax = x_ch.abs().amax(dim=0).clamp(min=1e-8).float()             # (D,)
    scale = absmax / _INT4_MAX                                          # (D,)
    q = (x_ch.float() / scale).round().clamp(-8, _INT4_MAX).to(torch.int8)
    return ('int4_chan', q, scale, (B, H, L, dh), orig_dtype)


def _unpack_int4_chan_4d(packed):
    _tag, q, scale, orig_shape, orig_dtype = packed
    B, H, L, dh = orig_shape
    x_ch = q.float() * scale                                            # (BL, D)
    x = x_ch.reshape(B, L, H, dh).transpose(1, 2).contiguous()          # (B, H, L, dh)
    return x.to(orig_dtype)


def _pack_fp4_chan_4d(tensor: torch.Tensor):
    """FP4 with per-channel scale (2×2 arm: format=FP4, granularity=channel).

    Uses the same signed 4-bit range [-7, 7] as our FP4 body encoding so the
    quant grid is identical to _pack_uniform(bits=4) but the scale is now
    per-channel instead of per-group.
    """
    assert tensor.dim() == 4, "_pack_fp4_chan_4d expects (B, H, L, dh)"
    B, H, L, dh = tensor.shape
    orig_dtype = tensor.dtype
    x_ch = tensor.transpose(1, 2).contiguous().reshape(B * L, H * dh)
    absmax = x_ch.abs().amax(dim=0).clamp(min=1e-8).float()             # (D,)
    scale = absmax / 7.0                                                # (D,) - _FP4_MAX
    q = (x_ch.float() / scale).round().clamp(-7, 7).to(torch.int8)
    return ('fp4_chan', q, scale, (B, H, L, dh), orig_dtype)


def _unpack_fp4_chan_4d(packed):
    _tag, q, scale, orig_shape, orig_dtype = packed
    B, H, L, dh = orig_shape
    x_ch = q.float() * scale
    x = x_ch.reshape(B, L, H, dh).transpose(1, 2).contiguous()
    return x.to(orig_dtype)


def _quantize_to_e2m1(y):
    """Nearest-level rounding onto the E2M1 grid. ``y`` is assumed clamped to
    [-6, 6]. Returns a tensor of the same shape/dtype whose values are exactly
    on the E2M1 grid."""
    device = y.device
    y_abs = y.abs()
    sign = torch.sign(y)
    midpoints = _E2M1_MIDPOINTS.to(device=device, dtype=y.dtype)
    levels = _E2M1_LEVELS_POS.to(device=device, dtype=y.dtype)
    idx = torch.bucketize(y_abs, midpoints)                # in [0..7]
    return sign * levels[idx]


def _pack_e2m1_block(tensor: torch.Tensor):
    """True FP4 (E2M1) with per-block absmax scale, group_size=128."""
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    x = tensor.reshape(-1)
    orig_numel = x.numel()
    pad = (_GROUP_SIZE - (orig_numel % _GROUP_SIZE)) % _GROUP_SIZE
    if pad > 0:
        x = F.pad(x, (0, pad))
    x = x.reshape(-1, _GROUP_SIZE)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).float()
    scale = absmax / _E2M1_MAX                                   # (num_groups, 1)
    y = (x.float() / scale).clamp(-_E2M1_MAX, _E2M1_MAX)
    q = _quantize_to_e2m1(y)                                     # values on E2M1 grid
    return ('e2m1_block', q.to(torch.bfloat16), scale, orig_shape, orig_numel, orig_dtype)


def _unpack_e2m1_block(packed):
    _tag, q, scale, orig_shape, orig_numel, orig_dtype = packed
    x = q.float() * scale
    x = x.reshape(-1)[:orig_numel]
    return x.reshape(orig_shape).to(orig_dtype)


def _pack_e2m1_chan_4d(tensor: torch.Tensor):
    """True FP4 (E2M1) with per-channel scale (HyC-LoRA-axis + FP grid)."""
    assert tensor.dim() == 4, "_pack_e2m1_chan_4d expects (B, H, L, dh)"
    B, H, L, dh = tensor.shape
    orig_dtype = tensor.dtype
    x_ch = tensor.transpose(1, 2).contiguous().reshape(B * L, H * dh)
    absmax = x_ch.abs().amax(dim=0).clamp(min=1e-8).float()       # (D,)
    scale = absmax / _E2M1_MAX
    y = (x_ch.float() / scale).clamp(-_E2M1_MAX, _E2M1_MAX)
    q = _quantize_to_e2m1(y)
    return ('e2m1_chan', q.to(torch.bfloat16), scale, (B, H, L, dh), orig_dtype)


def _unpack_e2m1_chan_4d(packed):
    _tag, q, scale, orig_shape, orig_dtype = packed
    B, H, L, dh = orig_shape
    x_ch = q.float() * scale
    x = x_ch.reshape(B, L, H, dh).transpose(1, 2).contiguous()
    return x.to(orig_dtype)


def _pack_identity_chan_4d(tensor: torch.Tensor):
    """No-op quantizer with the same transpose+reshape+contiguous chain as
    the per-channel path. If forward+backward under this pack is bit-exact vs
    no-hook, the chain itself is correct and any per-channel anomaly is
    localised to the quantiser code, not the tensor plumbing."""
    assert tensor.dim() == 4, "_pack_identity_chan_4d expects (B, H, L, dh)"
    B, H, L, dh = tensor.shape
    orig_dtype = tensor.dtype
    x_ch = tensor.transpose(1, 2).contiguous().reshape(B * L, H * dh)
    return ('identity_chan', x_ch, (B, H, L, dh), orig_dtype)


# ----------------------------------------------------------------
# GACT-style quantizer (Experiment 1, 2026-08-25).
#
# Reproduces the essentials of GACT (Chen et al. 2021):
#   * FLAT reshape: input.reshape(-1, group_size), crosses head boundaries.
#   * Group size = 256 (GACT default; ours is 128).
#   * Per-group AFFINE scale: (min, max) with zero-point, not symmetric absmax.
#   * Stochastic rounding by default (GACT's unbiasedness assumption).
#
# Purpose: show whether §5.1's Q/K collapse is specific to our E2M1 grid or a
# general property of blockwise 4-bit quantization on head-view tensors.
# ----------------------------------------------------------------

_GACT_GROUP_SIZE = 256
_GACT_LEVELS = 15.0   # 4-bit unsigned range [0, 15]


def _pack_gact_affine_block(tensor: torch.Tensor, stochastic: bool = True,
                             seed: int = 42):
    """GACT-style FLAT-reshape 4-bit affine quantization.

    Returns a tuple containing packed uint8 values plus per-group scale and
    zero-point (min). No pack_uint4 packing here: the probe reads dequant only,
    so we skip nibble packing to keep the code auditable.
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    x = tensor.reshape(-1)
    orig_numel = x.numel()
    pad = (_GACT_GROUP_SIZE - (orig_numel % _GACT_GROUP_SIZE)) % _GACT_GROUP_SIZE
    if pad > 0:
        x = F.pad(x, (0, pad))
    x = x.float().reshape(-1, _GACT_GROUP_SIZE)                 # (G, 256)
    mn = x.min(dim=-1, keepdim=True).values                      # (G, 1)
    mx = x.max(dim=-1, keepdim=True).values                      # (G, 1)
    scale = (mx - mn).clamp(min=1e-8) / _GACT_LEVELS             # (G, 1)
    y = (x - mn) / scale                                         # [0, 15]
    if stochastic:
        gen = torch.Generator(device=x.device).manual_seed(seed)
        floor_y = torch.floor(y)
        noise = torch.rand(y.shape, device=y.device, dtype=y.dtype, generator=gen)
        q = floor_y + (noise < (y - floor_y)).to(y.dtype)
    else:
        q = y.round()
    q = q.clamp(0, _GACT_LEVELS).to(torch.uint8)
    return ('gact_affine_block', q, scale, mn, orig_shape, orig_numel, orig_dtype)


def _unpack_gact_affine_block(packed):
    _tag, q, scale, mn, orig_shape, orig_numel, orig_dtype = packed
    x = q.float() * scale + mn                                   # dequantize
    x = x.reshape(-1)[:orig_numel]
    return x.reshape(orig_shape).to(orig_dtype)


def _unpack_identity_chan_4d(packed):
    _tag, x_ch, orig_shape, orig_dtype = packed
    B, H, L, dh = orig_shape
    x = x_ch.reshape(B, L, H, dh).transpose(1, 2).contiguous()
    return x.to(orig_dtype)


# ----------------------------------------------------------------
# Distribution stats (Experiment 4, 2026-08-26)
#
# Per-tensor summary numbers matching the ratios that INT4 and E2M1 zero out:
#   * INT4 has scale = absmax/7 with step 1; anything |x| < absmax/14 rounds
#     to 0. So ``frac_below_absmax_over_14`` is the fraction of elements INT4
#     silently discards.
#   * E2M1 has scale = absmax/6 and its smallest nonzero level is 0.5;
#     anything |x| < absmax/24 rounds to 0. So ``frac_below_absmax_over_24``
#     is the smaller fraction E2M1 silently discards.
# The delta between the two is exactly the near-zero mass E2M1 preserves
# that INT4 loses.
# ----------------------------------------------------------------

@torch.no_grad()
def _measure_role_stats(tensor: torch.Tensor) -> dict:
    """Single-tensor distribution stats. Cheap: at most one flatten+few reductions.

    Reports theoretical zero-out ratios (``frac_below_absmax_over_{14,24}``) AND
    empirical zero-out ratios after actually quantising the tensor with the
    same blockwise quantisers used by the probe filters. Divergence between
    theoretical and empirical measures per-group absmax variation.
    """
    x = tensor.detach().float().flatten()
    n = int(x.numel())
    if n == 0:
        return {'n': 0}
    x_abs = x.abs()
    absmax = float(x_abs.amax())
    if absmax == 0.0:
        return {'n': n, 'absmax': 0.0, 'all_zero': True}
    # Theoretical thresholds — global absmax reference.
    below14 = float((x_abs < (absmax / 14.0)).float().mean())
    below24 = float((x_abs < (absmax / 24.0)).float().mean())
    # Empirical quant-zero ratios — actual per-group scale.
    _, q_int4, _, _, orig_numel_i, _ = _pack_int4_block(tensor.detach())
    _, q_e2m1_bf16, _, _, orig_numel_e, _ = _pack_e2m1_block(tensor.detach())
    # Strip group_size padding so we measure only real elements.
    q_int4_real = q_int4.reshape(-1)[:orig_numel_i]
    q_e2m1_real = q_e2m1_bf16.reshape(-1)[:orig_numel_e].float()
    frac_zeroed_int4 = float((q_int4_real == 0).float().mean())
    frac_zeroed_e2m1 = float((q_e2m1_real == 0.0).float().mean())
    # median of |x|; median() on GPU is O(n log n) but fine for our sizes.
    med = float(x_abs.median())
    mean = float(x.mean())
    var = float(((x - mean) ** 2).mean())
    if var > 1e-20:
        kurt = float(((x - mean) ** 4).mean() / (var ** 2))
    else:
        kurt = 0.0
    return {
        'n': n,
        'absmax': absmax,
        'median_abs': med,
        'dynamic_range': absmax / max(med, 1e-12),
        'frac_below_absmax_over_14': below14,     # theoretical INT4 zero-out
        'frac_below_absmax_over_24': below24,     # theoretical E2M1 zero-out
        'frac_zeroed_int4': frac_zeroed_int4,     # empirical INT4 zero-out
        'frac_zeroed_e2m1': frac_zeroed_e2m1,     # empirical E2M1 zero-out
        'kurtosis': kurt,
        'shape_dim': int(tensor.dim()),
    }


def _aggregate_role_stats(entries: list) -> dict:
    """Weighted mean by n_elements across all pack calls seen for one role."""
    if not entries:
        return {'n_tensors': 0, 'n_elements': 0}
    entries = [e for e in entries if e.get('n', 0) > 0 and not e.get('all_zero')]
    if not entries:
        return {'n_tensors': 0, 'n_elements': 0}
    total_n = sum(e['n'] for e in entries)
    def wavg(field):
        return sum(e[field] * e['n'] for e in entries) / total_n
    return {
        'n_tensors': len(entries),
        'n_elements': total_n,
        'absmax_mean': wavg('absmax'),
        'absmax_max':  max(e['absmax']  for e in entries),
        'median_abs_mean':      wavg('median_abs'),
        'dynamic_range_mean':   wavg('dynamic_range'),
        'dynamic_range_max':    max(e['dynamic_range'] for e in entries),
        # theoretical (global absmax)
        'frac_below_absmax_over_14_mean': wavg('frac_below_absmax_over_14'),
        'frac_below_absmax_over_24_mean': wavg('frac_below_absmax_over_24'),
        'delta_frac_below_14_vs_24_mean': (
            wavg('frac_below_absmax_over_14') - wavg('frac_below_absmax_over_24')),
        # empirical (per-group absmax, after quantize)
        'frac_zeroed_int4_mean':  wavg('frac_zeroed_int4'),
        'frac_zeroed_e2m1_mean':  wavg('frac_zeroed_e2m1'),
        'delta_frac_zeroed_int4_vs_e2m1_mean': (
            wavg('frac_zeroed_int4') - wavg('frac_zeroed_e2m1')),
        'kurtosis_mean':   wavg('kurtosis'),
        'kurtosis_max':    max(e['kurtosis'] for e in entries),
    }


# ----------------------------------------------------------------
# Filter specs — closures (probe-only; production filter goes through
# `pack_skip_dims` / `pack_only_last_dims` per INVARIANT-8 spec).
# ----------------------------------------------------------------

def _alpha_paper(shape, dtype, role):
    # Paper §3.4: quantize gate_proj OUTPUT and up_proj OUTPUT plus residual.
    return role == 'residual' or role in ('gate_proj.out', 'up_proj.out')


FILTERS = [
    # baselines
    ('gamma_full',         lambda s, d, r: True),
    ('alpha_reproduction', _alpha_paper),
    ('residual_only',      lambda s, d, r: r == 'residual'),
    ('mlp_intermediate',   lambda s, d, r:
                           bool(s) and len(s) <= 3 and s[-1] == 8192),
    ('attn_proj_in',       lambda s, d, r:
                           r in ('q_proj.in', 'k_proj.in',
                                 'v_proj.in', 'o_proj.in')),
    ('attn_4d',            lambda s, d, r: len(s) == 4),
    ('drop_4d',            lambda s, d, r: len(s) <= 3),
    # ---- 4D fine breakdown (rotary vs QKV heads vs misc) ----
    # rotary cos/sin lives at (1, 1, L, head_dim). It's aliased across every
    # attention layer so uniq=0.07 MB but *count=112* pack calls hit it.
    # Compressing angular functions to 4 bits destroys position encoding.
    ('rotary_only',        lambda s, d, r: len(s) == 4 and s[1] == 1),
    # Q/K/V head views: shape[1] is num_heads or num_kv_heads.
    # Model-agnostic: any 4D tensor whose 2nd dim > 1 (rules out rotary).
    ('qkv_heads',          lambda s, d, r: len(s) == 4 and s[1] > 1),
    # attn_4d_misc: any 4D not caught by rotary_only or qkv_heads (should be empty).
    ('attn_4d_misc',       lambda s, d, r: False),
    # drop just rotary  -- proxy for "keep everything except position enc"
    ('drop_rotary',        lambda s, d, r:
                           not (len(s) == 4 and s[1] == 1)),
    # drop rotary + attention scale (fp32 3D scores)
    ('drop_pos_and_scale', lambda s, d, r:
                           not ((len(s) == 4 and s[1] == 1) or
                                (len(s) == 3 and bool(s) and s[1] == 24 and s[-1] != 128))),
    # Custom: keep γ bilevel for everything <=3D, but re-route Q/K/V head
    # views to uniform FP8 (bits=8 instead of the FP4 body mix). Coverage
    # stays 100%; memory hit is bounded by the FP8/FP4 ratio (~2x on 60 MB).
    ('qkv_heads_fp8',      ('CUSTOM_FP8_QKV',)),
    # ---- Role-based (2026-08-20 predicate check) ----
    # q_proj.out / k_proj.out live on the softmax input path (Q·Kᵀ amplifies).
    # v_proj.out is 4-D head-view of same shape but only multiplied by the
    # softmax result — no amplification. o_proj.out is downstream of softmax.
    # Splitting rank-4 into these roles tests whether the predicate is
    # "rank == 4" (bad = all 4-D heads) or "on the softmax bilinear input path"
    # (bad = q/k only, v/o fine).
    #
    # NOTE: register_attribution() tags Linear .out storage_ptrs. The
    # transpose that turns (B, L, D) into (B, H, L, dh) preserves storage_ptr,
    # so the tag flows into the 4-D view. contiguous() breaks the chain; the
    # coverage counters below (see collect_grads) surface how often that fires.
    ('qk_only',            lambda s, d, r: r in ('q_proj.out', 'k_proj.out')),
    # Q and K carry different tags already, so splitting the pair costs one
    # line each. If the two differ, the rank-4 rule can be applied to one of
    # them and the bit cost of the rule halves.
    ('q_only',             lambda s, d, r: r == 'q_proj.out'),
    # ---- 2026-09-08: depth threshold. `protect_layers < k` leaves layers
    # below k uncompressed and compresses the Q/K head views from k upward.
    # k=0 compresses everything and must reproduce qk_only exactly; k=1 is the
    # "protect layer 0 only" configuration at 1/28 of the bit cost. A tensor
    # whose tag lost its layer index has L=None and is NOT compressed, so an
    # untagged tensor cannot masquerade as layer 0.
    ('qk_from_L1',         lambda s, d, r, L: r in ('q_proj.out', 'k_proj.out')
                                              and L is not None and L >= 1),
    ('qk_from_L2',         lambda s, d, r, L: r in ('q_proj.out', 'k_proj.out')
                                              and L is not None and L >= 2),
    ('qk_from_L4',         lambda s, d, r, L: r in ('q_proj.out', 'k_proj.out')
                                              and L is not None and L >= 4),
    ('qk_from_L8',         lambda s, d, r, L: r in ('q_proj.out', 'k_proj.out')
                                              and L is not None and L >= 8),
    # k=0: the control for verification 4 -- must equal qk_only.
    ('qk_from_L0',         lambda s, d, r, L: r in ('q_proj.out', 'k_proj.out')
                                              and L is not None and L >= 0),
    ('k_only',             lambda s, d, r: r == 'k_proj.out'),
    ('v_only',             lambda s, d, r: r == 'v_proj.out'),
    ('o_only',             lambda s, d, r: r == 'o_proj.out'),
    ('qkv_all',            lambda s, d, r: r in ('q_proj.out',
                                                 'k_proj.out',
                                                 'v_proj.out')),
    # ---- Experiment 2 (2026-08-20): 2x2 grid over (granularity, format).
    # All four apply ONLY to Q/K head-views. Everything else stays uncompressed
    # (identity pass-through), so cos is directly attributable to the Q/K arm.
    ('qk_fp4_block',       ('CUSTOM_QK', 'fp4', 'block')),   # current (sanity)
    ('qk_fp4_chan',        ('CUSTOM_QK', 'fp4', 'chan')),
    ('qk_int4_block',      ('CUSTOM_QK', 'int4', 'block')),
    ('qk_int4_chan',       ('CUSTOM_QK', 'int4', 'chan')),   # HyC-LoRA condition
    # ---- Experiment 3 (2026-08-20): true E2M1 FP4 (non-uniform grid).
    # Compare uniform INT4 [-7,7] vs non-uniform E2M1 {0, +/-0.5, +/-1, +/-1.5,
    # +/-2, +/-3, +/-4, +/-6}; both at 4-bit. V arm tests whether the "safe"
    # role stays safe under FP4 encoding.
    ('qk_e2m1_block',      ('CUSTOM_QK', 'e2m1', 'block')),
    ('qk_e2m1_chan',       ('CUSTOM_QK', 'e2m1', 'chan')),
    # ---- 2026-09-04: the right-hand end of the dose-response curve.
    # Q/K at FP8 is what pack_4d_mode='fp8' actually stores, and it is the
    # condition shared by the two safe arms (E2M1 body and INT4 body), so this
    # is the x-coordinate at which they coincide. The older `qkv_heads_fp8`
    # filter is NOT a substitute: it gates on shape[1] > 1 and therefore
    # compresses V as well as Q/K, and one of its two recorded runs returned
    # cos=0.0 with an infinite norm ratio.
    ('qk_fp8_block',       ('CUSTOM_QK', 'fp8', 'block')),
    # Production per-channel INT4 (oamp.pack_hooks._pack_chan_int4_4d): nibble
    # packing and an fp16 scale, unlike `qk_int4_chan` above which keeps int8
    # codes and an fp32 scale because it only ever measured gradient error.
    # The fp16 scale rounds, so the two need not agree — this filter is the one
    # whose cosine describes what pack_4d_mode='chan_int4' actually does.
    ('qk_chan_int4_prod',  ('CUSTOM_QK', 'chan_int4', 'prod')),
    ('v_e2m1_block',       ('CUSTOM_V',  'e2m1', 'block')),
    ('v_int4_block',       ('CUSTOM_V',  'int4', 'block')),
    # ---- Experiment 4 (2026-08-26): mechanism of E2M1 gain over INT4.
    # Extends the qk/v comparison to residual (dim<=3, role=='residual') and
    # MLP intermediate (dim<=3, role in {gate_proj.out, up_proj.out,
    # down_proj.in}). Same fresh-model probe, single session.
    #
    # Reading: E2M1 gain per role tells us WHERE the non-uniform grid helps.
    #   * uniform gain across roles  -> "generally better grid" (weak claim)
    #   * concentrated on Q/K only    -> "outlier-heavy activations" (new predicate)
    #
    # 2026-08-31 refinement: residual is split into pre_attn / pre_mlp because
    # they carry different distributions (pre_mlp has the attention output
    # added on top of the entry residual, so its outlier tail may be heavier).
    ('residual_pre_attn_int4',   ('CUSTOM_RESIDUAL_PRE_ATTN', 'int4', 'block')),
    ('residual_pre_attn_e2m1',   ('CUSTOM_RESIDUAL_PRE_ATTN', 'e2m1', 'block')),
    ('residual_pre_mlp_int4',    ('CUSTOM_RESIDUAL_PRE_MLP',  'int4', 'block')),
    ('residual_pre_mlp_e2m1',    ('CUSTOM_RESIDUAL_PRE_MLP',  'e2m1', 'block')),
    ('mlp_int4',                 ('CUSTOM_MLP',      'int4', 'block')),
    ('mlp_e2m1',                 ('CUSTOM_MLP',      'e2m1', 'block')),
    # attention output (o_proj.in) — 3-D (B, L, D), post-attn, pre-o_proj.
    ('o_int4',                   ('CUSTOM_O',        'int4', 'block')),
    ('o_e2m1',                   ('CUSTOM_O',        'e2m1', 'block')),
    # Distribution-only pass: measures per-role absmax / median / dyn range /
    # frac-below-{absmax/14, absmax/24} / kurtosis. Zero compression, one
    # forward. Used to correlate distribution shape with E2M1 gain per role.
    ('dist_stats',               ('DIST_STATS',)),
    # ---- Experiment 1 (2026-08-25): GACT-style flat-reshape 4-bit affine.
    # Tests whether §5.1's Q/K collapse is a general property of blockwise 4-bit
    # quantization on head-view tensors, or specific to our E2M1 grid.
    ('qk_gact_block',      ('CUSTOM_QK', 'gact_affine', 'block')),
    ('v_gact_block',       ('CUSTOM_V',  'gact_affine', 'block')),
    # Deterministic (SR off) — isolates flat-reshape/granularity from SR variance.
    ('qk_gact_block_det',  ('CUSTOM_QK', 'gact_affine_det', 'block')),
    ('v_gact_block_det',   ('CUSTOM_V',  'gact_affine_det', 'block')),
    # ---- Sanity: identity pack with the same transpose+reshape chain as
    # per-channel quantisers. If this returns cos=1.0000, the plumbing is
    # correct and any per-channel anomaly is quantiser-local.
    ('qk_identity_chan',   ('CUSTOM_QK', 'identity', 'chan')),
]


# ----------------------------------------------------------------
# Gradient measurement
# ----------------------------------------------------------------

def make_gated_ctx(ctx: PackHooks, ptr_map: dict, filter_pred):
    """Return the same ctx with pack() wrapped to only compress matching tensors.

    ``filter_pred`` can be:
      - a callable ``(shape, dtype, role) -> bool``: True = compress via γ,
        False = pass-through.
      - the sentinel ``('CUSTOM_FP8_QKV',)``: 4D head-views go through
        ``_pack_uniform(bits=8)``; everything else uses γ bilevel.
      - the sentinel ``('CUSTOM_QK', fmt, gran)`` where fmt ∈ {'fp4','int4','fp8'}
        and gran ∈ {'block','chan'}: only Q/K head-view 4-D tensors go through
        the custom pack; everything else passes through uncompressed. This is
        the 2×2 experiment-2 arm.

    Also populates a coverage counter dict on the ctx object
    (``ctx._probe_coverage``) so callers can inspect how often each role
    was seen versus matched by the filter — useful for verifying that
    module-attribution tags actually reached pack() (transpose preserves
    storage_ptr but contiguous() breaks the chain).
    """
    original_pack = ctx.pack
    original_unpack = ctx.unpack

    # Read the filter's arity once rather than catching TypeError at call
    # time: a TypeError raised inside a filter body would otherwise be
    # silently retried with the shorter signature.
    _pred_takes_layer = False
    if callable(filter_pred):
        try:
            _pred_takes_layer = (
                len(inspect.signature(filter_pred).parameters) == 4)
        except (TypeError, ValueError):
            _pred_takes_layer = False

    is_qkv_fp8 = (isinstance(filter_pred, tuple)
                  and filter_pred and filter_pred[0] == 'CUSTOM_FP8_QKV')
    is_qk_custom = (isinstance(filter_pred, tuple)
                    and filter_pred and filter_pred[0] == 'CUSTOM_QK')
    is_v_custom = (isinstance(filter_pred, tuple)
                   and filter_pred and filter_pred[0] == 'CUSTOM_V')
    is_residual_custom = (isinstance(filter_pred, tuple)
                          and filter_pred and filter_pred[0] in ('CUSTOM_RESIDUAL',
                                                                 'CUSTOM_RESIDUAL_PRE_ATTN',
                                                                 'CUSTOM_RESIDUAL_PRE_MLP'))
    is_mlp_custom = (isinstance(filter_pred, tuple)
                     and filter_pred and filter_pred[0] == 'CUSTOM_MLP')
    is_o_custom = (isinstance(filter_pred, tuple)
                   and filter_pred and filter_pred[0] == 'CUSTOM_O')
    is_dist_stats = (isinstance(filter_pred, tuple)
                     and filter_pred and filter_pred[0] == 'DIST_STATS')
    custom_target_roles = None
    if is_qk_custom:
        _, custom_fmt, custom_gran = filter_pred
        custom_target_roles = ('q_proj.out', 'k_proj.out')
    elif is_v_custom:
        _, custom_fmt, custom_gran = filter_pred
        custom_target_roles = ('v_proj.out',)
    elif is_residual_custom:
        _, custom_fmt, custom_gran = filter_pred
        tag = filter_pred[0]
        if tag == 'CUSTOM_RESIDUAL_PRE_ATTN':
            custom_target_roles = ('residual_pre_attn',)
        elif tag == 'CUSTOM_RESIDUAL_PRE_MLP':
            custom_target_roles = ('residual_pre_mlp',)
        else:
            # legacy CUSTOM_RESIDUAL: both roles
            custom_target_roles = ('residual_pre_attn', 'residual_pre_mlp')
    elif is_mlp_custom:
        _, custom_fmt, custom_gran = filter_pred
        # MLP-intermediate activations: post-gate, post-up, and the down input.
        # All three carry the (B, L, intermediate) shape and are what the
        # backward pass would need to reconstitute the MLP grad.
        custom_target_roles = ('gate_proj.out', 'up_proj.out', 'down_proj.in')
    elif is_o_custom:
        _, custom_fmt, custom_gran = filter_pred
        # Attention output = o_proj input; 3-D (B, L, D). Post-attn/pre-o_proj.
        custom_target_roles = ('o_proj.in',)

    # `seen` counts every tensor carrying this role tag regardless of rank --
    # the rank-3 projection output and the rank-4 head view share a storage
    # pointer and so share a tag. `compressed` counts only those the filter
    # acts on, which for the Q/K and V filters is the rank-4 view alone. The
    # ratio between them is therefore a rank composition, not a coverage rate.
    # `by_rank` records the composition so the two cannot be confused, and
    # `untagged_rank4` counts rank-4 tensors that no pointer tag reached.
    # KNOWN DEFECT, not fixed (2026-09-07). The pointer map mislabels 13-23%
    # of the query head views: re-materialization gives them a storage pointer
    # that later resolves to a different role (down_proj.in is the observed
    # case, with shape (1, 24, L, 128) -- a query view, not an MLP tensor).
    # Those views escape the CUSTOM_QK filter and are measured uncompressed, so
    # the Q/K cosines this probe reports are upper bounds. We left it: the bias
    # runs toward one, i.e. against the claim the probe supports, and fixing it
    # would invalidate the per-role table, the dose-response x-axis and Figure 2
    # at once. `rank4_heads` below is what surfaced it -- compare the shape[1]
    # histogram of `unknown` and `down_proj.in` against q_proj.out.
    #
    # `rank4_heads` keys rank-4 tensors by shape[1], which separates the
    # populations the rank rule lumps together: 1 is a rotary table, 24 a query
    # head view on this model, 8 a key or value view before repeat_kv.
    coverage = defaultdict(lambda: {'seen': 0, 'compressed': 0,
                                    'rank4_by_layer': defaultdict(int),
                                    'by_rank': defaultdict(int),
                                    'rank4_heads': defaultdict(int),
                                    'rank4_shapes': defaultdict(int)})
    ctx._probe_coverage = coverage
    dist_stats = defaultdict(list)   # role -> list of _measure_role_stats dicts
    ctx._probe_dist_stats = dist_stats

    def gated(tensor):
        if not isinstance(tensor, torch.Tensor):
            return original_pack(tensor)
        ptr = tensor.untyped_storage().data_ptr()
        role_raw = ptr_map.get(ptr, 'unknown')
        role = _bucket_role(role_raw)
        layer = _layer_of(role_raw)
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype).replace('torch.', '')
        coverage[role]['seen'] += 1
        coverage[role]['by_rank'][tensor.dim()] += 1
        if tensor.dim() == 4:
            coverage[role]['rank4_heads'][int(shape[1])] += 1
            coverage[role]['rank4_shapes'][str(shape)] += 1
            # Per-layer census for rank-4 tensors. A depth sweep is only
            # readable if this is flat: a layer whose tensors lost their tag
            # escapes compression, which raises the cosine and reads as
            # "that depth did not need protecting".
            coverage[role]['rank4_by_layer'][layer] += 1
        if is_dist_stats:
            # Pass-through with per-role stats collection. Skip 'unknown' to
            # avoid contaminating aggregates with e.g. rotary cos/sin.
            if role != 'unknown':
                dist_stats[role].append(_measure_role_stats(tensor))
            return tensor
        if is_qkv_fp8:
            if len(shape) == 4 and shape[1] > 1:
                coverage[role]['compressed'] += 1
                return ctx._pack_uniform(tensor, bits=8)
            return original_pack(tensor)
        if is_qk_custom or is_v_custom:
            # Only the targeted roles take the custom path; V for CUSTOM_V,
            # Q/K for CUSTOM_QK. Everything else passes through uncompressed.
            if role in custom_target_roles and tensor.dim() == 4:
                coverage[role]['compressed'] += 1
                if custom_fmt == 'fp8' and custom_gran == 'block':
                    return ctx._pack_uniform(tensor, bits=8)  # 4-D FP8 routing
                if custom_fmt == 'chan_int4' and custom_gran == 'prod':
                    return ctx._pack_chan_int4_4d(tensor)     # production path
                if custom_fmt == 'fp4' and custom_gran == 'block':
                    return ctx._pack_uniform(tensor, bits=4)  # existing INT4[-7,7]
                if custom_fmt == 'fp4' and custom_gran == 'chan':
                    return _pack_fp4_chan_4d(tensor)          # (mislabelled INT4)
                if custom_fmt == 'int4' and custom_gran == 'block':
                    return _pack_int4_block(tensor)
                if custom_fmt == 'int4' and custom_gran == 'chan':
                    return _pack_int4_chan_4d(tensor)
                if custom_fmt == 'e2m1' and custom_gran == 'block':
                    return _pack_e2m1_block(tensor)           # true FP4 (E2M1)
                if custom_fmt == 'e2m1' and custom_gran == 'chan':
                    return _pack_e2m1_chan_4d(tensor)
                if custom_fmt == 'gact_affine' and custom_gran == 'block':
                    return _pack_gact_affine_block(tensor)                    # SR on (default)
                if custom_fmt == 'gact_affine_det' and custom_gran == 'block':
                    return _pack_gact_affine_block(tensor, stochastic=False)  # SR off
                if custom_fmt == 'identity' and custom_gran == 'chan':
                    return _pack_identity_chan_4d(tensor)
            return tensor
        if is_residual_custom or is_mlp_custom or is_o_custom:
            # 3-D activations (B, L, D). Route only the targeted roles through
            # blockwise INT4 or E2M1; everything else passes through. The
            # dim<=3 gate rules out the 4-D attn head-view tensors.
            if role in custom_target_roles and tensor.dim() <= 3:
                coverage[role]['compressed'] += 1
                if custom_fmt == 'int4' and custom_gran == 'block':
                    return _pack_int4_block(tensor)
                if custom_fmt == 'e2m1' and custom_gran == 'block':
                    return _pack_e2m1_block(tensor)
            return tensor
        if (filter_pred(shape, dtype, role, layer) if _pred_takes_layer
                else filter_pred(shape, dtype, role)):
            coverage[role]['compressed'] += 1
            return original_pack(tensor)
        return tensor

    def gated_unpack(packed):
        if isinstance(packed, tuple) and len(packed) > 0:
            tag = packed[0]
            if tag == 'int4_block':
                return _unpack_int4_block(packed)
            if tag == 'int4_chan':
                return _unpack_int4_chan_4d(packed)
            if tag == 'fp4_chan':
                return _unpack_fp4_chan_4d(packed)
            if tag == 'e2m1_block':
                return _unpack_e2m1_block(packed)
            if tag == 'e2m1_chan':
                return _unpack_e2m1_chan_4d(packed)
            if tag == 'gact_affine_block':
                return _unpack_gact_affine_block(packed)
            if tag == 'identity_chan':
                return _unpack_identity_chan_4d(packed)
        return original_unpack(packed)

    ctx.pack = gated
    ctx.unpack = gated_unpack
    return ctx, original_pack


def collect_grads(model, ids, ctx: Optional[PackHooks] = None,
                  ptr_map: Optional[dict] = None,
                  filter_pred: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
    """One forward+backward. Returns {param_name: grad.detach().float()}."""
    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    # Clear stale storage_ptr → role mappings from any previous forward.
    # The allocator reuses ptrs across calls; without this, a fresh tensor
    # can land on an old ptr and inherit a stale role tag.
    if ptr_map is not None:
        ptr_map.clear()
    if ctx is not None and filter_pred is not None:
        original_unpack_before = ctx.unpack
        _, original_pack = make_gated_ctx(ctx, ptr_map, filter_pred)
        with ctx:
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
        ctx.pack = original_pack
        ctx.unpack = original_unpack_before
    elif ctx is not None:
        with ctx:
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
    else:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()

    grads = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = p.grad.detach().clone().float()
    return grads, float(out.loss.item())


# ----------------------------------------------------------------
# Comparison metrics
# ----------------------------------------------------------------

def compare_grads(g_true: Dict[str, torch.Tensor],
                  g_test: Dict[str, torch.Tensor]) -> dict:
    per_param = {}
    for n in g_true:
        if n not in g_test:
            continue
        t = g_true[n].flatten()
        c = g_test[n].flatten()
        tn = float(t.norm())
        cn = float(c.norm())
        diff = float((c - t).norm())
        cos = float(F.cosine_similarity(t.unsqueeze(0), c.unsqueeze(0)).item()
                    if tn > 0 and cn > 0 else 0.0)
        per_param[n] = {
            'norm_true': tn, 'norm_test': cn, 'diff_norm': diff,
            'rel_err': diff / max(tn, 1e-12),
            'cos': cos,
            'norm_ratio': cn / max(tn, 1e-12),
        }
    # aggregate
    t_all = torch.cat([g_true[n].flatten() for n in g_true if n in g_test])
    c_all = torch.cat([g_test[n].flatten() for n in g_true if n in g_test])
    tn = float(t_all.norm())
    cn = float(c_all.norm())
    diff = float((c_all - t_all).norm())
    agg = {
        'norm_true': tn, 'norm_test': cn, 'diff_norm': diff,
        'rel_err': diff / max(tn, 1e-12),
        'cos': float(F.cosine_similarity(t_all.unsqueeze(0), c_all.unsqueeze(0)).item()),
        'norm_ratio': cn / max(tn, 1e-12),
    }
    return {'aggregate': agg, 'per_param': per_param}


def bucket_by_role_and_layer(per_param: dict) -> dict:
    """Slice per_param stats into buckets: lora_A/lora_B × layer∈{0,13,27}."""
    out = {}
    for name, stats in per_param.items():
        role = 'lora_A' if 'lora_A' in name else ('lora_B' if 'lora_B' in name else None)
        if role is None:
            continue
        layer = None
        for tok in name.split('.'):
            if tok.isdigit():
                layer = int(tok)
                break
        key = f'{role}_layer{layer}'
        out.setdefault(key, []).append((name, stats))
    return out


# ----------------------------------------------------------------
# Runner
# ----------------------------------------------------------------

def run_checkpoint(label: str, model, tok, param_ptrs, ptr_map,
                   ids, filter_names: Optional[list] = None,
                   body_encoding: str = 'int4') -> dict:
    vocab = model.config.vocab_size
    print(f"\n{'='*70}\n[{label}] gradient error sweep\n{'='*70}", flush=True)

    # γ ctx built once; we swap ctx.pack inside collect_grads.
    ctx = make_pack_hooks('oamp', fp8_ratio=0.20, group_size=128, min_numel=1024,
                          skip_last_dims={vocab}, param_ptrs=param_ptrs,
                          body_encoding=body_encoding)
    handles = register_attribution(model, ptr_map)   # keep alive!

    # (1) g_true — no compression
    print("  [g_true] uncompressed baseline ...", flush=True)
    g_true, loss_true = collect_grads(model, ids, ctx=None, ptr_map=ptr_map)
    print(f"    loss = {loss_true:.6f}", flush=True)

    results = {'checkpoint': label,
               'loss_true': loss_true,
               'g_true_norm': float(torch.cat(
                   [g.flatten() for g in g_true.values()]).norm()),
               'filters': {}}

    # Select which filters to run (default = all).
    all_filters = {name: pred for name, pred in FILTERS}
    if filter_names:
        selected = [(n, all_filters[n]) for n in filter_names if n in all_filters]
        missing = [n for n in filter_names if n not in all_filters]
        if missing:
            print(f"  [warn] unknown filters ignored: {missing}", flush=True)
    else:
        selected = FILTERS

    # (2) each filter
    for fname, fpred in selected:
        print(f"  [{fname}] ...", flush=True)
        g_i, loss_i = collect_grads(model, ids, ctx=ctx,
                                    ptr_map=ptr_map, filter_pred=fpred)
        cmp = compare_grads(g_true, g_i)
        a = cmp['aggregate']
        cov = dict(getattr(ctx, '_probe_coverage', {}))
        entry = {
            'loss': loss_i,
            'aggregate': a,
            'buckets': {},
            'role_coverage': {
                r: {'seen': c['seen'], 'compressed': c['compressed'],
                    'by_rank': {str(k): v for k, v in sorted(c['by_rank'].items())},
                    'rank4_heads': {str(k): v for k, v in sorted(c['rank4_heads'].items())},
                    'rank4_shapes': dict(c['rank4_shapes'])}
                for r, c in cov.items()},
        }
        # DIST_STATS filter emits per-role distribution summaries.
        if isinstance(fpred, tuple) and fpred and fpred[0] == 'DIST_STATS':
            raw = dict(getattr(ctx, '_probe_dist_stats', {}))
            by_role = {
                role: _aggregate_role_stats(entries)
                for role, entries in raw.items()
            }
            # numel share per role — required to interpret cross-role Δcos.
            # A large Δcos on a role with 1% share means less than the same
            # Δcos on a role with 40% share, but a large Δcos on a small role
            # is the signature of a predicate (E2M1 wins where INT4 loses).
            total_n = sum(s.get('n_elements', 0) for s in by_role.values())
            for role, s in by_role.items():
                s['numel_share'] = (
                    s.get('n_elements', 0) / total_n if total_n > 0 else 0.0)
            entry['dist_stats_by_role'] = by_role
            entry['dist_stats_total_numel'] = total_n
        results['filters'][fname] = entry
        # bucket into lora_A/B × layer 0/13/27
        buckets = bucket_by_role_and_layer(cmp['per_param'])
        for bk, items in buckets.items():
            # only report if layer ∈ {0, 13, 27}
            if not any(bk.endswith(f'_layer{L}') for L in (0, 13, 27)):
                continue
            cos_vals = [s['cos'] for _, s in items]
            rel_vals = [s['rel_err'] for _, s in items]
            nr_vals  = [s['norm_ratio'] for _, s in items]
            results['filters'][fname]['buckets'][bk] = {
                'cos_min': min(cos_vals), 'cos_mean': sum(cos_vals)/len(cos_vals),
                'rel_err_max': max(rel_vals), 'rel_err_mean': sum(rel_vals)/len(rel_vals),
                'norm_ratio_min': min(nr_vals), 'norm_ratio_max': max(nr_vals),
                'n_params': len(items),
            }
        print(f"    loss={loss_i:.6f}  cos={a['cos']:.4f}  "
              f"rel_err={a['rel_err']:.4f}  norm_ratio={a['norm_ratio']:.4f}",
              flush=True)

    # (3) partition sanity: drop_4d + attn_4d should reconstruct gamma_full
    # (they partition the compressible set by ndim). We check
    #   ||g_partition - g_full|| / ||g_full||
    # where g_partition = g_true + (g_drop4d - g_true) + (g_attn4d - g_true).
    # If filters are correctly non-overlapping and cover everything γ compresses,
    # this should be ~0 in exact arithmetic (up to fp32 accumulator noise).
    if 'drop_4d' in results['filters'] and 'attn_4d' in results['filters']:
        results['partition_check'] = {
            'note': 'drop_4d (dim<=3) + attn_4d (dim==4) = gamma_full',
            'sanity_gap': _partition_gap(model, ids, ctx, ptr_map),
        }
    return results


def _partition_gap(model, ids, ctx, ptr_map):
    """Compute ||(g_drop4d - g_true) + (g_attn4d - g_true) - (g_full - g_true)|| /
    ||g_full - g_true||. Exact partitioning => 0."""
    g_true, _ = collect_grads(model, ids, ctx=None)
    g_full, _ = collect_grads(model, ids, ctx=ctx, ptr_map=ptr_map,
                              filter_pred=lambda s, d, r: True)
    g_d4, _ = collect_grads(model, ids, ctx=ctx, ptr_map=ptr_map,
                            filter_pred=lambda s, d, r: len(s) <= 3)
    g_a4, _ = collect_grads(model, ids, ctx=ctx, ptr_map=ptr_map,
                            filter_pred=lambda s, d, r: len(s) == 4)

    def flat(g):
        return torch.cat([g[n].flatten() for n in sorted(g_true)])

    t = flat(g_true)
    f = flat(g_full) - t
    e = (flat(g_d4) - t) + (flat(g_a4) - t) - f
    return {
        'gap_rel': float(e.norm() / max(f.norm(), 1e-12)),
        'diff_norm_partition_sum': float(((flat(g_d4) - t) + (flat(g_a4) - t)).norm()),
        'diff_norm_full':          float(f.norm()),
    }


def main():
    global MODEL, ADAPTER
    import argparse
    p = argparse.ArgumentParser(description="Per-role gradient error probe.")
    p.add_argument('--model', default=MODEL,
                   help="HF model id or alias (default: env GRAD_PROBE_MODEL).")
    p.add_argument('--adapter', default=ADAPTER,
                   help="Path to LoRA adapter for checkpoint B (default: env "
                        "GRAD_PROBE_ADAPTER, or blank to skip).")
    p.add_argument('--filters', default=None,
                   help="Comma-separated filter names; omit to run all. "
                        "See FILTERS list in this file for the catalog.")
    p.add_argument('--output', default=None,
                   help="Output JSON path (default: results/gradient_error_probe_<ts>.json).")
    p.add_argument('--skip_fresh', action='store_true',
                   help="Skip the fresh-model checkpoint A.")
    p.add_argument('--group_size', type=int, default=None,
                   help="Block size for per-block scales (default 128, or env "
                        "GRAD_PROBE_GROUP_SIZE). Sweeping this changes bits per "
                        "element as 16/g, so report the pair together.")
    p.add_argument('--body_encoding', choices=['int4', 'e2m1'], default='int4',
                   help="4-bit body encoding used by the production PackHooks "
                        "path. Default int4 matches pre-2026-08-20 behaviour.")
    args = p.parse_args()

    if args.group_size is not None:
        global _GROUP_SIZE
        _GROUP_SIZE = args.group_size
        print(f"[probe] group_size = {_GROUP_SIZE} "
              f"(scale overhead {16 / _GROUP_SIZE:.4f} bits/element)", flush=True)

    # CLI overrides module-level env defaults for this run only.
    MODEL = args.model
    ADAPTER = args.adapter

    torch.manual_seed(0)
    ids = torch.randint(0, 128256, (B, L), device=DEV)   # fixed batch

    filter_names = None
    if args.filters:
        filter_names = [s.strip() for s in args.filters.split(',') if s.strip()]

    all_results = {
        'meta': {
            'model': MODEL, 'B': B, 'L': L,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'filters_requested': filter_names,
            'note': 'Adapter is from pilot_sr_lr (SR+lr=5e-5, 300 steps). '
                    'Not a pure γ-trained checkpoint; caveats apply for trained-state result.',
        },
        'checkpoints': {},
    }

    # ---- Checkpoint A: step 0 (fresh Arm 4) ----
    if not args.skip_fresh:
        ptr_map: dict = {}
        m, tok, param_ptrs = load_arm4_fresh()
        ra = run_checkpoint('step0_fresh', m, tok, param_ptrs, ptr_map, ids,
                            filter_names=filter_names,
                            body_encoding=args.body_encoding)
        all_results['checkpoints']['step0_fresh'] = ra
        del m
        torch.cuda.empty_cache()

    # ---- Checkpoint B: trained γ adapter (opt-in via env var GRAD_PROBE_ADAPTER) ----
    if ADAPTER and os.path.isdir(ADAPTER):
        ptr_map = {}
        mB, _, ptrs_B = load_arm4_trained(ADAPTER)
        rb = run_checkpoint('step300_gamma', mB, tok, ptrs_B, ptr_map, ids,
                            filter_names=filter_names,
                            body_encoding=args.body_encoding)
        all_results['checkpoints']['step300_gamma'] = rb
        del mB
        torch.cuda.empty_cache()
    else:
        print(f"[info] skipping checkpoint B (adapter='{ADAPTER}' not a directory)",
              flush=True)

    ts = all_results['meta']['timestamp']
    if args.output:
        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(out_path)
    else:
        out_dir = '/app/HMA_Project/results'
        out_path = os.path.join(out_dir, f'gradient_error_probe_{ts}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
