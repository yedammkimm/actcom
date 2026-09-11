"""Anchor churn probe — how stable is OAMP's fp8 anchor mask across steps?

Motivation: γ selects K = ratio × G groups per tensor by max-abs importance.
If max-abs is data-dependent, the selected set (fp8 mask) may drift step to
step. Seed-specific data ordering → seed-specific anchor trajectories → seed
variance. This probe measures consecutive-step Jaccard(M_t, M_{t+1}) at
representative layers.

Setup:
  - Arm 4 (NF4 + apply_dtype_policy + LoRA r=16 targets=q/k/v/o)
  - 100 optimizer steps, batch_size=1, grad_accum=4 (matches production)
  - Fixed input sequence per step (deterministic tokenizer replay) so
    routing variation comes purely from parameter drift, not batch shuffle.
  - Monkey-patch `_select_anchor_mask`: record (step, role_key, shape, mask.cpu())
  - Only track buckets {layer 0/13/27} × {up_proj.in, up_proj.out,
    gate_proj.out, down_proj.in, o_proj.in} (avoid mask storage blowup).

Metric per (role, layer):
    jacc[t] = |M_t ∩ M_{t+1}| / |M_t ∪ M_{t+1}|
Report min / mean / max Jaccard, plus mask-size and how many groups the mask
selects (= K = ratio × G).

Interpretation guide:
    mean ≥ 0.9  : anchor set essentially frozen after warmup (churn hypothesis rejected)
    0.3 - 0.7   : rapid rotation each step → data-driven churn is real
    < 0.3       : mask barely correlated across steps → strong candidate for
                  EMA-smoothed importance rescore.

Output: results/anchor_churn_probe_<ts>.json + printed summary.
"""

from __future__ import annotations

import os
import sys
import json
import time
from collections import defaultdict, Counter
from typing import Optional

os.environ.setdefault("HF_HOME", "/app/hf_cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model, TaskType,
                  prepare_model_for_kbit_training)
try:
    from bitsandbytes.optim import PagedAdamW8bit
except ImportError:
    PagedAdamW8bit = torch.optim.AdamW

from oamp.pack_hooks import PackHooks, make_pack_hooks, collect_param_ptrs
from oamp.dtype_policy import apply_dtype_policy

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CACHE = "/app/hf_cache"
DEV = torch.device("cuda:0")
B, L = 1, 512
N_STEPS = 100
GRAD_ACCUM = 4
LR = 2e-4
SEED = 42

TRACKED_LAYERS = (0, 13, 27)
TRACKED_ROLES = ('up_proj.in', 'up_proj.out', 'gate_proj.out',
                 'down_proj.in', 'o_proj.in')


def load_arm4():
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
                      lora_dropout=0.05,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      bias='none')
    m = get_peft_model(m, lcfg)
    m, param_ptrs, _ = apply_dtype_policy(m, weight_quant='nf4', bf16_rmsnorm=True)
    m.train()
    return m, tok, param_ptrs


_MODULE_BUCKETS = [
    ('q_proj', 'q_proj'), ('k_proj', 'k_proj'), ('v_proj', 'v_proj'),
    ('o_proj', 'o_proj'), ('gate_proj', 'gate_proj'),
    ('up_proj', 'up_proj'), ('down_proj', 'down_proj'),
]


def _short(name: str) -> str:
    for prefix in ('base_model.model.model.', 'base_model.model.'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _bucket_role(role: str) -> str:
    if role.endswith('.residual'):
        return 'residual'
    for needle, bucket in _MODULE_BUCKETS:
        if needle in role:
            io = 'in' if role.endswith('.in') else ('out' if role.endswith('.out') else '?')
            return f'{bucket}.{io}'
    return 'unknown'


def _layer_from_role(role: str) -> Optional[int]:
    # roles like "layers.13.mlp.gate_proj.out"
    for tok in role.split('.'):
        if tok.isdigit():
            return int(tok)
    return None


def register_attribution(model, ptr_map: dict):
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
                    ptr_map[inputs[0].untyped_storage().data_ptr()] = f'{sn}.residual'

            handles.append(mod.register_forward_pre_hook(declhook))
    return handles


def instrument_ctx_for_churn(ctx: PackHooks, ptr_map: dict, mask_log: dict,
                             step_ref: list, pack_stats_ref: list):
    """Wrap _select_anchor_mask to record masks per (role, layer) for tracked
    buckets. mask_log[key] is a list of numpy uint8 arrays, one per step_ref[0]."""
    original_pack = ctx.pack
    original_select = ctx._select_anchor_mask

    def select_wrap(x, num_groups, K):
        m = original_select(x, num_groups, K)
        # attribution: find role for x
        ptr = x.untyped_storage().data_ptr()
        role_raw = ptr_map.get(ptr, 'unknown')
        layer = _layer_from_role(role_raw)
        role_bucket = _bucket_role(role_raw)
        if layer in TRACKED_LAYERS and role_bucket in TRACKED_ROLES:
            key = f'layer{layer}_{role_bucket}'
            step = step_ref[0]
            entry = mask_log.setdefault(key, {})
            entry.setdefault(step, m.detach().cpu().numpy().astype('uint8'))
            entry.setdefault('K', int(K))
            entry.setdefault('num_groups', int(num_groups))
        return m

    ctx._select_anchor_mask = select_wrap


def jaccard(a, b) -> float:
    """Both boolean arrays of same length."""
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def summarize(mask_log: dict) -> dict:
    """For each (layer, role) key, compute consecutive-step Jaccard series."""
    out = {}
    for key, entry in mask_log.items():
        # entry contains {step_int: mask_array, 'K': int, 'num_groups': int}
        step_keys = sorted([k for k in entry if isinstance(k, int)])
        if len(step_keys) < 2:
            continue
        j_series = []
        for i in range(len(step_keys) - 1):
            s0, s1 = step_keys[i], step_keys[i + 1]
            m0 = entry[s0].astype(bool)
            m1 = entry[s1].astype(bool)
            if m0.shape != m1.shape:
                continue    # tensor shape changed (rare)
            j_series.append(jaccard(m0, m1))
        if not j_series:
            continue
        out[key] = {
            'K': entry.get('K'),
            'num_groups': entry.get('num_groups'),
            'n_pairs': len(j_series),
            'jacc_min': min(j_series),
            'jacc_mean': sum(j_series) / len(j_series),
            'jacc_max': max(j_series),
            'jacc_first10_mean': sum(j_series[:10]) / min(10, len(j_series)),
            'jacc_last10_mean': sum(j_series[-10:]) / min(10, len(j_series)),
        }
    return out


def main():
    torch.manual_seed(SEED)
    m, tok, param_ptrs = load_arm4()

    vocab = m.config.vocab_size
    ptr_map: dict = {}
    hooks = register_attribution(m, ptr_map)

    ctx = make_pack_hooks('oamp', fp8_ratio=0.20, group_size=128, min_numel=1024,
                          skip_last_dims={vocab}, param_ptrs=param_ptrs)

    mask_log: dict = {}
    step_ref = [0]
    pack_stats_ref = [None]
    instrument_ctx_for_churn(ctx, ptr_map, mask_log, step_ref, pack_stats_ref)

    lora_params = [p for p in m.parameters() if p.requires_grad]
    opt = PagedAdamW8bit(lora_params, lr=LR)
    # Deterministic replayed input each step; churn comes from param drift only.
    ids_seed = torch.randint(0, vocab, (B, L), device=DEV,
                             generator=torch.Generator(device=DEV).manual_seed(SEED))

    print(f"Anchor churn probe: {N_STEPS} steps, seed={SEED}", flush=True)
    t0 = time.time()
    for step in range(N_STEPS):
        step_ref[0] = step
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            with ctx:
                out = m(input_ids=ids_seed, labels=ids_seed)
                (out.loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  step {step+1}/{N_STEPS}  loss={out.loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    summary = summarize(mask_log)

    print("\n[per (layer, role) Jaccard]")
    print(f"{'key':<28} {'K':>5} {'G':>6} {'n':>4} "
          f"{'j_min':>7} {'j_mean':>7} {'j_max':>7} "
          f"{'first10':>8} {'last10':>8}")
    for k in sorted(summary):
        s = summary[k]
        print(f"{k:<28} {s['K']:>5} {s['num_groups']:>6} {s['n_pairs']:>4} "
              f"{s['jacc_min']:>7.3f} {s['jacc_mean']:>7.3f} {s['jacc_max']:>7.3f} "
              f"{s['jacc_first10_mean']:>8.3f} {s['jacc_last10_mean']:>8.3f}")

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'/app/HMA_Project/results/anchor_churn_probe_{ts}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'meta': {'model': MODEL, 'n_steps': N_STEPS, 'seed': SEED,
                            'timestamp': ts},
                   'summary': summary}, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)


if __name__ == '__main__':
    main()
