"""Shape-breakdown audit for OAMP coverage reduction (Arm 4 conditions).

Replicates production loading exactly:
  1. NF4 base + prepare_model_for_kbit_training(gc=False)
  2. get_peft_model(LoRA r=16 α=32 dropout=0.0 targets=q/k/v/o)
  3. apply_dtype_policy(bf16_rmsnorm=True)   ← THE key fix (previous audit missed this)

Then registers forward hooks on every nn.Linear so pack() can look up which
module a saved tensor came from — turning "112=4×28 → q/k/v/o line" from an
identity GUESS into an identity CERTAINTY.

Records per (shape, dtype, module_role) group:
  count, bytes_raw, bytes_uniq, orig_bytes

Prints raw filter projections A/B/C/D/E for the coverage-reduction study.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from collections import defaultdict, Counter

os.environ.setdefault("HF_HOME", "/app/hf_cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from oamp.pack_hooks import PackHooks, make_pack_hooks, collect_param_ptrs
from oamp.dtype_policy import apply_dtype_policy

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CACHE = "/app/hf_cache"
DEV = torch.device("cuda:0")
B = 1  # L parameterized per audit run


def _load_arm4_model():
    """Match run_experiment.py::_load_model + apply_dtype_policy exactly."""
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
    m, param_ptrs, dtype_report = apply_dtype_policy(
        m, weight_quant='nf4', bf16_rmsnorm=True)
    m.train()
    return m, tok, param_ptrs, dtype_report


# ----------------------------------------------------------------
# Module attribution: storage_ptr -> "layers.13.mlp.gate_proj.input"
# ----------------------------------------------------------------

def _short_name(name: str) -> str:
    # base_model.model.model.layers.5.mlp.gate_proj -> layers.5.mlp.gate_proj
    for prefix in ('base_model.model.model.', 'base_model.model.'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def register_attribution(model, ptr_map: dict) -> list:
    """Every Linear.forward records storage_ptrs of its input & output."""
    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        short = _short_name(name)

        def hook(m, inputs, output, sn=short):
            if inputs and isinstance(inputs[0], torch.Tensor):
                ptr_map[inputs[0].untyped_storage().data_ptr()] = f'{sn}.in'
            if isinstance(output, torch.Tensor):
                ptr_map[output.untyped_storage().data_ptr()] = f'{sn}.out'

        handles.append(mod.register_forward_hook(hook))
    return handles


# ----------------------------------------------------------------
# Pack instrumentation
# ----------------------------------------------------------------

def _packed_bytes(packed) -> int:
    if isinstance(packed, torch.Tensor):
        return packed.numel() * packed.element_size()
    if isinstance(packed, tuple):
        return sum(_packed_bytes(p) for p in packed
                   if isinstance(p, (torch.Tensor, tuple)))
    return 0


def _packed_numel(packed) -> int:
    if isinstance(packed, torch.Tensor):
        return packed.numel()
    if isinstance(packed, tuple):
        return sum(_packed_numel(p) for p in packed
                   if isinstance(p, (torch.Tensor, tuple)))
    return 0


def instrument(ctx: PackHooks, ptr_map: dict):
    ctx._agg_count = Counter()               # (shape, dtype, role) -> N
    ctx._agg_raw = defaultdict(int)          # bytes, no dedup
    ctx._agg_uniq = defaultdict(int)         # bytes, dedup by storage_ptr+shape
    ctx._agg_orig = defaultdict(int)         # original bytes, no dedup
    ctx._agg_numel_raw = defaultdict(int)    # elements, no dedup
    ctx._agg_numel_uniq = defaultdict(int)   # elements, dedup by same key as _agg_uniq
    ctx._seen = defaultdict(set)
    ctx._role_examples = defaultdict(set)    # (shape, dtype) -> {role strings}
    ctx._dtype_dist_raw = Counter()          # dtype -> total pre-pack bytes
    ctx._dtype_dist_numel = Counter()        # dtype -> total pre-pack elements
    # Per-branch pre-pack accounting (needed for true DCR coverage).
    _BRANCHES = ('bilevel', 'uniform', 'skip_small', 'skip_head',
                 'skip_misaligned', 'skip_param', 'skip_non_float', 'skip_by_dim')
    ctx._branch_bytes_raw = {b: 0 for b in _BRANCHES}
    ctx._branch_bytes_uniq = {b: 0 for b in _BRANCHES}
    ctx._branch_numel_raw = {b: 0 for b in _BRANCHES}
    ctx._branch_numel_uniq = {b: 0 for b in _BRANCHES}
    ctx._branch_seen = {b: set() for b in _BRANCHES}
    original_pack = ctx.pack

    def dkey(t):
        return (t.untyped_storage().data_ptr(), tuple(t.shape),
                tuple(t.stride()), t.storage_offset())

    def wrapped(tensor):
        if isinstance(tensor, torch.Tensor):
            pre_bytes = tensor.numel() * tensor.element_size()
            pre_numel = tensor.numel()
            dt = str(tensor.dtype).replace('torch.', '')
            ptr = tensor.untyped_storage().data_ptr()
            role = ptr_map.get(ptr, 'unknown')
            role_bucket = _bucket_role(role)
            key = (tuple(tensor.shape), dt, role_bucket)
            shape_key = (tuple(tensor.shape), dt)
            d = dkey(tensor)
            ctx._dtype_dist_raw[dt] += pre_bytes
            ctx._dtype_dist_numel[dt] += pre_numel
        else:
            key = None
            role_bucket = None
            d = None
        pre_snap = {b: ctx.pack_stats.get(b, 0) for b in _BRANCHES}
        before = ctx.pack_stats.get('kept', 0)
        result = original_pack(tensor)
        after = ctx.pack_stats.get('kept', 0)
        # Detect which branch just incremented (at most one per pack call).
        branch = None
        for b in _BRANCHES:
            if ctx.pack_stats.get(b, 0) > pre_snap[b]:
                branch = b
                break
        if key is not None and branch is not None:
            ctx._branch_bytes_raw[branch] += pre_bytes
            ctx._branch_numel_raw[branch] += pre_numel
            if d not in ctx._branch_seen[branch]:
                ctx._branch_seen[branch].add(d)
                ctx._branch_bytes_uniq[branch] += pre_bytes
                ctx._branch_numel_uniq[branch] += pre_numel
        if key is not None and after > before:
            b = _packed_bytes(result)
            ctx._agg_count[key] += 1
            ctx._agg_raw[key] += b
            ctx._agg_orig[key] += pre_bytes
            ctx._agg_numel_raw[key] += pre_numel
            ctx._role_examples[shape_key].add(role)
            if d not in ctx._seen[key]:
                ctx._seen[key].add(d)
                ctx._agg_uniq[key] += b
                ctx._agg_numel_uniq[key] += pre_numel
        return result

    ctx.pack = wrapped


_MODULE_BUCKETS = [
    ('q_proj', 'q_proj'), ('k_proj', 'k_proj'), ('v_proj', 'v_proj'),
    ('o_proj', 'o_proj'), ('gate_proj', 'gate_proj'),
    ('up_proj', 'up_proj'), ('down_proj', 'down_proj'),
    ('lora_A', 'lora'), ('lora_B', 'lora'),
    ('lm_head', 'lm_head'),
]


def _bucket_role(role: str) -> str:
    for needle, bucket in _MODULE_BUCKETS:
        if needle in role:
            io = 'in' if role.endswith('.in') else ('out' if role.endswith('.out') else '?')
            return f'{bucket}.{io}'
    return 'unknown'


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def _identity_hint(count: int) -> str:
    for factor, label in [(4, '4x28 q/k/v/o'), (3, '3x28 MLP g/u/d'),
                          (2, '2x28 attn-pair'), (1, '1x28 layer-once')]:
        if count == factor * 28:
            return label
    return f'?x28 (n={count})'


def run_audit(L: int):
    print(f"\n{'='*70}\n=== AUDIT RUN: B={B}  L={L} ===\n{'='*70}", flush=True)
    print("Loading Arm 4 model (NF4 + LoRA + apply_dtype_policy) ...", flush=True)
    m, tok, param_ptrs, dtype_report = _load_arm4_model()
    print(f"dtype_report keys: {list(dtype_report.keys())}", flush=True)
    print(f"  weight_quant     = {dtype_report.get('weight_quant')}", flush=True)
    print(f"  bf16_rmsnorm     = {dtype_report.get('bf16_rmsnorm')}", flush=True)
    print(f"  rmsnorm_swapped  = {dtype_report.get('rmsnorm_swapped')}", flush=True)
    print(f"  rmsnorm_classes  = {dtype_report.get('rmsnorm_classes_seen')}", flush=True)
    print(f"  norms dtypes     = {dtype_report.get('norms')}", flush=True)
    print(f"  lora dtypes      = {dtype_report.get('lora')}", flush=True)
    print(f"  embed dtype      = {dtype_report.get('embed')}", flush=True)
    print(f"  lm_head dtype    = {dtype_report.get('lm_head')}", flush=True)

    ptr_map: dict = {}
    attr_handles = register_attribution(m, ptr_map)

    vocab = m.config.vocab_size
    hidden = m.config.hidden_size
    inter = m.config.intermediate_size
    ctx = make_pack_hooks('oamp', fp8_ratio=0.20, group_size=128, min_numel=1024,
                          skip_last_dims={vocab}, param_ptrs=param_ptrs)
    instrument(ctx, ptr_map)

    ids = torch.randint(0, vocab, (B, L), device=DEV)
    print(f"Forward+backward B={B} L={L}  (hidden={hidden}, inter={inter}) ...", flush=True)
    with ctx:
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()

    for h in attr_handles:
        h.remove()

    total_raw = sum(ctx._agg_raw.values())
    total_uniq = sum(ctx._agg_uniq.values())
    total_orig = sum(ctx._agg_orig.values())
    total_numel_raw = sum(ctx._agg_numel_raw.values())
    total_numel_uniq = sum(ctx._agg_numel_uniq.values())
    print(f"\npack_stats = {dict(ctx.pack_stats)}")
    print(f"unique (shape,dtype,role) buckets: {len(ctx._agg_raw)}")
    print(f"bytes  raw = {total_raw/1e6:6.1f} MB   uniq = {total_uniq/1e6:6.1f} MB"
          f"   orig = {total_orig/1e6:6.1f} MB   raw/uniq = {total_raw/max(total_uniq,1):.2f}x")
    print(f"numel  raw = {total_numel_raw/1e6:6.2f} M    uniq = {total_numel_uniq/1e6:6.2f} M"
          f"   raw/uniq = {total_numel_raw/max(total_numel_uniq,1):.2f}x")

    print("\n[Arm 4 dtype verification -- all pre-pack tensors by dtype]")
    total_dist_bytes = sum(ctx._dtype_dist_raw.values())
    total_dist_numel = sum(ctx._dtype_dist_numel.values())
    for dt, nb in sorted(ctx._dtype_dist_raw.items(), key=lambda x: -x[1]):
        nnum = ctx._dtype_dist_numel[dt]
        print(f"  {dt:<20}  bytes={nb/1e6:8.2f} MB ({nb/max(1,total_dist_bytes)*100:5.1f}%)   "
              f"numel={nnum/1e6:8.2f} M ({nnum/max(1,total_dist_numel)*100:5.1f}%)")
    if any('float32' in dt for dt in ctx._dtype_dist_raw):
        f32 = sum(nb for dt, nb in ctx._dtype_dist_raw.items() if 'float32' in dt)
        f32n = sum(n for dt, n in ctx._dtype_dist_numel.items() if 'float32' in dt)
        print(f"  ! fp32 detected: {f32/1e6:.2f} MB ({f32n/1e6:.2f} M elems) — expected sources: logits + minor mask/scale")
    else:
        print("  (no fp32 among saved tensors)")

    # ---------- DCR branch-level coverage (kept vs skip_param) ----------
    print("\n[DCR branch-level counters -- pre-pack, from pack_stats deltas]")
    print(f"  {'branch':<20} {'bytes_raw_MB':>14} {'bytes_uniq_MB':>15} "
          f"{'numel_raw_M':>13} {'numel_uniq_M':>14}")
    print("  " + "-" * 76)
    for br, nb in sorted(ctx._branch_bytes_raw.items(), key=lambda x: -x[1]):
        bu = ctx._branch_bytes_uniq[br]
        nr = ctx._branch_numel_raw[br]
        nu = ctx._branch_numel_uniq[br]
        print(f"  {br:<20} {nb/1e6:>14.2f} {bu/1e6:>15.2f} "
              f"{nr/1e6:>13.2f} {nu/1e6:>14.2f}")

    packed_branches = ('bilevel', 'uniform')
    packed_bytes_raw = sum(ctx._branch_bytes_raw[b] for b in packed_branches)
    packed_bytes_uniq = sum(ctx._branch_bytes_uniq[b] for b in packed_branches)
    packed_numel_raw = sum(ctx._branch_numel_raw[b] for b in packed_branches)
    packed_numel_uniq = sum(ctx._branch_numel_uniq[b] for b in packed_branches)

    denom_bytes_raw = sum(ctx._branch_bytes_raw[b] for b in ctx._branch_bytes_raw
                          if b != 'skip_param')
    denom_bytes_uniq = sum(ctx._branch_bytes_uniq[b] for b in ctx._branch_bytes_uniq
                           if b != 'skip_param')
    denom_numel_raw = sum(ctx._branch_numel_raw[b] for b in ctx._branch_numel_raw
                          if b != 'skip_param')
    denom_numel_uniq = sum(ctx._branch_numel_uniq[b] for b in ctx._branch_numel_uniq
                           if b != 'skip_param')

    def pct(n, d): return 100.0 * n / d if d > 0 else float('nan')
    print("\n[DCR coverage -- (bilevel + uniform) / (total - skip_param)]")
    print(f"  bytes    raw   = {packed_bytes_raw/1e6:9.2f} / {denom_bytes_raw/1e6:9.2f} MB "
          f"= {pct(packed_bytes_raw, denom_bytes_raw):>6.2f}%")
    print(f"  bytes    uniq  = {packed_bytes_uniq/1e6:9.2f} / {denom_bytes_uniq/1e6:9.2f} MB "
          f"= {pct(packed_bytes_uniq, denom_bytes_uniq):>6.2f}%")
    print(f"  numel    raw   = {packed_numel_raw/1e6:9.2f} / {denom_numel_raw/1e6:9.2f} M  "
          f"= {pct(packed_numel_raw, denom_numel_raw):>6.2f}%")
    print(f"  numel    uniq  = {packed_numel_uniq/1e6:9.2f} / {denom_numel_uniq/1e6:9.2f} M  "
          f"= {pct(packed_numel_uniq, denom_numel_uniq):>6.2f}%")

    print("\n[per (shape, dtype, role) -- sorted by raw]")
    print(f"{'shape':<28} {'dtype':<10} {'role':<18} {'count':>5} "
          f"{'raw_MB':>7} {'uniq_MB':>7} {'orig_MB':>7}  {'hint'}")
    rows = sorted(ctx._agg_raw.items(), key=lambda kv: -kv[1])
    for (shape, dtype, role), raw in rows:
        cnt = ctx._agg_count[(shape, dtype, role)]
        uniq = ctx._agg_uniq[(shape, dtype, role)]
        orig = ctx._agg_orig[(shape, dtype, role)]
        print(f"{str(shape):<28} {dtype:<10} {role:<18} {cnt:>5} "
              f"{raw/1e6:>7.2f} {uniq/1e6:>7.2f} {orig/1e6:>7.2f}  "
              f"{_identity_hint(cnt)}")

    # ---------- grouped by last_dim + ndim (from shape only, ignoring role) ----------
    def _group_by(fn):
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for (shape, dtype, role), raw in ctx._agg_raw.items():
            key = fn(shape, dtype)
            agg[key][0] += raw
            agg[key][1] += ctx._agg_uniq[(shape, dtype, role)]
            agg[key][2] += ctx._agg_orig[(shape, dtype, role)]
            agg[key][3] += ctx._agg_count[(shape, dtype, role)]
        return agg

    print("\n[grouped by last_dim]")
    for last, (raw, uniq, orig, cnt) in sorted(
            _group_by(lambda s, d: s[-1]).items(), key=lambda x: -x[1][0]):
        print(f"  last={last:<6} raw {raw/1e6:6.1f} MB ({raw/total_raw*100:4.1f}%)  "
              f"uniq {uniq/1e6:6.1f} MB  orig {orig/1e6:6.1f} MB  count={cnt}")

    print("\n[grouped by module bucket]")
    for role, (raw, uniq, orig, cnt) in sorted(
            _group_by(lambda s, d: 'role_only').items(), key=lambda x: -x[1][0]):
        pass  # unused
    role_agg = defaultdict(lambda: [0, 0, 0, 0])
    for (shape, dtype, role), raw in ctx._agg_raw.items():
        role_agg[role][0] += raw
        role_agg[role][1] += ctx._agg_uniq[(shape, dtype, role)]
        role_agg[role][2] += ctx._agg_orig[(shape, dtype, role)]
        role_agg[role][3] += ctx._agg_count[(shape, dtype, role)]
    for role, (raw, uniq, orig, cnt) in sorted(role_agg.items(), key=lambda x: -x[1][0]):
        print(f"  {role:<20}  raw {raw/1e6:6.1f} MB ({raw/total_raw*100:4.1f}%)  "
              f"uniq {uniq/1e6:6.1f} MB  count={cnt}")

    # ---------- filter projections ----------
    def sum_matching(pred):
        raw = uniq = orig = cnt = 0
        numel_raw = numel_uniq = 0
        for (shape, dtype, role), r in ctx._agg_raw.items():
            if pred(shape, dtype, role):
                raw += r
                uniq += ctx._agg_uniq[(shape, dtype, role)]
                orig += ctx._agg_orig[(shape, dtype, role)]
                cnt += ctx._agg_count[(shape, dtype, role)]
                numel_raw += ctx._agg_numel_raw[(shape, dtype, role)]
                numel_uniq += ctx._agg_numel_uniq[(shape, dtype, role)]
        return raw, uniq, orig, cnt, numel_raw, numel_uniq

    NUMEL = 1_000_000
    filters = [
        ('A', f'last in {{{hidden},{inter}}} AND dim<=3',
            lambda s, d, r: len(s) <= 3 and s[-1] in (hidden, inter)),
        ('B', 'dim <= 3 (drop 4D)',
            lambda s, d, r: len(s) <= 3),
        ('C', f'numel >= {NUMEL:,}',
            lambda s, d, r: math.prod(s) >= NUMEL if s else False),
        ('D', f'last == {inter} (MLP intermediate)',
            lambda s, d, r: bool(s) and s[-1] == inter),
        ('E', f'last == {hidden}, dim<=3 (residual+proj-inputs+o_out)',
            lambda s, d, r: len(s) <= 3 and bool(s) and s[-1] == hidden),
        ('R1', 'role: gate_proj/up_proj/down_proj (paper alpha MLP)',
            lambda s, d, r: r in ('gate_proj.in', 'up_proj.in', 'down_proj.in')),
        ('R2', 'role: q/k/v/o_proj (attention line)',
            lambda s, d, r: r in ('q_proj.in', 'k_proj.in', 'v_proj.in', 'o_proj.in')),
    ]

    print("\n[filter projections -- SURVIVING mass under each filter]")
    print(f"  denom: total_raw_bytes={total_raw/1e6:.1f} MB  total_uniq_bytes={total_uniq/1e6:.1f} MB"
          f"  total_uniq_numel={total_numel_uniq/1e6:.2f} M")
    header = (f"{'ID':<4} {'description':<52} "
              f"{'byte_raw%':>10} {'byte_uniq%':>11} {'elem_uniq%':>11}")
    print(header)
    print('-' * len(header))
    for tag, desc, pred in filters:
        raw, uniq, orig, cnt, nraw, nuniq = sum_matching(pred)
        b_raw_pct = raw / max(1, total_raw) * 100
        b_uniq_pct = uniq / max(1, total_uniq) * 100
        e_uniq_pct = nuniq / max(1, total_numel_uniq) * 100
        print(f"{tag:<4} {desc:<52} "
              f"{b_raw_pct:>9.2f}% {b_uniq_pct:>10.2f}% {e_uniq_pct:>10.2f}%")


def main():
    p = argparse.ArgumentParser(description='OAMP shape coverage audit')
    p.add_argument('--seq_len', type=int, nargs='+', default=[512],
                   help='Sequence length(s) to audit; multiple values run sequentially')
    args = p.parse_args()
    for sl in args.seq_len:
        run_audit(sl)


if __name__ == '__main__':
    main()
