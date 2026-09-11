"""Two 5-min pre-sweep probes (2026-08-17).

Probe A -- zero-collapse ratio per group_size
    For each gs in {128, 32, 16, 8}, count what fraction of packed elements
    map to x_q == 0 after FP4 quantize on real Qwen activations. Predicts
    gs sweep success: if gs=32 halves the collapse fraction vs gs=128, the
    sweep is promising; if gs=32 stays > 80% collapse, no.

Probe B -- per-layer absmax distribution
    For each transformer layer's attention/MLP saved-activation tensors,
    record the maximum absmax observed. Predicts per-layer K feasibility: if
    only layer 0 is extreme, K_layer0 = 0.8 with K_rest = 0.2 fixes it; if all
    layers are extreme, group_size axis is the correct fix.

Run:
    docker exec -e PROBE_MODEL='Qwen/Qwen2.5-3B-Instruct' hma-container \\
        python oamp_train_engine/audit/probe_zero_collapse_and_absmax.py
"""
from __future__ import annotations
import os, sys, json, math, re
os.environ.setdefault("HF_HOME", "/app/hf_cache")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from oamp.pack_hooks import PackHooks, make_pack_hooks
from oamp.dtype_policy import apply_dtype_policy

MODEL = os.environ.get('PROBE_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
DEV = torch.device('cuda:0')
B, L = 1, 512
_FP4_MAX = 7.0
CLAMP = 1e-8
GS_LIST = [128, 32, 16, 8]

print(f"[load] {MODEL}", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir='/app/hf_cache',
                                         local_files_only=True,
                                         quantization_config=bnb).to(DEV)
m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none')
m = get_peft_model(m, lcfg)
m, param_ptrs, _ = apply_dtype_policy(m, weight_quant='nf4', bf16_rmsnorm=True)

# Fixed fixture (matches diagnose_pack_overflow.py)
g_rng = torch.Generator(device='cpu').manual_seed(0)
vocab = m.config.vocab_size
ids = torch.randint(0, vocab, (B, L), generator=g_rng).to(DEV)
labels = ids.clone()

# ======================================================================
# Probe A: zero-collapse ratio per gs
# ======================================================================
# harmful_zeros = elements whose |x| > absmax/128 (i.e. > 1 FP4 half-step)
#                 but that quantize to 0. These are true information losses.
# benign_zeros  = elements whose |x| <= absmax/128 (already near zero).
collapse = {gs: {'total_elems': 0, 'zero_elems': 0,
                 'harmful_zeros': 0, 'benign_zeros': 0,
                 'total_groups': 0, 'all_zero_groups': 0,
                 'total_absmax_sum': 0.0}
            for gs in GS_LIST}

def simulate_fp4_zero(x_bf16: torch.Tensor, gs: int):
    x = x_bf16.reshape(-1)
    n = x.numel()
    n_padded = ((n + gs - 1) // gs) * gs
    if n_padded > n:
        x = F.pad(x, (0, n_padded - n))
    xg = x.reshape(-1, gs).float()
    raw_absmax = xg.abs().amax(dim=-1, keepdim=True)
    absmax = raw_absmax.clamp(min=CLAMP)
    scale = absmax / _FP4_MAX
    y = (xg / scale).clamp(-_FP4_MAX, _FP4_MAX)
    x_q = y.round()
    zero_mask = (x_q == 0)
    # An element is harmful if it collapses to 0 despite being meaningfully large.
    # Threshold = absmax/128 (roughly 1/18 of the FP4 half-step; anything above this
    # is at least half a bin above zero).
    harmful_thresh = raw_absmax / 128.0
    meaningful_mask = xg.abs() > harmful_thresh
    harmful_mask = zero_mask & meaningful_mask
    benign_mask = zero_mask & ~meaningful_mask
    # tail correction
    tail = n_padded - n
    if tail > 0:
        # padded tail sits at end; its zeros should not count
        flat_zero = int(zero_mask.sum().item()) - tail
        flat_zero = max(0, flat_zero)
    else:
        flat_zero = int(zero_mask.sum().item())
    return {
        'n_real': n,
        'n_zero': flat_zero,
        'n_harmful': int(harmful_mask.sum().item()),
        'n_benign':  int(benign_mask.sum().item()),
        'n_groups': raw_absmax.numel(),
        'n_all_zero_groups': int((raw_absmax == 0).sum().item()),
        'sum_absmax': float(raw_absmax.sum()),
    }

# Instrument pack: on each pack call, record collapse per gs
orig_pack = PackHooks.pack

def wrap_pack(self, tensor):
    if isinstance(tensor, torch.Tensor) and tensor.numel() >= 1024 and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            xf = tensor.detach().to(torch.bfloat16)
            for gs in GS_LIST:
                st = simulate_fp4_zero(xf, gs)
                collapse[gs]['total_elems'] += st['n_real']
                collapse[gs]['zero_elems'] += st['n_zero']
                collapse[gs]['total_groups'] += st['n_groups']
                collapse[gs]['all_zero_groups'] += st['n_all_zero_groups']
                collapse[gs]['total_absmax_sum'] += st['sum_absmax']
                collapse[gs]['harmful_zeros'] += st['n_harmful']
                collapse[gs]['benign_zeros']  += st['n_benign']
        except Exception as e:
            pass
    return orig_pack(self, tensor)

PackHooks.pack = wrap_pack

# ======================================================================
# Probe B: per-layer absmax distribution
# ======================================================================
# Also on each pack call, extract the layer index from param_ptrs role or from
# the layer counter via a module-hook fallback. We use a forward hook on each
# transformer layer to bracket which layer's activations are seen.
per_layer = {}   # {layer_idx: {'max_absmax': float, 'n_tensors_seen': int, 'tensors': []}}
_current_layer = {'idx': None}

def _layer_pre_hook(idx):
    def fn(module, inputs):
        _current_layer['idx'] = idx
    return fn

def _layer_post_hook(idx):
    def fn(module, inputs, output):
        # do not clear; next layer's pre-hook will overwrite
        pass
    return fn

# Register on each layer
layers_module = None
for path in ['model.model.layers', 'base_model.model.model.layers']:
    obj = m
    ok = True
    for p in path.split('.'):
        obj = getattr(obj, p, None)
        if obj is None:
            ok = False; break
    if ok:
        layers_module = obj
        break
assert layers_module is not None, "cannot find transformer layers"
print(f"[probe B] instrumented {len(layers_module)} layers", flush=True)
handles = []
for i, layer in enumerate(layers_module):
    per_layer[i] = {'max_absmax': 0.0,
                    'median_abs_max_over_tensors': 0.0,
                    'dyn_range_max': 0.0,
                    'n_tensors_seen': 0,
                    'total_elems': 0,
                    'harmful_zero_gs128': 0,
                    'harmful_zero_gs32':  0,
                    'roles': {}}
    handles.append(layer.register_forward_pre_hook(_layer_pre_hook(i)))

# v2: also record per-layer median|x|, dyn_range = absmax/median, and per-layer
# harmful-zero attribution for gs=128 vs gs=32.
def wrap_pack_v2(self, tensor):
    if isinstance(tensor, torch.Tensor) and tensor.numel() >= 1024 and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            xf = tensor.detach().to(torch.bfloat16)
            layer_harmful = {128: 0, 32: 0}
            for gs in GS_LIST:
                st = simulate_fp4_zero(xf, gs)
                collapse[gs]['total_elems'] += st['n_real']
                collapse[gs]['zero_elems'] += st['n_zero']
                collapse[gs]['total_groups'] += st['n_groups']
                collapse[gs]['all_zero_groups'] += st['n_all_zero_groups']
                collapse[gs]['total_absmax_sum'] += st['sum_absmax']
                collapse[gs]['harmful_zeros'] += st['n_harmful']
                collapse[gs]['benign_zeros']  += st['n_benign']
                if gs in layer_harmful:
                    layer_harmful[gs] = st['n_harmful']
            lyr = _current_layer['idx']
            if lyr is not None and lyr in per_layer:
                xf_abs = xf.float().abs()
                a = float(xf_abs.max())
                med = float(xf_abs.median())
                dyn = a / max(med, 1e-12)
                pl = per_layer[lyr]
                pl['max_absmax'] = max(pl['max_absmax'], a)
                pl['median_abs_max_over_tensors'] = max(pl['median_abs_max_over_tensors'], med)
                pl['dyn_range_max'] = max(pl['dyn_range_max'], dyn)
                pl['n_tensors_seen'] += 1
                pl['total_elems'] += xf.numel()
                pl['harmful_zero_gs128'] += layer_harmful[128]
                pl['harmful_zero_gs32']  += layer_harmful[32]
                shape_key = f"{tuple(tensor.shape)}"
                pl['roles'][shape_key] = pl['roles'].get(shape_key, 0) + 1
        except Exception:
            pass
    return orig_pack(self, tensor)

PackHooks.pack = wrap_pack_v2

# Run a single forward+backward with uniform_fp8 hooks (pack fires for every tensor)
hooks = make_pack_hooks('uniform_fp8', fp8_ratio=1.0, group_size=128,
                        min_numel=1024, dedupe=False, stochastic_rounding=False,
                        pack_4d_mode='fp8', param_ptrs=param_ptrs,
                        skip_last_dims={m.config.vocab_size})

m.train()
print("[run] forward+backward once ...", flush=True)
with hooks:
    out = m(input_ids=ids, labels=labels)
    loss = out.loss
    loss.backward()

for h in handles:
    h.remove()

# ======================================================================
# Report
# ======================================================================
print()
print("="*72)
print("PROBE A: zero-collapse fraction per gs (harmful = collapsed & |x| > absmax/128)")
print("="*72)
print(f"  {'gs':>5}  {'total':>12}  {'zero':>12}  {'zero%':>7}  {'harmful':>12}  {'harm%':>7}  {'benign':>12}  {'avg_absmax':>10}")
for gs in GS_LIST:
    c = collapse[gs]
    zpct = 100 * c['zero_elems']    / max(1, c['total_elems'])
    hpct = 100 * c['harmful_zeros'] / max(1, c['total_elems'])
    avg_abs = c['total_absmax_sum'] / max(1, c['total_groups'])
    print(f"  {gs:>5}  {c['total_elems']:>12,}  {c['zero_elems']:>12,}  {zpct:>6.2f}%  {c['harmful_zeros']:>12,}  {hpct:>6.2f}%  {c['benign_zeros']:>12,}  {avg_abs:>10.3f}")

print()
print("="*72)
print("PROBE B: per-layer absmax + dynamic range + harmful zeros")
print("="*72)
sorted_layers = sorted(per_layer.items(), key=lambda kv: -kv[1]['dyn_range_max'])
print(f"  Layer count: {len(per_layer)}")
print(f"  Sorted by dyn_range_max = max(absmax / median|x|) across tensors seen at that layer.")
print(f"  {'layer':>5}  {'absmax':>10}  {'med|x|':>10}  {'dyn_range':>10}  {'harm_gs128':>10}  {'harm_gs32':>10}  {'elems':>10}")
for lidx, info in sorted_layers[:15]:
    dyn = info['dyn_range_max']
    total = info['total_elems']
    hp128 = 100 * info['harmful_zero_gs128'] / max(1, total)
    hp32  = 100 * info['harmful_zero_gs32']  / max(1, total)
    print(f"  {lidx:>5}  {info['max_absmax']:>10.3f}  {info['median_abs_max_over_tensors']:>10.4f}  {dyn:>10.1f}  {hp128:>9.2f}%  {hp32:>9.2f}%  {total:>10,}")
print("  ...")
for lidx, info in sorted_layers[-5:]:
    dyn = info['dyn_range_max']
    total = info['total_elems']
    hp128 = 100 * info['harmful_zero_gs128'] / max(1, total)
    hp32  = 100 * info['harmful_zero_gs32']  / max(1, total)
    print(f"  {lidx:>5}  {info['max_absmax']:>10.3f}  {info['median_abs_max_over_tensors']:>10.4f}  {dyn:>10.1f}  {hp128:>9.2f}%  {hp32:>9.2f}%  {total:>10,}")

# Distribution buckets on dyn_range
buckets = [10, 100, 1000, 10000, 100000]
print(f"\n  dyn_range distribution:")
for th in buckets:
    n = sum(1 for _, i in per_layer.items() if i['dyn_range_max'] > th)
    print(f"    layers with dyn_range > {th:>6}: {n}/{len(per_layer)}")

# Save
ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
outp = f"/app/HMA_Project/results/probe_zero_collapse_absmax_{ts}.json"
with open(outp, 'w') as f:
    json.dump({
        'model': MODEL,
        'B': B, 'L': L,
        'collapse_by_gs': collapse,
        'per_layer_absmax': per_layer,
    }, f, indent=2, default=str)
print(f"\n[done] saved to {outp}")
