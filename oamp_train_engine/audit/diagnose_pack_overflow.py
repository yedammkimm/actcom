"""Diagnose why Qwen gradient probe shows inf/1e13 norm_test on FP8 filter.

Hypothesis: _quantize_fp8's clamp(min=1e-8) creates scale=2.23e-11 when a group
has absmax=0 (or below clamp). Any noise-level nonzero in that group then
becomes >_FP8_MAX during (x/scale) → saturates → dequant returns inf →
gradient corrupts.

Also checks:
  1. Are LoRA modules fp32 (would explain Qwen vs Llama differently)?
  2. Do saved activations contain exact zeros / all-zero groups?
  3. What fraction of groups hit the clamp?
  4. Do dequant outputs contain nonfinite values?

Run:
  docker exec -e DIAG_MODEL='Qwen/Qwen2.5-3B-Instruct' hma-container \\
      python oamp_train_engine/audit/diagnose_pack_overflow.py
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("HF_HOME", "/app/hf_cache")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from oamp.pack_hooks import PackHooks, make_pack_hooks
from oamp.dtype_policy import apply_dtype_policy

MODEL = os.environ.get('DIAG_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')
DEV = torch.device('cuda:0')
B, L = 1, 512

print(f"[load] {MODEL}")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir='/app/hf_cache',
                                         local_files_only=True,
                                         quantization_config=bnb).to(DEV)
m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                  lora_dropout=0.0,
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  bias='none')
m = get_peft_model(m, lcfg)
m, param_ptrs, dtype_report = apply_dtype_policy(m, weight_quant='nf4', bf16_rmsnorm=True)
print("[dtype_report]", json.dumps({k: v for k, v in dtype_report.items() if k != 'norms'}, indent=2, default=str)[:2000])

# Check LoRA dtypes explicitly
print("\n[LoRA parameter dtypes]")
lora_dtypes = {}
for n, p in m.named_parameters():
    if 'lora_' in n:
        key = 'lora_A' if 'lora_A' in n else 'lora_B'
        lora_dtypes.setdefault(key, {}).setdefault(str(p.dtype), 0)
        lora_dtypes[key][str(p.dtype)] += 1
print(json.dumps(lora_dtypes, indent=2))

# Instrument PackHooks pack methods
orig_fp8 = PackHooks._quantize_fp8
orig_fp4 = PackHooks._quantize_fp4_groups
orig_dq_fp4 = PackHooks._dequantize_fp4

stats = {'fp8': {'calls': 0, 'clamped_groups': 0, 'total_groups': 0,
                 'nonfinite_out': 0, 'min_scale': float('inf'), 'max_scale': 0.0,
                 'max_input_absmax': 0.0, 'nonfinite_input': 0,
                 'scale_fp16_underflow_groups': 0,
                 'scale_fp16_underflow_absmax_max': 0.0},
         'fp4': {'calls': 0, 'clamped_groups': 0, 'total_groups': 0,
                 'nonfinite_out': 0, 'min_scale': float('inf'), 'max_scale': 0.0,
                 'max_input_absmax': 0.0, 'nonfinite_input': 0,
                 'scale_fp16_underflow_groups': 0,
                 'scale_fp16_underflow_absmax_max': 0.0}}
# Also track worst-offender per method
worst = {'fp8': None, 'fp4': None}

_FP8_MAX = 448.0
_FP4_MAX = 7.0
CLAMP = 1e-8

def wrap_fp8(self, tensor):
    gs = self.group_size
    x = tensor.reshape(-1)
    n = x.numel()
    n_padded = ((n + gs - 1) // gs) * gs
    if n_padded > n:
        x = F.pad(x, (0, n_padded - n))
    x = x.reshape(-1, gs)
    raw_absmax = x.abs().amax(dim=-1, keepdim=True).float()
    clamped = (raw_absmax < CLAMP).sum().item()
    absmax = raw_absmax.clamp(min=CLAMP)
    scale = absmax / _FP8_MAX
    x_over = (x.float() / scale)
    # Cast path
    result = x_over.to(torch.bfloat16).to(torch.float8_e4m3fn)
    # Dequant check: manual reverse
    dequant = result.to(torch.bfloat16).float() * scale
    nonfin_out = (~torch.isfinite(dequant)).sum().item()
    nonfin_in = (~torch.isfinite(x)).sum().item()

    s = stats['fp8']
    s['calls'] += 1
    s['clamped_groups'] += clamped
    s['total_groups'] += raw_absmax.numel()
    s['nonfinite_out'] += nonfin_out
    s['nonfinite_input'] += nonfin_in
    s['min_scale'] = min(s['min_scale'], float(scale.min()))
    s['max_scale'] = max(s['max_scale'], float(scale.max()))
    s['max_input_absmax'] = max(s['max_input_absmax'], float(raw_absmax.max()))
    # storage-cast underflow: scale.to(fp16) == 0 destroys the group at dequant
    scale_fp16 = scale.to(torch.float16)
    underflow_mask = (scale_fp16 == 0)
    n_uf = int(underflow_mask.sum().item())
    s['scale_fp16_underflow_groups'] += n_uf
    if n_uf > 0:
        s['scale_fp16_underflow_absmax_max'] = max(
            s['scale_fp16_underflow_absmax_max'],
            float(raw_absmax[underflow_mask].max()))
    if nonfin_out > 0 and worst['fp8'] is None:
        worst['fp8'] = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'nonfin_out': nonfin_out,
            'raw_absmax_min': float(raw_absmax.min()),
            'raw_absmax_max': float(raw_absmax.max()),
            'scale_min': float(scale.min()),
            'clamped_groups': clamped,
            'total_groups': raw_absmax.numel(),
            'x_over_max': float(x_over.abs().max()),
            'sample_bad_group_absmax': float(raw_absmax[(dequant != dequant).any(dim=-1) if False else (dequant.abs() == float('inf')).any(dim=-1)][:5].flatten().mean()) if nonfin_out > 0 else None,
        }
    # Do the original quantization for correct return signature
    x_scaled = x_over.to(torch.bfloat16).to(torch.float8_e4m3fn)
    return x_scaled, scale.to(torch.float16), n

def wrap_fp4(self, x):
    gs = self.group_size
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    n_padded = ((n + gs - 1) // gs) * gs
    if n_padded > n:
        x_flat = F.pad(x_flat, (0, n_padded - n))
    x_groups = x_flat.reshape(-1, gs)
    raw_absmax = x_groups.abs().amax(dim=-1, keepdim=True)
    clamped = (raw_absmax < CLAMP).sum().item()
    absmax = raw_absmax.clamp(min=CLAMP)
    scale = absmax / _FP4_MAX
    y = (x_groups / scale)
    nonfin_out = (~torch.isfinite(y)).sum().item()
    nonfin_in = (~torch.isfinite(x_groups)).sum().item()

    s = stats['fp4']
    s['calls'] += 1
    s['clamped_groups'] += clamped
    s['total_groups'] += raw_absmax.numel()
    s['nonfinite_out'] += nonfin_out
    s['nonfinite_input'] += nonfin_in
    s['min_scale'] = min(s['min_scale'], float(scale.min()))
    s['max_scale'] = max(s['max_scale'], float(scale.max()))
    s['max_input_absmax'] = max(s['max_input_absmax'], float(raw_absmax.max()))
    # storage-cast underflow: scale.to(fp16) == 0 destroys the group at dequant
    scale_fp16 = scale.to(torch.float16)
    underflow_mask = (scale_fp16 == 0)
    n_uf = int(underflow_mask.sum().item())
    s['scale_fp16_underflow_groups'] += n_uf
    if n_uf > 0:
        s['scale_fp16_underflow_absmax_max'] = max(
            s['scale_fp16_underflow_absmax_max'],
            float(raw_absmax[underflow_mask].max()))
    if nonfin_out > 0 and worst['fp4'] is None:
        worst['fp4'] = {
            'shape': tuple(x.shape),
            'nonfin_out': nonfin_out,
            'raw_absmax_min': float(raw_absmax.min()),
            'raw_absmax_max': float(raw_absmax.max()),
            'clamped_groups': clamped,
            'total_groups': raw_absmax.numel(),
            'y_max': float(y.abs().max()),
        }
    return orig_fp4(self, x)

PackHooks._quantize_fp8 = wrap_fp8
PackHooks._quantize_fp4_groups = wrap_fp4

# Instrument FP4 dequant to catch the inf's origin
dq_stats = {'fp4_dequant': {'calls': 0, 'nonfinite_scale_in': 0, 'nonfinite_out': 0,
                            'max_scale_in': 0.0, 'max_dequant_absmax': 0.0,
                            'worst_shape': None}}

def wrap_dq_fp4(self, packed, scale, original_n, dtype):
    result = orig_dq_fp4(self, packed, scale, original_n, dtype)
    s = dq_stats['fp4_dequant']
    s['calls'] += 1
    nfin_scale = (~torch.isfinite(scale)).sum().item()
    nfin_out = (~torch.isfinite(result)).sum().item()
    s['nonfinite_scale_in'] += nfin_scale
    s['nonfinite_out'] += nfin_out
    try:
        s['max_scale_in'] = max(s['max_scale_in'], float(scale[torch.isfinite(scale)].abs().max() if torch.isfinite(scale).any() else 0))
    except Exception:
        pass
    try:
        finite_result = result[torch.isfinite(result)]
        if finite_result.numel() > 0:
            s['max_dequant_absmax'] = max(s['max_dequant_absmax'], float(finite_result.abs().max()))
    except Exception:
        pass
    if (nfin_out > 0 or nfin_scale > 0) and s['worst_shape'] is None:
        s['worst_shape'] = {'orig_n': original_n, 'dtype': str(dtype),
                            'scale_shape': tuple(scale.shape),
                            'scale_dtype': str(scale.dtype),
                            'nfin_scale': nfin_scale, 'nfin_out': nfin_out}
    return result

PackHooks._dequantize_fp4 = wrap_dq_fp4

# Build the same fixture as gradient_error_probe
g = torch.Generator(device='cpu').manual_seed(0)
vocab = m.config.vocab_size
ids = torch.randint(0, vocab, (B, L), generator=g).to(DEV)
labels = ids.clone()

# Compare two configurations:
#   (a) uniform_fp8: all FP8 — sanity check (should be clean)
#   (b) naive_fp4 with pack_4d_mode='fp8': matches actual failing Qwen setup
CONFIG = os.environ.get('DIAG_CONFIG', 'naive_fp4')  # 'uniform_fp8' | 'naive_fp4'
print(f"\n[config] pack config: {CONFIG}")
if CONFIG == 'uniform_fp8':
    hooks = make_pack_hooks('uniform_fp8', fp8_ratio=1.0, group_size=128,
                            min_numel=1024, dedupe=False, stochastic_rounding=False,
                            pack_4d_mode='fp8', param_ptrs=param_ptrs,
                            skip_last_dims={m.config.vocab_size})
else:
    hooks = make_pack_hooks('naive_fp4', fp8_ratio=0.0, group_size=128,
                            min_numel=1024, dedupe=False, stochastic_rounding=False,
                            pack_4d_mode='fp8', param_ptrs=param_ptrs,
                            skip_last_dims={m.config.vocab_size})

m.train()
for p in m.parameters():
    if p.grad is not None:
        p.grad.detach_(); p.grad.zero_()

with hooks:
    out = m(input_ids=ids, labels=labels)
    loss = out.loss
    print(f"\n[forward] loss = {loss.item():.4f}")
    loss.backward()

# Check gradient finiteness
nonfin_grads = {}
total_finite = 0
total_nonfinite = 0
for n, p in m.named_parameters():
    if p.grad is not None:
        nf = (~torch.isfinite(p.grad)).sum().item()
        total_nonfinite += nf
        total_finite += p.grad.numel() - nf
        if nf > 0:
            nonfin_grads[n] = nf
print(f"\n[gradient] total elements: {total_finite + total_nonfinite}")
print(f"[gradient] nonfinite: {total_nonfinite} ({100*total_nonfinite/(total_finite+total_nonfinite):.4f}%)")
print(f"[gradient] params with nonfinite grad: {len(nonfin_grads)}")
if nonfin_grads:
    for name, cnt in list(nonfin_grads.items())[:10]:
        print(f"  {name}: {cnt} nonfinite")

print("\n[pack stats]")
print(json.dumps(stats, indent=2))
print("\n[dequant stats]")
print(json.dumps(dq_stats, indent=2, default=str))
print("\n[worst offender]")
print(json.dumps(worst, indent=2, default=str))

# Save
ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
outp = f'/app/HMA_Project/results/diagnose_pack_overflow_{ts}.json'
with open(outp, 'w') as f:
    json.dump({'model': MODEL, 'stats': stats, 'worst': worst,
               'gradient': {'total': total_finite+total_nonfinite,
                            'nonfinite': total_nonfinite,
                            'nonfinite_params': len(nonfin_grads),
                            'sample_nonfin_params': list(nonfin_grads.items())[:20]}}, f, indent=2, default=str)
print(f"\n[done] saved to {outp}")
