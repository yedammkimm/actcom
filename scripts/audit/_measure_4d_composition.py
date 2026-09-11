"""Measure what the 4-D branch actually holds: Q/K/V head views vs rotary.

pack_stats lumps every 4-D saved tensor into one counter, so the rotary share
has to be measured by shape. Rotary cos/sin arrive as (1, 1, L, head_dim);
head views have shape[1] > 1. Two optimiser steps is enough — the composition
is a property of the architecture, not of training.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import sys, collections, torch
sys.path.insert(0, ROOT)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from oamp.pack_hooks import PackHooks
from oamp.dtype_policy import apply_dtype_policy
from oamp.data import load_task, format_train_prompt

CACHE = '/app/hf_cache'
MID = 'meta-llama/Llama-3.2-3B-Instruct'
tok = AutoTokenizer.from_pretrained(MID, cache_dir=CACHE, local_files_only=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained(MID, cache_dir=CACHE, local_files_only=True,
                                         quantization_config=bnb, low_cpu_mem_usage=True,
                                         device_map={'': 0})
m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                 target_modules=['q_proj','k_proj','v_proj','o_proj'],
                                 task_type='CAUSAL_LM'))
m, param_ptrs, _ = apply_dtype_policy(m, weight_quant='nf4', bf16_rmsnorm=True)
m.train()

tally = collections.Counter()
numel = collections.Counter()
ctx = PackHooks(fp8_ratio=0.0, routing='none', group_size=128, min_numel=1024,
                skip_last_dims={m.config.vocab_size}, param_ptrs=param_ptrs,
                pack_4d_mode='fp8', body_encoding='e2m1')
_orig = ctx.pack
def spy(t):
    if isinstance(t, torch.Tensor) and t.dim() == 4:
        key = 'rotary (shape[1]==1)' if t.shape[1] == 1 else f'head view h={t.shape[1]}'
        tally[key] += 1; numel[key] += t.numel()
    return _orig(t)
ctx.pack = spy

data = load_task('gsm8k', 'train', n_samples=8, seed=42, cache_dir=CACHE, shuffle=True)
opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)
for s in data[:2]:
    ids = tok(format_train_prompt('gsm8k', s), return_tensors='pt',
              truncation=True, max_length=512).input_ids.cuda()
    with ctx:
        out = m(input_ids=ids, labels=ids)
    out.loss.backward(); opt.zero_grad(set_to_none=True)

tot = sum(numel.values())
print('\n=== 4-D saved tensors, 2 optimiser steps ===')
print(f"  {'kind':<26} {'tensors':>9} {'elements':>14} {'share of 4-D':>13}")
for k in sorted(numel, key=lambda k: -numel[k]):
    print(f'  {k:<26} {tally[k]:>9,} {numel[k]:>14,} {numel[k]/tot*100:>12.4f}%')
print(f"  {'TOTAL 4-D':<26} {sum(tally.values()):>9,} {tot:>14,}")
ps = ctx.pack_stats
print(f"\n  pack_stats numel_kept       = {ps['numel_kept']:,}")
print(f"  pack_stats 4-D              = {ps['numel_uniform_fp8_4d']:,}"
      f"  = {ps['numel_uniform_fp8_4d']/ps['numel_kept']*100:.2f}% of kept")
rot = numel.get('rotary (shape[1]==1)', 0)
print(f"\n  rotary / 4-D elements       = {rot/tot*100:.4f}%")
print(f"  rotary / all kept elements  = {rot/ps['numel_kept']*100:.4f}%")
