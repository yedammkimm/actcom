"""Does any tensor of rank greater than four survive the filters?

Equation (2) states the dispatch as rank >= 4 while the implementation tests
rank == 4. The two coincide only if no higher-rank tensor reaches the branch,
and pack_stats cannot answer that: a rank-5 tensor would fall through to the
3-D path and be counted as uniform_fp4_3d, silently.

This tallies the rank of every tensor that PASSES all five filters, i.e. at the
point where the dispatch would see it. Skipped tensors are irrelevant to the
claim, so they are excluded by replaying the filter chain before counting.

Two optimiser steps suffice: the set of saved shapes is a property of the
architecture, not of training.
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

rank_tally = collections.Counter()
rank_numel = collections.Counter()
shapes = collections.defaultdict(collections.Counter)
ctx = PackHooks(fp8_ratio=0.0, routing='none', group_size=128, min_numel=1024,
                skip_last_dims={m.config.vocab_size}, param_ptrs=param_ptrs,
                pack_4d_mode='fp8', body_encoding='e2m1')
_orig = ctx.pack
def survives(t):
    """Replay pack()'s filter chain; True if the tensor reaches the dispatch."""
    if ctx._is_param(t):                                     return False
    if t.numel() < ctx.min_numel:                            return False
    if not t.is_floating_point():                            return False
    if t.dim() > 0 and t.shape[-1] in ctx.skip_last_dims:     return False
    if t.dim() > 0 and t.shape[-1] % ctx.group_size != 0:     return False
    return True

def spy(t):
    if isinstance(t, torch.Tensor) and survives(t):
        rank_tally[t.dim()] += 1
        rank_numel[t.dim()] += t.numel()
        shapes[t.dim()][tuple(t.shape)] += 1
    return _orig(t)
ctx.pack = spy

data = load_task('gsm8k', 'train', n_samples=8, seed=42, cache_dir=CACHE, shuffle=True)
opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)

for i in range(2):
    ids = tok(format_train_prompt('gsm8k', data[i]), return_tensors='pt',
              truncation=True, max_length=512).input_ids.cuda()
    with ctx:
        out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    opt.zero_grad(set_to_none=True)

tot = sum(rank_tally.values())
print(f"\n{'='*64}\ntensors passing all five filters: {tot:,}\n{'='*64}")
for r in sorted(rank_tally):
    print(f"  rank {r}: {rank_tally[r]:>7,} tensors  {rank_numel[r]:>14,} elements  "
          f"({rank_tally[r]/tot*100:5.1f}% of tensors)")
hi = [r for r in rank_tally if r > 4]
print(f"\n  rank > 4: {'NONE — equation (2) and the implementation coincide' if not hi else hi}")
print(f"\n  distinct shapes per rank:")
for r in sorted(shapes):
    top = shapes[r].most_common(4)
    print(f"    rank {r}: {len(shapes[r])} distinct; " +
          ", ".join(f"{s}x{n}" for s, n in top))
