"""Verify oamp.dtype_policy against spec v1 §3.

Gates:
  D1  RMSNorm swap count == 57 on Llama-3.2-3B (28 layers × 2 norms + final).
  D2  dtype_report after Arm 4: norms / embed / lm_head / lora all bf16-only.
  D3  BF16 base path also applies the swap (§3.4).
  D4  param_ptrs collected AFTER cast, so pack filters won't miss on LoRA.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/app/hf_cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from oamp.dtype_policy import apply_dtype_policy, BF16RMSNorm

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "/app/hf_cache"
DEVICE = torch.device("cuda:0")


def build(weight_quant: str):
    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      bias='none')
    if weight_quant == 'nf4':
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
            quantization_config=bnb).to(DEVICE)
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
            torch_dtype=torch.bfloat16).to(DEVICE)
    m = get_peft_model(m, lcfg)
    return m


def count_rmsnorm_children(model):
    """How many *RMSNorm* modules currently sit in the tree (any subclass)."""
    n_bf16 = 0
    n_other_rmsnorm = 0
    for _, parent in model.named_modules():
        for _, child in parent.named_children():
            cls = type(child).__name__
            if cls == 'BF16RMSNorm':
                n_bf16 += 1
            elif 'RMSNorm' in cls:
                n_other_rmsnorm += 1
    return n_bf16, n_other_rmsnorm


def run(weight_quant: str):
    print(f"\n{'='*60}\n[{weight_quant.upper()}] build + apply_dtype_policy\n{'='*60}")
    m = build(weight_quant)

    # Baseline: how many pre-swap RMSNorms exist?
    bf16_before, other_before = count_rmsnorm_children(m)
    print(f"  before swap: BF16RMSNorm={bf16_before}, other *RMSNorm*={other_before}")

    m, ptrs, report = apply_dtype_policy(m, weight_quant=weight_quant, bf16_rmsnorm=True)

    bf16_after, other_after = count_rmsnorm_children(m)
    print(f"  after  swap: BF16RMSNorm={bf16_after}, other *RMSNorm*={other_after}")
    print(f"  dtype_report: {report}")
    print(f"  param_ptrs: {len(ptrs)} unique storages")

    # D1: swap count
    assert report['rmsnorm_swapped'] == 57, \
        f"D1 FAIL ({weight_quant}): expected 57 swaps, got {report['rmsnorm_swapped']}"
    assert bf16_after == 57 and other_after == 0, \
        f"D1 FAIL ({weight_quant}): counts after swap wrong (bf16={bf16_after}, other={other_after})"

    # D2: dtype report — every category bf16-only
    for cat in ('norms', 'embed', 'lm_head', 'lora'):
        dtypes = report[cat]
        assert dtypes == ['torch.bfloat16'], \
            f"D2 FAIL ({weight_quant}): {cat} = {dtypes}, expected ['torch.bfloat16']"

    # D4: param_ptrs collected AFTER cast — LoRA storage pointers reflect bf16 params
    for name, p in m.named_parameters():
        if 'lora_' in name.lower() and p.requires_grad:
            try:
                sp = p.untyped_storage().data_ptr()
            except Exception:
                sp = p.data_ptr()
            assert sp in ptrs, f"D4 FAIL ({weight_quant}): LoRA param {name!r} not in param_ptrs"
            break  # one is enough

    print(f"  [{weight_quant}] PASS (D1 swap=57, D2 all-bf16, D4 param_ptrs post-cast)")
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("dtype_policy verification (spec v1 §3)")

    # D3: cover both weight_quant paths; both must run the swap.
    run('bf16')
    run('nf4')

    print("\nALL CHECKS PASSED")
