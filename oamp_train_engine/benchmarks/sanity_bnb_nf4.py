#!/usr/bin/env python3
"""bitsandbytes NF4 sanity on GB10 (Blackwell SM 12.1).

Answers ONE question: does NF4 4bit weight + LoRA backward work on GB10?
Run inside hma-container:
  docker exec hma-container python /app/HMA_Project/oamp_train_engine/benchmarks/sanity_bnb_nf4.py
"""
import os, sys, traceback
os.environ.setdefault('HF_HOME', '/app/hf_cache')

import torch
try:
    import bitsandbytes as bnb
except ImportError as e:
    print(f"bnb import FAIL: {e}"); sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

print(f"bnb: {bnb.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}, SM {torch.cuda.get_device_capability(0)}")
free0, total0 = torch.cuda.mem_get_info()
print(f"VRAM before: free={free0/1e9:.1f} GB / total={total0/1e9:.1f} GB\n")

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CACHE = "/app/hf_cache"

# 1. NF4 config
print("[1] Building NF4 config (double_quant, bf16 compute) ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load 3B with NF4
print(f"[2] Loading {MODEL} with NF4 ...")
torch.cuda.reset_peak_memory_stats()
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        cache_dir=CACHE,
        local_files_only=True,
    )
except Exception as e:
    traceback.print_exc(); print(f"[2] FAIL: {e}"); sys.exit(2)
peak_load = torch.cuda.max_memory_allocated() / 1e9
print(f"    peak VRAM after load: {peak_load:.2f} GB (bf16 baseline was 6.43 GB)")

# 3. Check first q_proj is actually 4bit
first_linear = None
for n, m in model.named_modules():
    if 'q_proj' in n and hasattr(m, 'weight'):
        first_linear = (n, m); break
name, mod = first_linear
print(f"\n[3] First q_proj: {name}")
print(f"    module class: {type(mod).__name__}")
print(f"    weight class: {type(mod.weight).__name__}")
print(f"    weight dtype: {mod.weight.dtype}")
if hasattr(mod.weight, 'quant_state'):
    qs = mod.weight.quant_state
    print(f"    quant_type: {getattr(qs, 'quant_type', 'n/a')}")
    print(f"    blocksize: {getattr(qs, 'blocksize', 'n/a')}")
is_4bit = 'Params4bit' in type(mod.weight).__name__ or 'Linear4bit' in type(mod).__name__
print(f"    IS 4bit: {is_4bit}")
if not is_4bit:
    print("[3] FAIL: q_proj not quantized to 4bit"); sys.exit(3)

# 4. k-bit prep + LoRA
print("\n[4] prepare_model_for_kbit_training + LoRA (r=16) ...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05, bias='none',
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 5. Forward + Backward
print("\n[5] Forward + Backward ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True)
    inputs = tokenizer("What is 2+2? Let's think step by step.", return_tensors="pt").to("cuda")
    inputs["labels"] = inputs["input_ids"]

    torch.cuda.reset_peak_memory_stats()
    out = peft_model(**inputs)
    loss = out.loss
    print(f"    forward OK. loss={loss.item():.4f} (expected 2-10)")
    if not torch.isfinite(loss):
        print("[5] FAIL: loss non-finite"); sys.exit(5)

    loss.backward()
    print("    backward OK.")
    peak_bwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"    peak VRAM (fwd+bwd): {peak_bwd:.2f} GB")
except Exception as e:
    traceback.print_exc(); print(f"[5] FAIL: {e}"); sys.exit(5)

# 6. LoRA gradient coverage
print("\n[6] LoRA gradient coverage ...")
have_grad = total = 0
zero_grad_names = []
for n, p in peft_model.named_parameters():
    if 'lora' in n.lower():
        total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            have_grad += 1
        else:
            zero_grad_names.append(n)
print(f"    LoRA params w/ nonzero grad: {have_grad}/{total}")
if zero_grad_names[:3]:
    print(f"    zero-grad examples: {zero_grad_names[:3]}")
if have_grad < total:
    print(f"[6] WARN: {total-have_grad} LoRA params have zero grad")
else:
    print("    All LoRA params receive gradient — training path viable.")

# 7. Optimizer step (paged AdamW 8bit — QLoRA reference)
print("\n[7] Paged 8bit AdamW step ...")
try:
    opt = bnb.optim.PagedAdamW8bit(
        [p for p in peft_model.parameters() if p.requires_grad], lr=1e-4)
    opt.step(); opt.zero_grad()
    print("    optimizer step OK.")
except Exception as e:
    traceback.print_exc(); print(f"[7] FAIL: {e}"); sys.exit(7)

# 8. Summary
free1, _ = torch.cuda.mem_get_info()
print("\n" + "=" * 60)
print("SUMMARY (all steps 1-7 passed)")
print(f"  VRAM peak (fwd+bwd): {peak_bwd:.2f} GB")
print(f"  VRAM free after:     {free1/1e9:.1f} GB")
print(f"  bnb NF4 + LoRA backward + paged 8bit AdamW: WORKS on GB10 SM 12.1")
print("=" * 60)
