#!/usr/bin/env python3
"""Task 2: torchao GB10 4bit weight-only sanity — forward + LoRA backward.

Run inside hma-container:
  docker exec hma-container python /app/HMA_Project/oamp_train_engine/benchmarks/sanity_torchao_int4.py
"""
import os, sys, time, traceback

os.environ.setdefault('HF_HOME', '/app/hf_cache')
import torch

print(f"torch: {torch.__version__}")
try:
    import torchao
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    print(f"torchao: {torchao.__version__}")
except Exception as e:
    print(f"torchao import FAIL: {e}"); sys.exit(1)

# Available packing formats (some require external kernels like mslk).
try:
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    _packing_formats = list(Int4PackingFormat)
    print(f"int4 packing formats: {[f.value for f in _packing_formats]}")
except Exception:
    _packing_formats = None

DEV = 'cuda'
MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
CACHE = '/app/hf_cache'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

t0 = time.time()
print(f"\n[1] Loading {MODEL} (bf16) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, cache_dir=CACHE, local_files_only=True,
    torch_dtype=torch.bfloat16).to(DEV)
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True)
print(f"    loaded in {time.time()-t0:.1f}s")

torch.cuda.reset_peak_memory_stats()
mem_bf16 = torch.cuda.max_memory_allocated() / 1e9
print(f"    peak VRAM (bf16 load): {mem_bf16:.2f} GB")

# ── Forward-only sanity on bf16 first ──────────────────────────────────
ids = torch.randint(0, tok.vocab_size, (1, 128), device=DEV)
with torch.no_grad():
    out = model(ids)
print(f"[2] BF16 forward OK: logits={tuple(out.logits.shape)}")

# ── Apply int4 weight-only ─────────────────────────────────────────────
print("\n[3] Applying int4_weight_only ...")
torch.cuda.reset_peak_memory_stats()

# Try formats in order of GB10 compatibility; fall back through them.
_candidates = []
if _packing_formats is not None:
    for fmt in _packing_formats:
        _candidates.append(('v2', {'int4_packing_format': fmt}))
_candidates.append(('v1', {'version': 1}))

_quantized = False
for tag, kw in _candidates:
    try:
        # Reload weights first (a failed quantize_ leaves the model partially modified)
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, cache_dir=CACHE, local_files_only=True,
            torch_dtype=torch.bfloat16).to(DEV)
        t0 = time.time()
        quantize_(model, Int4WeightOnlyConfig(**kw))
        print(f"    quantized in {time.time()-t0:.1f}s using {tag}/{kw}")
        _quantized = True
        break
    except Exception as e:
        print(f"    {tag}/{kw} FAIL: {type(e).__name__}: {e}")
        continue

if not _quantized:
    print("[3] all int4 configs failed"); sys.exit(2)

mem_int4 = torch.cuda.max_memory_allocated() / 1e9
print(f"    peak VRAM after int4: {mem_int4:.2f} GB")

# ── Forward with int4 ──────────────────────────────────────────────────
try:
    with torch.no_grad():
        out = model(ids)
    print(f"[4] int4 forward OK: logits={tuple(out.logits.shape)} dtype={out.logits.dtype}")
except Exception as e:
    traceback.print_exc(); print(f"[4] int4 forward FAIL: {e}"); sys.exit(3)

# ── LoRA + backward ────────────────────────────────────────────────────
print("\n[5] Adding LoRA (r=16) + backward ...")
try:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none',
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    out = peft_model(input_ids=ids, labels=ids)
    print(f"    loss = {out.loss.item():.4f}")
    out.loss.backward()
    mem_peft = torch.cuda.max_memory_allocated() / 1e9
    print(f"[5] backward OK, peak VRAM (int4 + LoRA fwd+bwd): {mem_peft:.2f} GB")
except Exception as e:
    traceback.print_exc(); print(f"[5] LoRA backward FAIL: {e}"); sys.exit(4)

# ── Optimizer step ─────────────────────────────────────────────────────
try:
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-4)
    opt.step(); opt.zero_grad()
    print("[6] Optimizer step OK")
except Exception as e:
    traceback.print_exc(); print(f"[6] Optimizer step FAIL: {e}"); sys.exit(5)

print("\nALL SANITY CHECKS PASS — Blackwell/GB10 int4 weight-only training path is viable.")
print(f"Summary: VRAM bf16-load={mem_bf16:.2f}GB, int4-load={mem_int4:.2f}GB, int4+LoRA-train={mem_peft:.2f}GB")
