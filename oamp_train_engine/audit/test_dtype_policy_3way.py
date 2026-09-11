"""3-way loss curve + peak VRAM check for dtype policy decision.

  Arm 1: fp32 everything (default prepare_model_for_kbit_training, LoRA inherits fp32)
  Arm 2: restore_bf16 (norms + lm_head + embed → bf16) but LoRA stays fp32
  Arm 3: restore_bf16 + LoRA also cast to bf16 (full unification)
  Arm 4: with BF16RMSNorm

Runs 100 identical training steps (seed=42, 3B NF4, synthetic random token stream
so results are deterministic across arms). Records per-step CE loss, checks for
NaN/Inf, and reports max_memory_allocated after training completes.

Judgment: |Δloss| < 5e-3 elementwise vs Arm 1 → PASS.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

CACHE_DIR = "/app/hf_cache"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 512


def format_training_prompt(q, a):
    return f"Question: {q}\nLet's solve this step by step.\n{a}"


def load_gsm8k(n_samples, seed):
    import random
    ds = load_dataset('gsm8k', 'main', cache_dir=CACHE_DIR)
    data = []
    for item in ds['train']:
        data.append({'question': item['question'], 'answer_text': item['answer']})
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:n_samples]


def build_data_order(n_samples, seed, epochs):
    rng = np.random.RandomState(seed)
    order = []
    for _ in range(epochs):
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        order.append(idx.tolist())
    return order


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(dtype_policy, seed):
    """dtype_policy in {'fp32', 'restore_bf16_lora_fp32', 'restore_bf16_lora_bf16',
                        'arm4_bf16_rmsnorm'}.

    Order matters to keep initial parameter values identical across arms:
      1. load NF4 base
      2. prepare_model_for_kbit_training  (base -> fp32)
      3. get_peft_model                    (LoRA A/B initialized in fp32 for ALL arms)
      4. Optionally cast norms/embed/lm_head/LoRA to bf16 afterwards.
      5. Arm 4 additionally swaps LlamaRMSNorm with a bf16-only variant that
         skips the reference implementation's float32 cast (CompAct-style).
    """
    set_seed(seed)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-3B-Instruct', quantization_config=bnb,
        device_map={"": 0}, cache_dir=CACHE_DIR, local_files_only=True,
    )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=LORA_TARGETS, bias='none',
    )
    peft_model = get_peft_model(m, lora)
    peft_model.train()

    if dtype_policy != 'fp32':
        for name, module in peft_model.named_modules():
            cls_name = type(module).__name__
            if 'RMSNorm' in cls_name or 'LayerNorm' in cls_name:
                for p in module.parameters(recurse=False):
                    p.data = p.data.to(torch.bfloat16)
            if isinstance(module, nn.Embedding):
                for p in module.parameters(recurse=False):
                    p.data = p.data.to(torch.bfloat16)
        for name, module in peft_model.named_modules():
            if name.endswith('lm_head') and isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)

    if dtype_policy in ('restore_bf16_lora_bf16', 'arm4_bf16_rmsnorm'):
        for n, p in peft_model.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                p.data = p.data.to(torch.bfloat16)

    if dtype_policy == 'arm4_bf16_rmsnorm':
        _swap_rmsnorm_for_bf16(peft_model)

    if hasattr(peft_model, 'gradient_checkpointing_disable'):
        peft_model.gradient_checkpointing_disable()

    return peft_model


class BF16RMSNorm(nn.Module):
    """RMSNorm computed in the source dtype -- no fp32 upcast for variance.

    Reuses the original weight tensor so parameter identity is preserved.
    """
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def _swap_rmsnorm_for_bf16(model):
    """Replace every LlamaRMSNorm subtree with BF16RMSNorm using the same weight."""
    swapped = 0
    for parent_name, parent in model.named_modules():
        for attr, child in list(parent.named_children()):
            if type(child).__name__ == 'LlamaRMSNorm':
                new = BF16RMSNorm(child.weight, child.variance_epsilon).to(child.weight.device)
                setattr(parent, attr, new)
                swapped += 1
    print(f"  [arm4] swapped {swapped} LlamaRMSNorm -> BF16RMSNorm", flush=True)


def train_arm(arm_name, dtype_policy, steps, seed, lr, data_source, train_data, data_order, tokenizer):
    print(f"\n{'='*70}\n[{arm_name}] dtype_policy={dtype_policy} data_source={data_source}\n{'='*70}", flush=True)
    peft_model = build_model(dtype_policy, seed)

    # Report LoRA / norm / lm_head dtypes for quick sanity.
    lora_dtypes = set()
    norm_dtypes = set()
    for n, p in peft_model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            lora_dtypes.add(str(p.dtype))
        elif 'norm' in n.lower():
            norm_dtypes.add(str(p.dtype))
    lm_head_p = None
    for n, p in peft_model.named_parameters():
        if 'lm_head' in n:
            lm_head_p = p; break
    print(f"  lora_dtypes={lora_dtypes} norm_dtypes={norm_dtypes} "
          f"lm_head_dtype={str(lm_head_p.dtype) if lm_head_p is not None else 'n/a'}",
          flush=True)

    cfg = AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B-Instruct',
                                      cache_dir=CACHE_DIR, local_files_only=True)
    vocab_size = cfg.vocab_size

    import bitsandbytes as bnb
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.01)

    # Detect the SDPA backend actually chosen by running one attention op through
    # each candidate and seeing which succeeds under the current context.
    def _detect_sdpa_backend():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch.nn.functional as F
        q = torch.randn(1, 24, SEQ_LEN, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 24, SEQ_LEN, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 24, SEQ_LEN, 128, device='cuda', dtype=torch.bfloat16)
        for name, backend in [('FLASH', SDPBackend.FLASH_ATTENTION),
                              ('EFFICIENT', SDPBackend.EFFICIENT_ATTENTION),
                              ('MATH', SDPBackend.MATH)]:
            try:
                with sdpa_kernel([backend]):
                    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                return name
            except Exception:
                continue
        return 'UNKNOWN'

    probe = _detect_sdpa_backend()
    print(f"  sdpa probe (bf16 QKV, causal, no mask): {probe}", flush=True)

    step_losses = []
    accum_loss = 0.0
    micro_idx = 0
    global_step = 0
    nan_count = 0
    inf_count = 0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    if data_source == 'synth':
        gen = torch.Generator(device='cuda').manual_seed(seed)
        for step in range(steps):
            input_ids = torch.randint(
                0, vocab_size, (BATCH_SIZE, SEQ_LEN),
                device='cuda', dtype=torch.long, generator=gen,
            )
            out = peft_model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            if torch.isnan(loss): nan_count += 1
            if torch.isinf(loss): inf_count += 1
            step_losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            if (step + 1) % 10 == 0:
                recent = np.mean(step_losses[-10:])
                print(f"  [{arm_name}] step {step+1}/{steps} recent={recent:.6f} "
                      f"latest={step_losses[-1]:.6f}", flush=True)
    else:
        # GSM8K with grad accumulation identical to Test 1.
        for epoch_idx, order in enumerate(data_order):
            for i, idx in enumerate(order):
                if global_step >= steps:
                    break
                sample = train_data[idx]
                text = format_training_prompt(sample['question'], sample['answer_text'])
                inputs = tokenizer(text, return_tensors='pt', truncation=True,
                                   max_length=MAX_SEQ_LEN, padding=False).to('cuda')
                input_ids = inputs['input_ids']
                attn = inputs['attention_mask']
                out = peft_model(input_ids=input_ids, attention_mask=attn,
                                 labels=input_ids)
                loss = out.loss
                if torch.isnan(loss): nan_count += 1
                if torch.isinf(loss): inf_count += 1
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                (loss / GRAD_ACCUM_STEPS).backward()
                accum_loss += loss.item() / GRAD_ACCUM_STEPS
                micro_idx += 1
                if micro_idx % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_losses.append(accum_loss)
                    accum_loss = 0.0
                    global_step += 1
                    if global_step % 10 == 0:
                        recent = np.mean(step_losses[-10:])
                        print(f"  [{arm_name}] ep{epoch_idx+1} step {global_step}/{steps} "
                              f"recent={recent:.6f} latest={step_losses[-1]:.6f}", flush=True)
            if global_step >= steps:
                break

    train_wall = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"[{arm_name}] DONE  final_step_loss={step_losses[-1]:.6f}  "
          f"NaN={nan_count} Inf={inf_count} peak={peak_gb:.2f} GB "
          f"wall={train_wall:.1f}s", flush=True)

    result = {
        'arm': arm_name,
        'dtype_policy': dtype_policy,
        'data_source': data_source,
        'steps': len(step_losses),
        'step_losses': step_losses,
        'final_loss': step_losses[-1] if step_losses else None,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'peak_alloc_gb': peak_gb,
        'train_wall_seconds': train_wall,
        'lora_dtypes': sorted(lora_dtypes),
        'norm_dtypes': sorted(norm_dtypes),
        'lm_head_dtype': str(lm_head_p.dtype) if lm_head_p is not None else None,
        'sdpa_probe': probe,
    }

    del optimizer, peft_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def diff(a, b):
    la = a['step_losses']; lb = b['step_losses']
    n = min(len(la), len(lb))
    arr_a = np.array(la[:n]); arr_b = np.array(lb[:n])
    d = arr_a - arr_b
    return {
        'n_steps': n,
        'max_abs': float(np.max(np.abs(d))),
        'mean_abs': float(np.mean(np.abs(d))),
        'final_abs': float(abs(d[-1])),
        'pass_5e_3': bool(np.max(np.abs(d)) < 5e-3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--data_source', choices=['synth', 'gsm8k'], default='synth')
    p.add_argument('--n_train', type=int, default=500,
                   help='(gsm8k only) how many GSM8K samples to draw before stepping.')
    p.add_argument('--epochs', type=int, default=1,
                   help='(gsm8k only) shuffle passes over --n_train samples.')
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()

    train_data = None
    data_order = None
    tokenizer = None
    if args.data_source == 'gsm8k':
        train_data = load_gsm8k(args.n_train, args.seed)
        data_order = build_data_order(len(train_data), args.seed, args.epochs)
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.2-3B-Instruct', cache_dir=CACHE_DIR, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    res1 = train_arm('Arm1_fp32',                'fp32',
                     args.steps, args.seed, args.lr, args.data_source,
                     train_data, data_order, tokenizer)
    res2 = train_arm('Arm2_bf16norms_LoRAfp32', 'restore_bf16_lora_fp32',
                     args.steps, args.seed, args.lr, args.data_source,
                     train_data, data_order, tokenizer)
    res3 = train_arm('Arm3_bf16all_LoRAbf16',   'restore_bf16_lora_bf16',
                     args.steps, args.seed, args.lr, args.data_source,
                     train_data, data_order, tokenizer)
    res4 = train_arm('Arm4_bf16RMSNorm',        'arm4_bf16_rmsnorm',
                     args.steps, args.seed, args.lr, args.data_source,
                     train_data, data_order, tokenizer)

    diffs = {
        'arm2_vs_arm1': diff(res2, res1),
        'arm3_vs_arm1': diff(res3, res1),
        'arm3_vs_arm2': diff(res3, res2),
        'arm4_vs_arm1': diff(res4, res1),
        'arm4_vs_arm3': diff(res4, res3),
    }

    payload = {
        'timestamp': datetime.now().isoformat(),
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'weight_quant': 'nf4',
        'seq_len': SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'data_source': args.data_source,
        'n_train_samples': args.n_train if args.data_source == 'gsm8k' else None,
        'epochs': args.epochs if args.data_source == 'gsm8k' else None,
        'steps': args.steps,
        'seed': args.seed,
        'lr': args.lr,
        'arm1': res1,
        'arm2': res2,
        'arm3': res3,
        'arm4': res4,
        'diffs': diffs,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)

    print("\n### PEAK VRAM (GB) ###", flush=True)
    for r in (res1, res2, res3, res4):
        print(f"  {r['arm']:32s} peak={r['peak_alloc_gb']:.3f}  final_loss={r['final_loss']:.6f}  "
              f"NaN={r['nan_count']} Inf={r['inf_count']}", flush=True)
    print("\n### DIFF (vs Arm1) ###", flush=True)
    for name, d in diffs.items():
        rel = d['max_abs'] / max(1e-9, res1['final_loss']) * 100.0
        pass_abs = 'PASS' if d['pass_5e_3'] else 'FAIL'
        print(f"  {name:22s}  max_abs={d['max_abs']:.6e}  mean_abs={d['mean_abs']:.6e}  "
              f"final={d['final_abs']:.6e}  rel_max_pct={rel:.3f}%  [abs5e-3:{pass_abs}]", flush=True)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == '__main__':
    main()
