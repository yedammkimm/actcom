"""Probe whether FLASH backend actually runs on Llama-3.2-3B train forward+backward.

Runs a single train step under sdpa_kernel([FLASH]) and reports:
  - Whether FLASH accepted the call (no fallback error)
  - What kind of mask HF passed to SDPA (via a monkey-patch on F.scaled_dot_product_attention)
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
CACHE_DIR  = '/app/hf_cache'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2)
    ap.add_argument('--L', type=int, default=512)
    ap.add_argument('--backend', choices=['none', 'flash', 'math', 'efficient'], default='flash')
    args = ap.parse_args()

    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    ).to(dev)
    print(f'_attn_implementation = {model.config._attn_implementation}')
    print(f'layer0 class         = {model.model.layers[0].self_attn.__class__.__name__}')

    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                      lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj'],
                      bias='none')
    model = get_peft_model(model, lcfg)
    model.train()

    # Intercept SDPA calls to see mask shape / dtype / is_causal
    orig_sdpa = F.scaled_dot_product_attention
    captured = []
    def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        info = {
            'q_shape': tuple(q.shape), 'q_dtype': str(q.dtype),
            'k_shape': tuple(k.shape), 'v_shape': tuple(v.shape),
            'mask_type': type(attn_mask).__name__ if attn_mask is not None else 'None',
            'mask_shape': tuple(attn_mask.shape) if isinstance(attn_mask, torch.Tensor) else None,
            'mask_dtype': str(attn_mask.dtype) if isinstance(attn_mask, torch.Tensor) else None,
            'is_causal': is_causal,
            'dropout_p': dropout_p,
        }
        captured.append(info)
        return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale, **kw)
    F.scaled_dot_product_attention = patched_sdpa

    ids = torch.randint(0, model.config.vocab_size, (args.B, args.L), device=dev)

    backend_map = {
        'flash': [SDPBackend.FLASH_ATTENTION],
        'math': [SDPBackend.MATH],
        'efficient': [SDPBackend.EFFICIENT_ATTENTION],
    }

    def run_step():
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.loss.item()

    torch.cuda.reset_peak_memory_stats()
    try:
        if args.backend == 'none':
            loss = run_step()
            print(f'[no ctx] loss={loss:.4f}')
        else:
            with sdpa_kernel(backend_map[args.backend]):
                loss = run_step()
                print(f'[{args.backend}] loss={loss:.4f}')
    except Exception as e:
        print(f'[{args.backend}] EXCEPTION: {type(e).__name__}: {e}')

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f'peak_alloc = {peak_gb:.3f} GB')

    if captured:
        print(f'\ncaptured {len(captured)} SDPA calls, first 2:')
        for c in captured[:2]:
            print(' ', c)
        # Summarize mask usage
        with_mask = sum(1 for c in captured if c['mask_type'] != 'None')
        causal_flag = sum(1 for c in captured if c['is_causal'])
        print(f'\nsummary: total={len(captured)}  with_4d_mask={with_mask}  is_causal_true={causal_flag}')


if __name__ == '__main__':
    main()
