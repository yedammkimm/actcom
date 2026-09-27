"""Two-minute forward-only parity check: do the saved-tensor hooks change the forward numerics?

Loads the model once, feeds one deterministic sample and computes the loss
twice, as a plain forward and inside the hooks context. If pack() is truly
backward-only the two losses match bit for bit, or within 1e-6 of floating
point noise; a larger difference means pack() is touching the forward
computation. Runs in eval() mode so dropout cannot add noise.

This uses NativeOAMPHooks, the earlier implementation that lived in legacy/,
which the released repository does not include. run_experiment.py performs
the same check on PackHooks before every run.
"""

import argparse
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NativeOAMPHooks is the earlier implementation, which this repository does not include.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'benchmarks')))

from benchmark_native_packing_vram import NativeOAMPHooks  # noqa: E402

CACHE_DIR = "/app/hf_cache"


def load_nf4(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0},
        cache_dir=CACHE_DIR, local_files_only=True,
    )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pass_attn_mask', action='store_true', default=False)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

    cfg = AutoConfig.from_pretrained(args.model, cache_dir=CACHE_DIR, local_files_only=True)
    vocab_size = cfg.vocab_size

    base = load_nf4(args.model)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none',
    )
    peft_model = get_peft_model(base, lora)
    peft_model.eval()

    input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids) if args.pass_attn_mask else None

    def one_forward():
        with torch.no_grad():
            if args.pass_attn_mask:
                out = peft_model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            else:
                out = peft_model(input_ids=input_ids, labels=input_ids)
        return out.loss.item(), out.logits.detach()

    print(f"[cfg] seq_len={args.seq_len} pass_attn_mask={args.pass_attn_mask}", flush=True)

    # Arm A: plain HF forward
    loss_std, logits_std = one_forward()
    print(f"[A] plain HF forward           loss={loss_std!r}", flush=True)

    # Arm B: same forward inside NativeOAMPHooks context
    hooks = NativeOAMPHooks(fp8_ratio=0.20)
    with hooks:
        loss_gamma, logits_gamma = one_forward()
    print(f"[B] inside NativeOAMPHooks     loss={loss_gamma!r}", flush=True)

    # Arm C: repeat plain HF forward (sanity that Arm A is reproducible)
    loss_std2, logits_std2 = one_forward()
    print(f"[C] plain HF forward (again)   loss={loss_std2!r}", flush=True)

    print()
    print(f"|A - B| (hooks vs standard) = {abs(loss_std - loss_gamma):.6e}")
    print(f"|A - C| (reproducibility) = {abs(loss_std - loss_std2):.6e}")
    print(f"logits max abs (A vs B) = {(logits_std - logits_gamma).abs().max().item():.6e}")
    print(f"logits max abs (A vs C) = {(logits_std - logits_std2).abs().max().item():.6e}")

    verdict_ab = abs(loss_std - loss_gamma) < 1e-4
    verdict_ac = abs(loss_std - loss_std2) < 1e-6
    print()
    print(f"PASS (forward unchanged by the hooks): {verdict_ab}")
    print(f"PASS (HF reproducible):     {verdict_ac}")


if __name__ == '__main__':
    main()
