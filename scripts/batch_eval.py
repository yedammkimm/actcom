"""Batch re-evaluation of a saved LoRA adapter (2026-08-18).

Purpose: run GSM8K accuracy on a saved `_adapter/` directory without retraining.
Useful when the eval protocol needs to change (e.g. longer max_new_tokens
because trunc=n_samples/n_samples) or an initial eval crashed on OOM.

Order matches run_experiment.py::main so forward numerics are bit-for-bit
identical to the training-time forward:

    1. Load base model (NF4 online or unsloth pre-quantized -bnb-4bit).
    2. prepare_model_for_kbit_training (NF4 only).
    3. PeftModel.from_pretrained(base, adapter_dir)   # replaces get_peft_model.
    4. apply_dtype_policy (BF16RMSNorm swap + bf16 embed/lm_head/norm/LoRA).
    5. evaluate.

Usage:
    docker exec hma-container bash -c "cd /app/HMA_Project && python \
        scripts/batch_eval.py \
        --adapter_path results/70b_accuracy/accuracy__standard__...__adapter \
        --n_samples 500 --max_new_tokens 384"
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/app/hf_cache")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from oamp.evaluate import evaluate, set_seed
from oamp.data import load_task, default_stop_strings
from oamp.dtype_policy import apply_dtype_policy


CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _parse_stop_strings(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [s for s in raw.split(",") if s]


def main():
    p = argparse.ArgumentParser(description="Re-evaluate a saved LoRA adapter.")
    p.add_argument("--adapter_path", required=True,
                   help="Path to the `_adapter/` directory produced by run_experiment.py.")
    p.add_argument("--base_model", default=None,
                   help="HF id of the base model. If omitted, read from adapter_config.json.")
    p.add_argument("--weight_quant", default="nf4", choices=["nf4", "bf16"])
    p.add_argument("--bf16_rmsnorm", dest="bf16_rmsnorm", action="store_true", default=True)
    p.add_argument("--no_bf16_rmsnorm", dest="bf16_rmsnorm", action="store_false")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--fewshot", type=int, default=8,
                   help="Kept for record; evaluate() defaults to task-specific fewshot.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop_strings", type=str, default=None,
                   help="Comma-separated or JSON list. Omit for task default.")
    p.add_argument("--no_stop_strings", action="store_true",
                   help="Disable stop_strings (matches pre-2026-08-19 protocol).")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Default: <adapter>_reeval_n<N>_mnt<T>.json")
    args = p.parse_args()

    if args.no_stop_strings:
        stop_strings = None
    elif args.stop_strings is not None:
        stop_strings = _parse_stop_strings(args.stop_strings)
    else:
        stop_strings = default_stop_strings(args.task)

    # ---- Infer base model from adapter config if not provided.
    base_model_id = args.base_model
    if base_model_id is None:
        cfg_path = os.path.join(args.adapter_path, "adapter_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"adapter_config.json not found in {args.adapter_path}; "
                                    f"pass --base_model explicitly.")
        ac = json.load(open(cfg_path))
        base_model_id = ac.get("base_model_name_or_path")
        if not base_model_id:
            raise ValueError("adapter_config.json has no 'base_model_name_or_path'; "
                             "pass --base_model explicitly.")

    print(f"[batch_eval] adapter    = {args.adapter_path}", flush=True)
    print(f"[batch_eval] base_model = {base_model_id}", flush=True)
    print(f"[batch_eval] weight_quant={args.weight_quant}  bf16_rmsnorm={args.bf16_rmsnorm}", flush=True)
    print(f"[batch_eval] task={args.task}  n={args.n_samples}  "
          f"max_new_tokens={args.max_new_tokens}  seed={args.seed}", flush=True)
    print(f"[batch_eval] stop_strings={stop_strings!r}", flush=True)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, cache_dir=CACHE_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 1-2: base model + prepare_model_for_kbit_training.
    t0 = time.time()
    if args.weight_quant == "nf4":
        if base_model_id.endswith("-bnb-4bit"):
            # Pre-quantized checkpoint carries its own quantization_config.
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, cache_dir=CACHE_DIR, local_files_only=True,
                low_cpu_mem_usage=True, device_map={"": 0})
        else:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, cache_dir=CACHE_DIR, local_files_only=True,
                quantization_config=bnb,
                low_cpu_mem_usage=True, device_map={"": 0})
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, cache_dir=CACHE_DIR, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE)
    print(f"[batch_eval] base loaded in {time.time()-t0:.1f}s  "
          f"cuda_alloc={torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB", flush=True)

    # ---- Step 3: attach the saved LoRA adapter (replaces get_peft_model).
    t1 = time.time()
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    print(f"[batch_eval] adapter loaded in {time.time()-t1:.1f}s", flush=True)

    # ---- Step 4: dtype policy AFTER PEFT (matches run_experiment.py::main).
    model, _param_ptrs, dtype_report = apply_dtype_policy(
        model, weight_quant=args.weight_quant, bf16_rmsnorm=args.bf16_rmsnorm)
    print(f"[batch_eval] dtype_report: rmsnorm_swapped={dtype_report.get('rmsnorm_swapped')}  "
          f"norms={dtype_report.get('norms')}", flush=True)

    # ---- Step 5: eval.
    test_data = load_task(args.task, "test", n_samples=args.n_samples,
                          seed=args.seed, cache_dir=CACHE_DIR, shuffle=False)
    res = evaluate(
        model, tokenizer, args.task, test_data,
        device=DEVICE, n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        stop_strings=stop_strings,
        label="batch_eval", log_every=50,
    )

    # ---- Save.
    if args.output is None:
        args.output = args.adapter_path.rstrip("/") + \
            f"_reeval_n{args.n_samples}_mnt{args.max_new_tokens}.json"

    payload = {
        "adapter_path": args.adapter_path,
        "base_model": base_model_id,
        "weight_quant": args.weight_quant,
        "bf16_rmsnorm": args.bf16_rmsnorm,
        "task": args.task,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "stop_strings": stop_strings,
        "fewshot": args.fewshot,
        "seed": args.seed,
        "dtype_report": dtype_report,
        "results": res,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[batch_eval] wrote {args.output}", flush=True)
    print(f"[batch_eval] accuracy = {res['accuracy_pct']:.2f}% "
          f"({res['n_correct']}/{res['n_samples']})", flush=True)


if __name__ == "__main__":
    main()
