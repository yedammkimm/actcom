"""Base-model GSM8K evaluation (no LoRA) to disambiguate whether the ~68% we see
in Qwen skip-check results is fine-tuning effect or the base model floor.

Uses the same eval configuration as run_experiment.py::_run_accuracy:
- NF4 quantized base weights (bnb_4bit + bf16 compute), or pre-quantized
  ``-bnb-4bit`` checkpoints (skip quant config).
- GSM8K 8-shot fewshot, greedy decode.

CLI overrides env-var defaults for backward-compat with the pre-2026-08-19
chain script. Env-var names are ``BASE_EVAL_<UPPER_ARG>``.

Args:
    --model              HF id (BASE_EVAL_MODEL, default Qwen/Qwen2.5-3B-Instruct)
    --task               task name (BASE_EVAL_TASK, default gsm8k)
    --n_samples          eval sample count (BASE_EVAL_N, default 100)
    --seed               (BASE_EVAL_SEED, default 42)
    --max_new_tokens     (BASE_EVAL_MAX_NEW_TOKENS, default 256)
    --stop_strings       comma-separated (BASE_EVAL_STOP_STRINGS,
                         default = default_stop_strings(task))
    --no_stop_strings    disable stop_strings (matches pre-2026-08-19 protocol)
    --output             (BASE_EVAL_OUT)
"""
import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/app/hf_cache")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from oamp.evaluate import evaluate, set_seed
from oamp.data import load_task, default_stop_strings


CACHE = "/app/hf_cache"


def _env(name, default):
    return os.environ.get(f"BASE_EVAL_{name}", default)


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
    p = argparse.ArgumentParser(description="Base-model GSM8K eval (no LoRA).")
    p.add_argument("--model", default=_env("MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    p.add_argument("--task", default=_env("TASK", "gsm8k"))
    p.add_argument("--n_samples", type=int, default=int(_env("N", "100")))
    p.add_argument("--seed", type=int, default=int(_env("SEED", "42")))
    p.add_argument("--max_new_tokens", type=int,
                   default=int(_env("MAX_NEW_TOKENS", "256")))
    p.add_argument("--fewshot", type=int, default=int(_env("FEWSHOT", "8")),
                   help="Recorded only; evaluate() uses task-default fewshot.")
    p.add_argument("--stop_strings", type=str,
                   default=_env("STOP_STRINGS", None),
                   help="Comma-separated or JSON list. Omit for task default.")
    p.add_argument("--no_stop_strings", action="store_true",
                   help="Disable stop_strings (matches pre-2026-08-19 protocol).")
    p.add_argument("--output", default=_env("OUT", None))
    args = p.parse_args()

    if args.no_stop_strings:
        stop_strings = None
    elif args.stop_strings is not None:
        stop_strings = _parse_stop_strings(args.stop_strings)
    else:
        stop_strings = default_stop_strings(args.task)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    slug = args.model.replace("/", "_").replace("-", "_")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        args.output = (f"/app/HMA_Project/results/base_eval_{slug}_{args.task}_"
                       f"n{args.n_samples}_{ts}.json")

    print(f"[base_eval] model={args.model} task={args.task} "
          f"n={args.n_samples} seed={args.seed}", flush=True)
    print(f"[base_eval] max_new_tokens={args.max_new_tokens} "
          f"stop_strings={stop_strings!r}", flush=True)

    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    if args.model.endswith("-bnb-4bit"):
        m = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=CACHE, local_files_only=True,
            low_cpu_mem_usage=True, device_map={"": 0})
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=CACHE, local_files_only=True,
            quantization_config=bnb).to(device)
    print(f"[base_eval] model loaded in {time.time()-t0:.1f}s, "
          f"cuda_alloc={torch.cuda.memory_allocated(device)/1e9:.2f} GB",
          flush=True)

    test_data = load_task(args.task, "test", n_samples=args.n_samples,
                          seed=args.seed, cache_dir=CACHE, shuffle=False)

    res = evaluate(m, tok, args.task, test_data, device=device,
                   n_samples=args.n_samples,
                   max_new_tokens=args.max_new_tokens,
                   stop_strings=stop_strings,
                   label="base_eval", log_every=20)

    payload = {
        "model": args.model,
        "task": args.task,
        "n": args.n_samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "stop_strings": stop_strings,
        "fewshot": args.fewshot,
        "load_time_s": time.time() - t0,
        "results": res,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[base_eval] wrote {args.output}", flush=True)
    print(f"[base_eval] acc = {res['accuracy_pct']:.2f}% "
          f"({res['n_correct']}/{res['n_samples']})", flush=True)


if __name__ == "__main__":
    main()

