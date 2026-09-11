"""Held-out perplexity for a saved LoRA adapter (2026-08-26).

Loads the base model with the same dtype policy used in training, attaches the
adapter via ``PeftModel.from_pretrained`` (matches ``batch_eval.py``, differs
from the ``set_peft_model_state_dict`` path used in the gradient probe), and
computes token-level perplexity on two held-out corpora:

  * GSM8K test set  — same ``format_train_prompt`` template used during LoRA
    training; measures how well the adapter learned the training distribution.
  * WikiText-2 raw v1 test — natural text concatenated then chunked; measures
    whether general LM ability was damaged by compression.

Load order (bit-for-bit identical to run_experiment.py::main and batch_eval.py):
    1. Base model + prepare_model_for_kbit_training (NF4 only).
    2. PeftModel.from_pretrained(base, adapter_dir).
    3. apply_dtype_policy(model, weight_quant, bf16_rmsnorm).
    4. model.eval() + torch.no_grad() forward.

Perplexity math:

    out = model(input_ids=ids, labels=ids)     # HF shifts internally
    n_tok = ids.shape[1] - 1                   # tokens that receive a loss
    nll_sum += out.loss.item() * n_tok         # HF loss is per-token mean
    ppl = exp(nll_sum / total_tokens)

Usage:
    docker exec hma-container bash -c "cd /app/HMA_Project && python \\
        scripts/eval_perplexity.py \\
        --adapter_path results/naive4bit_e2m1/....seed42..._adapter \\
        --wikitext_chunks 200 --gsm8k_samples 500 --max_len 512"
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/app/hf_cache")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from oamp.data import load_task, format_train_prompt
from oamp.dtype_policy import apply_dtype_policy


CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_base_and_adapter(base_model_id: str, adapter_path: str,
                          weight_quant: str, bf16_rmsnorm: bool):
    """Same order as scripts/batch_eval.py (verified adapter-load path)."""
    t0 = time.time()
    if weight_quant == "nf4":
        if base_model_id.endswith("-bnb-4bit"):
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
    print(f"[ppl] base loaded in {time.time()-t0:.1f}s  "
          f"cuda_alloc={torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB", flush=True)

    t1 = time.time()
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    print(f"[ppl] adapter loaded in {time.time()-t1:.1f}s", flush=True)

    model, _param_ptrs, dtype_report = apply_dtype_policy(
        model, weight_quant=weight_quant, bf16_rmsnorm=bf16_rmsnorm)
    print(f"[ppl] dtype_report: rmsnorm_swapped={dtype_report.get('rmsnorm_swapped')}  "
          f"norms={dtype_report.get('norms')}", flush=True)
    model.eval()
    return model, dtype_report


@torch.no_grad()
def compute_ppl_from_ids(model, id_lists, *, label: str, log_every: int = 25):
    """Token-weighted perplexity from pre-tokenized id lists.

    Each element of ``id_lists`` is a python list of int token ids. No further
    truncation or padding is done — callers are responsible for producing lists
    of length in [2, max_len].
    """
    total_nll = 0.0
    total_tok = 0
    n_used = 0
    n_skipped_short = 0
    n_skipped_nonfinite = 0
    t0 = time.time()
    n_total = len(id_lists)
    for i, ids_list in enumerate(id_lists):
        if len(ids_list) < 2:
            n_skipped_short += 1
            continue
        ids = torch.tensor(ids_list, dtype=torch.long, device=DEVICE).unsqueeze(0)
        out = model(input_ids=ids, labels=ids)
        loss = float(out.loss.detach())
        if not math.isfinite(loss):
            n_skipped_nonfinite += 1
            continue
        n_tok = ids.shape[1] - 1
        total_nll += loss * n_tok
        total_tok += n_tok
        n_used += 1
        if log_every and (i + 1) % log_every == 0:
            partial = math.exp(total_nll / total_tok) if total_tok else float('nan')
            print(f"[ppl:{label}]   {i+1:>5}/{n_total:<5}  "
                  f"toks={total_tok:>7}  running_ppl={partial:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    if total_tok == 0:
        return float('nan'), 0, 0
    ppl = math.exp(total_nll / total_tok)
    print(f"[ppl:{label}] DONE  ppl={ppl:.4f}  tokens={total_tok}  "
          f"n_texts={n_used}/{n_total}  "
          f"skipped_short={n_skipped_short}  skipped_nonfinite={n_skipped_nonfinite}  "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    return ppl, total_tok, n_used


def build_gsm8k_id_lists(tokenizer, n_samples: int, seed: int, *, max_len: int):
    """Tokenize the GSM8K test set with the exact training template."""
    test = load_task("gsm8k", "test", n_samples=n_samples, seed=seed,
                     cache_dir=CACHE_DIR, shuffle=False)
    out = []
    for s in test:
        text = format_train_prompt("gsm8k", s)
        ids = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=max_len).input_ids
        out.append(ids)
    return out


def build_wikitext_id_lists(tokenizer, *, max_len: int, n_chunks: int):
    """Concatenate WikiText-2 raw v1 test and split into ``max_len`` token windows.

    Standard PPL protocol: skip empty rows, concatenate remainder, chunk. The
    last (short) chunk is dropped so every window has identical length. No
    decode+re-encode round-trip — ids are used directly.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                      cache_dir=CACHE_DIR, split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer(text, add_special_tokens=False).input_ids
    total_chunks = len(ids) // max_len
    if n_chunks and n_chunks > 0:
        total_chunks = min(total_chunks, n_chunks)
    return [ids[i * max_len : (i + 1) * max_len] for i in range(total_chunks)]


# ----------------------------------------------------------------
# Extra OOD corpora (2026-08-31) — beyond WikiText-2, to show the 4-D FP4
# damage generalises across distributions.  Each entry lists a HF dataset,
# a row → text extractor, and a note for the JSON output.
#
# Only offline-cached datasets are listed; on-the-fly downloads are avoided
# in the container. c4 needs network access and is not included by default.
# ----------------------------------------------------------------

def _narrativeqa_text(row):
    d = row.get('document') or {}
    if isinstance(d, dict):
        return d.get('text') or ''
    return str(d)


_OOD_CORPORA = {
    'narrativeqa': dict(
        hf_name='narrativeqa', config=None, split='test',
        row_to_text=_narrativeqa_text,
        note='Stories / documents (fiction, film scripts, book chapters).',
    ),
    'govreport': dict(
        hf_name='ccdv/govreport-summarization', config='document', split='test',
        row_to_text=lambda r: r.get('report') or '',
        note='US Government reports (formal, legal, technical prose).',
    ),
}


def build_ood_id_lists(tokenizer, corpus_name: str, *,
                       max_len: int, n_chunks: int, seed: int = 42):
    """Row-streaming chunker for a cached HF dataset.

    Streams rows one at a time, tokenises each independently, and stops once
    ``n_chunks * max_len`` tokens are collected (plus a 5% buffer). This bounds
    memory: naively concatenating 10 k narrativeqa documents before tokenising
    blows the 80 GB cgroup. Chunks are equal-length ``max_len`` windows.
    """
    if corpus_name not in _OOD_CORPORA:
        raise ValueError(f"unknown OOD corpus {corpus_name!r}. "
                         f"available: {list(_OOD_CORPORA)}")
    cfg = _OOD_CORPORA[corpus_name]
    from datasets import load_dataset
    if cfg['config']:
        ds = load_dataset(cfg['hf_name'], cfg['config'],
                          cache_dir=CACHE_DIR, split=cfg['split'])
    else:
        ds = load_dataset(cfg['hf_name'],
                          cache_dir=CACHE_DIR, split=cfg['split'])
    target = int(n_chunks * max_len * 1.05) if n_chunks and n_chunks > 0 else None
    eos = tokenizer.eos_token_id
    all_ids = []
    for r in ds:
        t = cfg['row_to_text'](r)
        if not (isinstance(t, str) and t.strip()):
            continue
        ids = tokenizer(t.strip(), add_special_tokens=False).input_ids
        all_ids.extend(ids)
        if eos is not None:
            all_ids.append(eos)   # document boundary
        if target is not None and len(all_ids) >= target:
            break
    total_chunks = len(all_ids) // max_len
    if n_chunks and n_chunks > 0:
        total_chunks = min(total_chunks, n_chunks)
    return [all_ids[i * max_len : (i + 1) * max_len] for i in range(total_chunks)]


def read_training_config(adapter_path: str):
    """Best-effort: read the matching accuracy JSON to record training config."""
    ap = adapter_path.rstrip("/")
    if ap.endswith("_adapter"):
        base = ap[: -len("_adapter")]
        cand = base + ".json"
        if os.path.exists(cand):
            try:
                return json.load(open(cand)).get("config", {})
            except Exception:
                pass
    return {}


def main():
    p = argparse.ArgumentParser(description="Held-out perplexity for a saved LoRA adapter.")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--base_model", default=None,
                   help="HF id. Defaults to adapter_config.json's base_model_name_or_path.")
    p.add_argument("--weight_quant", default="nf4", choices=["nf4", "bf16"])
    p.add_argument("--bf16_rmsnorm", dest="bf16_rmsnorm", action="store_true", default=True)
    p.add_argument("--no_bf16_rmsnorm", dest="bf16_rmsnorm", action="store_false")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--gsm8k_samples", type=int, default=500,
                   help="Number of GSM8K test rows (0 = full 1319).")
    p.add_argument("--wikitext_chunks", type=int, default=200,
                   help="Number of max_len windows from WikiText-2 test (0 = all).")
    p.add_argument("--extra_corpora", type=str, default="",
                   help="Comma-separated names of extra OOD corpora to add. "
                        "Available: " + ",".join(sorted(_OOD_CORPORA)))
    p.add_argument("--extra_chunks", type=int, default=200,
                   help="Number of max_len windows per extra corpus (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_gsm8k", action="store_true")
    p.add_argument("--skip_wikitext", action="store_true")
    p.add_argument("--output", default=None,
                   help="Output JSON. Default: <adapter>_ppl.json")
    args = p.parse_args()

    base_model_id = args.base_model
    if base_model_id is None:
        cfg_path = os.path.join(args.adapter_path, "adapter_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"adapter_config.json not found in {args.adapter_path}")
        base_model_id = json.load(open(cfg_path))["base_model_name_or_path"]

    print(f"[ppl] adapter    = {args.adapter_path}", flush=True)
    print(f"[ppl] base_model = {base_model_id}", flush=True)
    print(f"[ppl] weight_quant={args.weight_quant}  bf16_rmsnorm={args.bf16_rmsnorm}  "
          f"max_len={args.max_len}", flush=True)

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, cache_dir=CACHE_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, dtype_report = load_base_and_adapter(
        base_model_id, args.adapter_path, args.weight_quant, args.bf16_rmsnorm)

    payload = {
        "adapter_path": args.adapter_path,
        "base_model": base_model_id,
        "weight_quant": args.weight_quant,
        "bf16_rmsnorm": args.bf16_rmsnorm,
        "max_len": args.max_len,
        "gsm8k_samples": args.gsm8k_samples,
        "wikitext_chunks": args.wikitext_chunks,
        "seed": args.seed,
        "dtype_report": dtype_report,
        "training_config": read_training_config(args.adapter_path),
    }

    if not args.skip_gsm8k:
        gsm_ids = build_gsm8k_id_lists(tokenizer, args.gsm8k_samples, args.seed,
                                       max_len=args.max_len)
        print(f"[ppl] GSM8K test rows = {len(gsm_ids)}", flush=True)
        ppl, toks, nused = compute_ppl_from_ids(
            model, gsm_ids, label="gsm8k")
        payload["gsm8k_test_ppl"] = ppl
        payload["gsm8k_test_tokens"] = toks
        payload["gsm8k_test_n_texts"] = nused

    if not args.skip_wikitext:
        wt_ids = build_wikitext_id_lists(
            tokenizer, max_len=args.max_len, n_chunks=args.wikitext_chunks)
        print(f"[ppl] WikiText-2 windows = {len(wt_ids)} "
              f"({args.max_len} tok each)", flush=True)
        ppl, toks, nused = compute_ppl_from_ids(
            model, wt_ids, label="wikitext2")
        payload["wikitext2_ppl"] = ppl
        payload["wikitext2_tokens"] = toks
        payload["wikitext2_n_texts"] = nused

    # Extra OOD corpora — measured with the same chunking protocol as WikiText-2
    # so cross-corpus comparisons are apples-to-apples at fixed token budget.
    extra_names = [s.strip() for s in args.extra_corpora.split(",") if s.strip()]
    for corpus_name in extra_names:
        if corpus_name not in _OOD_CORPORA:
            print(f"[ppl] SKIP unknown OOD corpus {corpus_name!r} "
                  f"(known: {list(_OOD_CORPORA)})", flush=True)
            continue
        ids_list = build_ood_id_lists(
            tokenizer, corpus_name,
            max_len=args.max_len, n_chunks=args.extra_chunks, seed=args.seed)
        print(f"[ppl] {corpus_name} windows = {len(ids_list)} "
              f"({args.max_len} tok each) — {_OOD_CORPORA[corpus_name]['note']}",
              flush=True)
        ppl, toks, nused = compute_ppl_from_ids(
            model, ids_list, label=corpus_name)
        payload[f"{corpus_name}_ppl"] = ppl
        payload[f"{corpus_name}_tokens"] = toks
        payload[f"{corpus_name}_n_texts"] = nused

    out = args.output or (args.adapter_path.rstrip("/") + "_ppl.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[ppl] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
