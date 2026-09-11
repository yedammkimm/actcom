"""Diagnostic: is the 500-step WikiText blow-up in the model or in the eval path?

Measures WikiText-2 perplexity on the identical 200 windows used by every run
in this project, for four models in one process:

    base           no adapter at all — establishes what the eval path reports
                   for an untouched model
    C_STD@3736     known-good reference; must reproduce 16.0717
    C_STD@500 s456 the catastrophic run; must reproduce 5646.1273
    C_STD@500 s789 the healthy run from the same cohort; must reproduce 15.3075

Reading:
    base ~16, 3736 reproduces, s456 reproduces  -> the model really is broken;
                                                   eval path and adapter loading
                                                   are both fine
    3736 does NOT reproduce                     -> adapter load path is broken,
                                                   and every perplexity number
                                                   in the project is suspect
    base is itself absurd                       -> the eval path is broken
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import sys, time
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "scripts")

import torch
from transformers import AutoTokenizer
import eval_perplexity as E

BASE = "meta-llama/Llama-3.2-3B-Instruct"
CASES = [
    ("base (no adapter)", None, None),
    ("C_STD@3736 s42", "results/pilot_e2m1/accuracy__standard__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed42__20260820_135332_adapter", 16.0717),
    ("C_STD@500 s456", "results/axis5_short/C_STD_s456_adapter", 5646.1273),
    ("C_STD@500 s789", "results/axis5_short/C_STD_s789_adapter", 15.3075),
]

tok = AutoTokenizer.from_pretrained(BASE, cache_dir=E.CACHE_DIR, local_files_only=True)
wt_ids = E.build_wikitext_id_lists(tok, max_len=512, n_chunks=200)
print(f"[diag] wikitext windows = {len(wt_ids)} x {len(wt_ids[0])} tok", flush=True)

results = []
for label, adapter, expected in CASES:
    print(f"\n{'='*70}\n[diag] {label}\n{'='*70}", flush=True)
    t0 = time.time()
    if adapter is None:
        # Same base-loading branch as load_base_and_adapter, without the PEFT wrap.
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        from oamp.dtype_policy import apply_dtype_policy
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, cache_dir=E.CACHE_DIR, local_files_only=True,
            quantization_config=bnb, low_cpu_mem_usage=True, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model, _p, rep = apply_dtype_policy(model, weight_quant="nf4", bf16_rmsnorm=True)
        print(f"[diag] rmsnorm_swapped={rep.get('rmsnorm_swapped')} norms={rep.get('norms')}", flush=True)
        model.eval()
    else:
        model, rep = E.load_base_and_adapter(BASE, adapter, "nf4", True)

    ppl, toks, n = E.compute_ppl_from_ids(model, wt_ids, label="wikitext2", log_every=100)
    delta = "" if expected is None else f"   expected {expected:.4f}   diff {ppl-expected:+.4f}"
    print(f"[diag] {label}: ppl={ppl:.4f} tokens={toks} n={n} ({time.time()-t0:.0f}s){delta}", flush=True)
    results.append((label, ppl, expected))

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*70}\n[diag] SUMMARY\n{'='*70}")
for label, ppl, expected in results:
    e = "     —" if expected is None else f"{expected:>10.4f}"
    ok = "" if expected is None else ("  REPRODUCED" if abs(ppl-expected)/expected < 0.02 else "  *** MISMATCH ***")
    print(f"  {label:<22} ppl={ppl:>12.4f}   expected={e}{ok}")
