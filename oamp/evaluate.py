"""Task-agnostic evaluation loop.

Wraps the task backends from :mod:`oamp.data` in a single greedy generation
loop. Bit-exact deterministic verified in Test 0a on Llama-3.2-3B-Instruct
(500/500 samples identical across two consecutive passes, 2026-08-12).
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

import numpy as np
import torch

from .data import (
    default_fewshot, extract_answer_with_source, format_eval_prompt, is_correct, load_task,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, tokenizer, task: str, test_data: List[dict], *,
             device,
             n_samples: int,
             max_new_tokens: int = 256,
             fewshot: Optional[List[dict]] = None,
             stop_strings: Optional[List[str]] = None,
             label: str = 'eval',
             log_every: int = 50,
             abort_gb: Optional[float] = None,
             abort_check_every: int = 20,
             on_abort: Optional[callable] = None) -> dict:
    """Task-agnostic accuracy loop. Returns the ``results.*`` block for the JSON.

    ``stop_strings=None`` disables text-based early stopping (preserves the
    pre-2026-08-19 protocol). Pass ``default_stop_strings(task)`` from
    :mod:`oamp.data` to enable few-shot boundary stopping.

    Memory watchdog (2026-08-31):
      * If ``abort_gb`` is set, every ``abort_check_every`` samples the loop
        checks ``torch.cuda.memory_reserved(device) / 1e9`` and, if it exceeds
        the threshold, calls ``on_abort(diag)`` (if provided — meant to save a
        partial JSON) and then ``sys.exit(1)``.
      * ``sys.exit(1)`` is used because SystemExit propagates past bare
        ``except Exception`` handlers, so a training driver that swallows
        exceptions cannot silently continue past an abort.
    """
    import sys
    fs = fewshot if fewshot is not None else default_fewshot(task)
    n = min(n_samples, len(test_data)) if n_samples > 0 else len(test_data)
    print(f"\n[{label}] running {task} eval over {n} samples", flush=True)
    if stop_strings:
        print(f"[{label}] stop_strings={stop_strings!r}", flush=True)
    if abort_gb is not None:
        print(f"[{label}] mem watchdog: abort if reserved > {abort_gb:.1f} GB "
              f"(check every {abort_check_every} samples)", flush=True)
    model.eval()

    per_sample = []
    correct = 0
    t0 = time.time()
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if stop_strings:
        gen_kwargs['stop_strings'] = list(stop_strings)
        gen_kwargs['tokenizer'] = tokenizer
    for i, sample in enumerate(test_data[:n]):
        prompt = format_eval_prompt(task, sample, fewshot=fs)
        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_kwargs,
            )
        gen_ids = gen[0, inputs['input_ids'].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred, source = extract_answer_with_source(task, gen_text)
        gold = sample['answer_number']
        ok = is_correct(task, pred, gold)
        if ok:
            correct += 1
        n_gen = int(gen_ids.shape[0])
        per_sample.append({
            'idx':          i,
            'n_gen_tokens': n_gen,
            'truncated':    bool(n_gen >= max_new_tokens),
            'pred':         pred,
            'gold':         gold,
            'correct':      bool(ok),
            'source':       source,
        })
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{n}  acc={correct/(i+1)*100:.2f}%  "
                  f"elapsed={elapsed:.1f}s", flush=True)
        # ---- Memory watchdog ----
        if abort_gb is not None and (i + 1) % abort_check_every == 0:
            r = torch.cuda.memory_reserved(device) / 1e9
            if r > abort_gb:
                partial_acc = correct / (i + 1) * 100.0
                print(f"[{label}:watchdog] ABORT reserved={r:.2f}GB > "
                      f"abort_gb={abort_gb:.1f}GB at sample {i+1}/{n} "
                      f"partial_acc={partial_acc:.2f}% ({correct}/{i+1})",
                      flush=True)
                diag = {
                    'oom_aborted': True,
                    'abort_reason': 'eval_reserved_gb_exceeded',
                    'abort_reserved_gb': float(r),
                    'abort_at_sample': int(i + 1),
                    'abort_of_total': int(n),
                    'partial_accuracy_pct': float(partial_acc),
                    'partial_n_correct': int(correct),
                    'partial_per_sample': per_sample,
                }
                if on_abort is not None:
                    try:
                        on_abort(diag)
                    except Exception as e:                       # noqa: BLE001
                        print(f"[{label}:watchdog] on_abort raised: "
                              f"{type(e).__name__}: {e}", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                sys.exit(1)

    acc = correct / n * 100.0
    dt = time.time() - t0
    print(f"[{label}] acc={acc:.4f}%  ({correct}/{n})  time={dt:.1f}s", flush=True)

    def _c(src):
        return sum(1 for s in per_sample if s['source'] == src)
    n_trunc = sum(1 for s in per_sample if s['truncated'])
    mean_gen = (sum(s['n_gen_tokens'] for s in per_sample) / n) if n else 0.0
    diagnostics = {
        'n_truncated':      n_trunc,
        'n_from_answer_is': _c('answer_is'),
        'n_from_hash':      _c('hash'),
        'n_from_fallback':  _c('fallback'),
        'n_empty':          _c('empty'),
        'mean_gen_tokens':  float(mean_gen),
        'max_new_tokens':   int(max_new_tokens),
        'stop_strings':     list(stop_strings) if stop_strings else None,
    }
    print(f"[{label}] diag: trunc={n_trunc}  "
          f"answer_is={diagnostics['n_from_answer_is']}  "
          f"hash={diagnostics['n_from_hash']}  "
          f"fallback={diagnostics['n_from_fallback']}  "
          f"empty={diagnostics['n_empty']}  "
          f"mean_gen={mean_gen:.1f}", flush=True)

    return {
        'accuracy_pct':      acc,
        'n_correct':         correct,
        'n_samples':         n,
        'eval_wall_s':       dt,
        'per_sample':        per_sample,
        'eval_diagnostics':  diagnostics,
    }


__all__ = ['set_seed', 'evaluate', 'load_task']
