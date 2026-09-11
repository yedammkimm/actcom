"""Multiple-choice benchmarks by log-likelihood, on the verified adapter path.

Why a new script. batch_eval.py scores by generation: it calls `generate` and
parses a string out of the result. A multiple-choice benchmark is scored by
comparing the log-likelihood the model assigns to each candidate continuation,
which never generates a token. The two share nothing but the model.

What is reused. The adapter loading order from eval_perplexity.py, unchanged:
base -> prepare_model_for_kbit_training -> PeftModel -> apply_dtype_policy.
That path is the one under which three adapters reproduced their perplexity to
+/-0.0000, so forward numerics here match training. The script prints
`rmsnorm_swapped` on every run; on Llama-3.2-3B it must read 57. If it does not,
the forward pass is not the one the adapter was trained under and the numbers
are not comparable.

Scoring. For each item we build a context and one continuation per choice, and
sum the log-probabilities of the continuation tokens. Two aggregates are
reported and neither is discarded:

    acc_sum    argmax over the summed log-probability
    acc_mean   argmax over the per-token mean

These correspond in spirit to the harness convention of `acc` and `acc_norm`,
but the normalisation here is per token rather than per byte, so they are
reported under their own names rather than claimed to be identical.

Determinism. Nothing is subsampled and nothing is shuffled. MMLU few-shot
examples are the five dev-split rows for the item's own subject, taken in
dataset order, which is the standard construction. The SHA-256 of the ordered
item list goes into the output JSON so two runs can be shown to have scored the
same items in the same order -- the check that caught a dead `seed` argument in
the perplexity path earlier in this project.

Usage:
    python scripts/eval_multichoice.py --tasks mmlu --output results/x.json
    python scripts/eval_multichoice.py --adapter_path <dir> --tasks all ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/app/HMA_Project")
sys.path.insert(0, "/app/HMA_Project/scripts")

import eval_perplexity as EP          # loading path + CACHE_DIR + DEVICE

CACHE_DIR = EP.CACHE_DIR
DEVICE = EP.DEVICE

TASKS = ("mmlu", "hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande")


# ----------------------------------------------------------------- datasets

def _load(path, cfg=None):
    from datasets import load_dataset
    return (load_dataset(path, cfg, cache_dir=CACHE_DIR) if cfg
            else load_dataset(path, cache_dir=CACHE_DIR))


def build_mmlu():
    """5-shot, letter-scored, few-shot drawn from this subject's dev rows."""
    ds = _load("cais/mmlu", "all")
    dev_by_subject = {}
    for r in ds["dev"]:
        dev_by_subject.setdefault(r["subject"], []).append(r)

    def block(r, with_answer):
        letters = "ABCD"
        lines = [r["question"].strip()]
        for i, c in enumerate(r["choices"]):
            lines.append(f"{letters[i]}. {c}")
        lines.append("Answer:" + (f" {letters[r['answer']]}" if with_answer else ""))
        return "\n".join(lines)

    items = []
    for r in ds["test"]:
        subj = r["subject"].replace("_", " ")
        head = f"The following are multiple choice questions (with answers) about {subj}.\n\n"
        shots = "\n\n".join(block(s, True) for s in dev_by_subject[r["subject"]][:5])
        ctx = head + shots + "\n\n" + block(r, False)
        items.append({"ctx": ctx,
                      "choices": [" A", " B", " C", " D"],
                      "gold": int(r["answer"]),
                      "group": r["subject"]})
    return items


def build_hellaswag():
    ds = _load("Rowan/hellaswag")["validation"]
    items = []
    for r in ds:
        ctx = (r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()).strip()
        items.append({"ctx": ctx,
                      "choices": [" " + e.strip() for e in r["endings"]],
                      "gold": int(r["label"]), "group": None})
    return items


def _arc(cfg):
    ds = _load("allenai/ai2_arc", cfg)["test"]
    items = []
    for r in ds:
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        items.append({"ctx": "Question: " + r["question"].strip() + "\nAnswer:",
                      "choices": [" " + t.strip() for t in r["choices"]["text"]],
                      "gold": labels.index(r["answerKey"]), "group": None})
    return items


def build_arc_challenge(): return _arc("ARC-Challenge")
def build_arc_easy():      return _arc("ARC-Easy")


def build_piqa():
    ds = _load("baber/piqa")["validation"]
    return [{"ctx": "Question: " + r["goal"].strip() + "\nAnswer:",
             "choices": [" " + r["sol1"].strip(), " " + r["sol2"].strip()],
             "gold": int(r["label"]), "group": None} for r in ds]


def build_winogrande():
    """Partial evaluation: the context differs per option, the tail is shared."""
    ds = _load("allenai/winogrande", "winogrande_xl")["validation"]
    items = []
    for r in ds:
        idx = r["sentence"].index("_")
        tail = r["sentence"][idx + 1:]
        items.append({"ctx_per_choice": [r["sentence"][:idx] + r["option1"],
                                         r["sentence"][:idx] + r["option2"]],
                      "cont": tail,
                      "gold": int(r["answer"]) - 1, "group": None})
    return items


BUILDERS = {"mmlu": build_mmlu, "hellaswag": build_hellaswag,
            "arc_challenge": build_arc_challenge, "arc_easy": build_arc_easy,
            "piqa": build_piqa, "winogrande": build_winogrande}


# ----------------------------------------------------------------- scoring

@torch.no_grad()
def score_shared_single(model, tok, ctxs, choice_ids, fast=True):
    """One forward per item when every choice is a single token on one context.

    MMLU is scored by the answer letter, so the four candidates share a
    1200-token prompt and differ only in the final token. Reading the four
    logits off the last position of a single pass gives the same
    log-probabilities as four separate sequences at a quarter of the cost.

    With ``fast`` the lm_head is asked for that one position only
    (``logits_to_keep=1``), which turns a [B, 1200, 128256] block into
    [B, 1, 128256] -- 4.9 GB to 4 MB at batch 16. That requires LEFT padding:
    with right padding the final position of a short sequence is a pad token,
    not the one that predicts the letter. Left padding in turn requires
    explicit ``position_ids``, because Llama derives them from a plain arange
    over the sequence and would otherwise rotate every real token by the pad
    count. ``--verify_fast_path`` checks the two agree.
    """
    enc = [tok(c, add_special_tokens=True).input_ids for c in ctxs]
    maxlen = max(len(e) for e in enc)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp = torch.full((len(enc), maxlen), pad_id, dtype=torch.long)
    att = torch.zeros((len(enc), maxlen), dtype=torch.long)
    for i, e in enumerate(enc):
        if fast:
            inp[i, maxlen - len(e):] = torch.tensor(e, dtype=torch.long)
            att[i, maxlen - len(e):] = 1
        else:
            inp[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            att[i, :len(e)] = 1
    inp, att = inp.to(DEVICE), att.to(DEVICE)
    if fast:
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=inp, attention_mask=att, position_ids=pos,
                    logits_to_keep=1)
        sel = out.logits[:, -1, :]
    else:
        out = model(input_ids=inp, attention_mask=att)
        idx = torch.tensor([len(e) - 1 for e in enc], device=DEVICE)
        sel = out.logits[torch.arange(len(enc), device=DEVICE), idx]
    logp = torch.log_softmax(sel.float(), dim=-1)
    return [[(logp[i, c].item(), logp[i, c].item(), 1) for c in choice_ids]
            for i in range(len(enc))]


@torch.no_grad()
def verify_fast_path(model, tok, items, choice_ids, n=20, batch_size=4):
    """Score the same items both ways and report the largest disagreement.

    The fast path changes padding side, adds position_ids and truncates the
    lm_head. Each of those is a place where an equivalence that holds in
    principle can fail in practice, so it is measured rather than assumed.
    """
    sub = [it["ctx"] for it in items[:n]]
    a, b = [], []
    for i in range(0, len(sub), batch_size):
        a += score_shared_single(model, tok, sub[i:i+batch_size], choice_ids, fast=True)
        b += score_shared_single(model, tok, sub[i:i+batch_size], choice_ids, fast=False)
    worst = 0.0
    flips = 0
    for ra, rb in zip(a, b):
        for (x, _, _), (y, _, _) in zip(ra, rb):
            worst = max(worst, abs(x - y))
        if max(range(len(ra)), key=lambda k: ra[k][0]) != \
           max(range(len(rb)), key=lambda k: rb[k][0]):
            flips += 1
    return worst, flips, len(sub)



@torch.no_grad()
def score_batch(model, tok, pairs):
    """pairs: list of (context_text, continuation_text).

    Returns [(sum_logp, mean_logp, n_cont_tokens)] in the same order.
    Right padding with an attention mask; a decoder-only model reads left to
    right, so the pad tail is masked out and never reached by the positions we
    score.
    """
    enc = [(tok(c, add_special_tokens=True).input_ids,
            tok(x, add_special_tokens=False).input_ids) for c, x in pairs]
    full = [c + x for c, x in enc]
    maxlen = max(len(f) for f in full)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp = torch.full((len(full), maxlen), pad_id, dtype=torch.long)
    att = torch.zeros((len(full), maxlen), dtype=torch.long)
    for i, f in enumerate(full):
        inp[i, :len(f)] = torch.tensor(f, dtype=torch.long)
        att[i, :len(f)] = 1
    out = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE))
    res = []
    for i, (ctx_ids, cont_ids) in enumerate(enc):
        n_ctx, n_cont = len(ctx_ids), len(cont_ids)
        if n_cont == 0:
            res.append((float("-inf"), float("-inf"), 0)); continue
        # Only the continuation positions are scored, so only those rows are
        # normalised: [n_cont, vocab] rather than the whole sequence block.
        sl = out.logits[i, n_ctx - 1:n_ctx + n_cont - 1]
        lp = torch.log_softmax(sl.float(), dim=-1)
        tgt = torch.tensor(cont_ids, device=lp.device)
        ch = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res.append((ch.sum().item(), ch.mean().item(), n_cont))
    return res


MEM_GUARD_GB = 60.0
MEM_SOFT_GB  = 40.0


def fixed_batches(order, batch_size):
    """Group a length-sorted index list into batches of a fixed item count.

    Token-budget batching was implemented, verified and rejected. At a budget
    of 150,000 padded tokens MMLU ran at 1.43 items per second against 1.68 for
    a fixed count of 16, with an identical accuracy (56.97) and an identical
    item hash, so the loss was throughput and not correctness. Batches of 126
    items were no faster than batches of 16, which says the GPU is already
    saturated at 16 and the extra padding buys nothing. A fixed count is also
    the configuration the memory number was measured under: MMLU at batch 16
    peaks at 27 GB reserved.

    Length sorting is what makes a fixed count safe, so it is kept: it puts
    items of similar length together, and a batch is then not sized by one
    outlier.
    """
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]


def memory_guard(name, nb, state=None):
    """Fail this task rather than let the container be killed.

    The GPU sits in an 80 GB cgroup, and a job that reaches the limit is killed
    with no Python traceback -- that is how an early batch-32 attempt vanished
    with exit 0 and no output file. Raising at 60 GB leaves 20 GB of headroom
    and turns an invisible kill into an exception, so the chain script logs the
    task as failed and moves on to the next one instead of losing the run.

    Reserved is not usage. `memory_reserved` counts every block the caching
    allocator holds, including blocks it has finished with and is keeping for
    reuse. Length-sorted batching hands the allocator a slightly different
    shape almost every batch, so those blocks accumulate: HellaSwag reached
    60.8 GB reserved while `memory_allocated` sat at 2.3 GB, and the first
    version of this guard read that as a 60 GB job and killed all 15 cells of
    stage 2, each after 39 minutes of correct work.

    Lowering the batch size, which that version's message advised, makes this
    worse rather than better -- more batches means more distinct shapes.

    So the guard now reclaims before it judges. Above a soft threshold it
    releases the unused blocks and re-reads; only a job still above the hard
    guard after reclaiming is genuinely that large, and only that job raises.
    The safety property is unchanged, because what the cgroup can see is what
    survives `empty_cache`.
    """
    r = torch.cuda.memory_reserved() / 1e9
    if r <= MEM_SOFT_GB:
        return
    a = torch.cuda.memory_allocated() / 1e9
    torch.cuda.empty_cache()
    r2 = torch.cuda.memory_reserved() / 1e9
    if state is not None:
        state["reclaims"] = state.get("reclaims", 0) + 1
        state["reclaimed_gb"] = state.get("reclaimed_gb", 0.0) + (r - r2)
        if state["reclaims"] == 1 or state["reclaims"] % 200 == 0:
            print(f"  [{name}] reclaim #{state['reclaims']} at batch {nb}: "
                  f"reserved {r:.1f} -> {r2:.1f} GB  (allocated {a:.1f})",
                  flush=True)
    if r2 > MEM_GUARD_GB:
        raise RuntimeError(
            f"[{name}] reserved {r2:.1f} GB after reclaim crossed the "
            f"{MEM_GUARD_GB:.0f} GB guard at batch {nb} (allocated {a:.1f} GB) "
            f"-- this one is really that large")


def run_task(model, tok, name, items, batch_size, log_every=200):
    """Flatten every (item, choice) into one queue so batches are full.

    Items whose choices are all single tokens on one shared context take the
    one-pass path above instead.
    """
    single = None
    if all("choices" in it for it in items):
        cand = {tuple(it["choices"]) for it in items}
        if len(cand) == 1:
            ids = [tok(c, add_special_tokens=False).input_ids for c in next(iter(cand))]
            if all(len(i) == 1 for i in ids):
                single = [i[0] for i in ids]

    def _progress(done_units, total_units, t0, extra=""):
        done = done_units / total_units
        print(f"  [{name}] {done_units}/{total_units}  {done*100:5.1f}%  "
              f"{time.time()-t0:6.0f}s  eta {(time.time()-t0)*(1/done-1):5.0f}s  "
              f"peak={torch.cuda.max_memory_reserved()/1e9:.1f} GB{extra}", flush=True)

    if single is not None:
        print(f"  [{name}] shared-context path: {len(items)} passes "
              f"instead of {len(items)*len(single)}", flush=True)
        lens = [len(tok(it["ctx"]).input_ids) for it in items]
        order = sorted(range(len(items)), key=lambda i: lens[i])
        print(f"  [{name}] ctx tokens min={min(lens)} max={max(lens)} "
              f"mean={sum(lens)/len(lens):.0f}  batch={batch_size}", flush=True)
        per_item = [None] * len(items)
        mem_state = {}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        done = nb = 0
        for idx in fixed_batches(order, batch_size):
            got = score_shared_single(model, tok, [items[i]["ctx"] for i in idx],
                                      single, fast=False)
            for k, r in zip(idx, got):
                per_item[k] = r
            done += len(idx); nb += 1
            memory_guard(name, nb, mem_state)
            if nb % log_every == 0:
                _progress(done, len(items), t0, f"  bs={len(idx)}")
        print(f"  [{name}] {nb} batches of {batch_size}", flush=True)
        return _aggregate(items, per_item, name, t0)

    flat, owner = [], []
    for j, it in enumerate(items):
        if "ctx_per_choice" in it:
            for c in it["ctx_per_choice"]:
                flat.append((c, it["cont"])); owner.append(j)
        else:
            for ch in it["choices"]:
                flat.append((it["ctx"], ch)); owner.append(j)
    lens = [len(tok(c).input_ids) + len(tok(x, add_special_tokens=False).input_ids)
            for c, x in flat]
    order = sorted(range(len(flat)), key=lambda i: lens[i])
    print(f"  [{name}] {len(flat)} seqs  tokens min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.0f}  batch={batch_size}", flush=True)
    scores = [None] * len(flat)
    mem_state = {}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    done = nb = 0
    for idx in fixed_batches(order, batch_size):
        for k, r in zip(idx, score_batch(model, tok, [flat[i] for i in idx])):
            scores[k] = r
        done += len(idx); nb += 1
        memory_guard(name, nb, mem_state)
        if nb % log_every == 0:
            _progress(done, len(flat), t0, f"  bs={len(idx)}")
    print(f"  [{name}] {nb} batches of {batch_size}", flush=True)
    per_item = [[] for _ in items]
    for k, j in enumerate(owner):
        per_item[j].append(scores[k])
    return _aggregate(items, per_item, name, t0)


def _aggregate(items, per_item, name, t0):
    n_sum = n_mean = 0
    by_group_sum, by_group_n = {}, {}
    per_sample = []
    for it, sc in zip(items, per_item):
        p_sum = max(range(len(sc)), key=lambda i: sc[i][0])
        p_mean = max(range(len(sc)), key=lambda i: sc[i][1])
        ok_s, ok_m = int(p_sum == it["gold"]), int(p_mean == it["gold"])
        n_sum += ok_s; n_mean += ok_m
        g = it.get("group")
        if g is not None:
            by_group_sum[g] = by_group_sum.get(g, 0) + ok_s
            by_group_n[g] = by_group_n.get(g, 0) + 1
        per_sample.append({"gold": it["gold"], "pred_sum": p_sum,
                           "pred_mean": p_mean, "correct_sum": ok_s})
    n = len(items)
    out = {"n": n, "n_correct_sum": n_sum, "n_correct_mean": n_mean,
           "acc_sum": 100.0 * n_sum / n, "acc_mean": 100.0 * n_mean / n,
           "wall_s": time.time() - t0,
           # The quantity that actually constrains this experiment. Recorded
           # per task so the trend across a 90-hour chain is visible without
           # re-reading the logs.
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
           "per_sample": per_sample}
    if by_group_n:
        out["by_group"] = {g: 100.0 * by_group_sum[g] / by_group_n[g]
                           for g in sorted(by_group_n)}
    return out


def items_hash(items):
    h = hashlib.sha256()
    for it in items:
        key = it.get("ctx") or "".join(it["ctx_per_choice"])
        h.update(key.encode()); h.update(str(it["gold"]).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--adapter_path", default=None,
                   help="omit to score the base model with no adapter")
    p.add_argument("--weight_quant", default="nf4")
    p.add_argument("--bf16_rmsnorm", action="store_true", default=True)
    p.add_argument("--tasks", default="mmlu",
                   help="comma-separated, or 'all'")
    p.add_argument("--batch_size", type=int, default=16,
                   help="items (or context/continuation pairs) per forward. "
                        "16 is the measured MMLU setting, 27 GB reserved; the "
                        "short 0-shot tasks run at 64. Token-budget batching "
                        "was measured slower and removed")
    p.add_argument("--limit", type=int, default=None,
                   help="first N items per task; for plumbing checks only. MMLU "
                        "test is ordered by subject, so a prefix is one subject "
                        "and its accuracy says nothing about the whole benchmark")
    p.add_argument("--verify_batch", type=int, default=0, metavar="N",
                   help="score the first N items at two batch sizes and check "
                        "every prediction agrees; batch composition must not "
                        "change a score")
    p.add_argument("--verify_fast_path", type=int, default=0, metavar="N",
                   help="score the first N items both with and without the "
                        "one-position lm_head and report the largest "
                        "disagreement, then exit")
    p.add_argument("--stride", type=int, default=None,
                   help="every Nth item, which keeps the subject mix; use this "
                        "rather than --limit when a subset has to be read as an "
                        "estimate of the full number")
    p.add_argument("--mem_soft_gb", type=float, default=None,
                   help="reclaim threshold; default 40. Set it low to force "
                        "the reclaim path on a task that never reaches it, "
                        "which is how the equivalence check exercises it.")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    if a.mem_soft_gb is not None:
        global MEM_SOFT_GB
        MEM_SOFT_GB = a.mem_soft_gb

    tasks = list(TASKS) if a.tasks == "all" else [t.strip() for t in a.tasks.split(",")]
    for t in tasks:
        if t not in BUILDERS:
            raise SystemExit(f"unknown task {t!r}; choose from {TASKS}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model, cache_dir=CACHE_DIR,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    t0 = time.time()
    if a.adapter_path:
        model, dtype_report = EP.load_base_and_adapter(
            a.base_model, a.adapter_path, a.weight_quant, a.bf16_rmsnorm)
    else:
        # Same branch without the PEFT wrap, so the base number is measured
        # under the same dtype policy as every adapter.
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        from oamp.dtype_policy import apply_dtype_policy
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            a.base_model, cache_dir=CACHE_DIR, local_files_only=True,
            quantization_config=bnb, low_cpu_mem_usage=True, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model, _p, dtype_report = apply_dtype_policy(
            model, weight_quant=a.weight_quant, bf16_rmsnorm=a.bf16_rmsnorm)
        model.eval()
    print(f"[mc] model ready in {time.time()-t0:.0f}s  "
          f"rmsnorm_swapped={dtype_report.get('rmsnorm_swapped')}  "
          f"lora={dtype_report.get('lora')}", flush=True)
    if dtype_report.get("rmsnorm_swapped") != 57:
        print("[mc] WARNING rmsnorm_swapped != 57 -- forward path differs from "
              "training; results are not comparable", flush=True)

    result = {"base_model": a.base_model, "adapter_path": a.adapter_path,
              "weight_quant": a.weight_quant, "bf16_rmsnorm": a.bf16_rmsnorm,
              "batch_size": a.batch_size, "limit": a.limit, "stride": a.stride,
              "mem_guard_gb": MEM_GUARD_GB,
              "mem_soft_gb": MEM_SOFT_GB,
              "dtype_report": dtype_report, "tasks": {}}
    if a.verify_batch:
        # Batch composition must not change a score. Two batch sizes put
        # different items together and pad to different lengths; if the numbers
        # move, the padding or the masking is leaking between rows.
        items = BUILDERS[tasks[0]]()[:a.verify_batch]
        outs = []
        for bs in (4, a.batch_size):
            r = run_task(model, tok, tasks[0], items, bs)
            outs.append([(p["pred_sum"], p["pred_mean"]) for p in r["per_sample"]])
            print(f"[mc] batch {bs:>4}: acc_sum={r['acc_sum']:.4f}  "
                  f"peak={r['peak_reserved_gb']:.1f} GB", flush=True)
        same = sum(1 for x, y in zip(*outs) if x == y)
        print(f"[mc] batch-invariance: {same}/{len(items)} items agree on both "
              f"aggregates", flush=True)
        raise SystemExit(0 if same == len(items) else 5)

    if a.verify_fast_path:
        items = BUILDERS[tasks[0]]()
        ids = [tok(c, add_special_tokens=False).input_ids[0]
               for c in items[0]["choices"]]
        worst, flips, n = verify_fast_path(model, tok, items, ids,
                                           n=a.verify_fast_path)
        print(f"[mc] fast-path check on {n} items: max |dlogp| = {worst:.2e}, "
              f"argmax flips = {flips}", flush=True)
        ok = worst < 1e-4 and flips == 0
        print(f"[mc] {'AGREE (< 1e-4)' if ok else 'DISAGREE -- do not use the fast path'}",
              flush=True)
        raise SystemExit(0 if ok else 4)

    for t in tasks:
        items = BUILDERS[t]()
        if a.stride:
            items = items[::a.stride]
        if a.limit:
            items = items[:a.limit]
        h = items_hash(items)
        print(f"[mc] {t}: {len(items)} items  hash={h}", flush=True)
        r = run_task(model, tok, t, items, a.batch_size)
        r["items_hash"] = h
        result["tasks"][t] = r
        print(f"[mc] {t}: acc_sum={r['acc_sum']:.2f}  acc_mean={r['acc_mean']:.2f}  "
              f"({r['n_correct_sum']}/{r['n']})  {r['wall_s']:.0f}s  "
              f"peak={r['peak_reserved_gb']:.1f} GB", flush=True)
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
        with open(a.output, "w") as f:
            json.dump(result, f, indent=1)
    print(f"[mc] wrote {a.output}", flush=True)


if __name__ == "__main__":
    # A CUDA OOM or a killed worker inside `docker exec` can surface as a bare
    # exit with no traceback, which is how the first batch-32 attempt vanished:
    # exit 0, no output file, no error. Catch everything, print it, and exit
    # non-zero so a chain script can see the failure.
    try:
        main()
    except SystemExit:
        raise                      # 의도된 종료 코드는 그대로 전달
    except BaseException as e:
        import traceback
        print(f"[mc] FAILED {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        try:
            print(f"[mc] cuda alloc={torch.cuda.memory_allocated()/1e9:.2f} GB "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB", flush=True)
        except Exception:
            pass
        raise SystemExit(3)
