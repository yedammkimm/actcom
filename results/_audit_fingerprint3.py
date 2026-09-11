#!/usr/bin/env python3
"""Full-schema fingerprint match: walks summary_table, summary, results, seed_results,
raw_results, per_run_results, and any *_per_seed lists."""
import json, math, os, re
from glob import glob

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob(os.path.join(RESULTS_DIR, "*.json")))

TARGETS = {
    "Run-A / Standard  [58.0,59.4,59.2]": [58.0, 59.4, 59.2],
    "Run-A / OAMP      [58.2,58.0,59.6]": [58.2, 58.0, 59.6],
    "Run-A / Naive-FP4 [61.2,57.6,58.4]": [61.2, 57.6, 58.4],
    "Run-D / Standard  [57.6,57.8,58.2]": [57.6, 57.8, 58.2],
}
STAT_TARGETS = [
    ("Run-B  Standard mean=58.5 std=0.7", 58.5, 0.7),
    ("Run-C  Standard mean=59.3 std=0.3", 59.3, 0.3),
    ("Run-C  OAMP-full mean=58.7 std=0.8", 58.7, 0.8),   # Table 18 OAMP-full
]
SINGLE_TARGETS = [
    ("Run-E  seed42 Standard=58.6",  42, "standard", 58.6),
    ("Run-E  seed42 OAMP-full=60.4", 42, "oamp",     60.4),
]
TOL = 0.15

def std(v, d):
    n = len(v)
    if n - d <= 0: return None
    m = sum(v)/n
    return math.sqrt(sum((x-m)**2 for x in v)/(n-d))

def to_pct(x):
    if not isinstance(x, (int, float)): return None
    return x*100.0 if 0.0 < x <= 1.5 else float(x)

def valid_pct_list(vals):
    conv = [to_pct(v) for v in vals]
    if any(c is None for c in conv): return None
    if not all(1.0 <= c <= 99.9 for c in conv): return None
    return conv

records = []

def add(fn, label, seeds, accs, meta=None):
    accs = [round(a, 4) for a in accs]
    records.append({"file": fn, "label": label, "seeds": seeds, "accs": accs, "meta": meta or {}})

def scan_per_seed_dict(fn, label, per_seed_dict, meta):
    """per_seed = {seed_str: {'accuracy': ..} | float}"""
    if not isinstance(per_seed_dict, dict): return
    seeds, vals = [], []
    for k, v in per_seed_dict.items():
        try: sk = int(k)
        except: sk = k
        seeds.append(sk)
        if isinstance(v, dict):
            av = v.get("accuracy") or v.get("acc")
            if av is None: return
            vals.append(av)
        elif isinstance(v, (int, float)):
            vals.append(v)
        else: return
    conv = valid_pct_list(vals)
    if not conv: return
    add(fn, label, seeds, conv, meta)

def scan_file(path):
    fn = os.path.basename(path)
    try: data = json.load(open(path))
    except: return
    if not isinstance(data, dict): return
    top_seeds = data.get("seeds") or []
    model = data.get("model")
    meta = {"model": model, "top_seeds": top_seeds, "config": data.get("config")}

    # results[method]{accuracies: [...], per_seed: {...}}
    r = data.get("results")
    if isinstance(r, dict):
        for method, mres in r.items():
            if not isinstance(mres, dict): continue
            accs = mres.get("accuracies") or mres.get("acc_per_seed")
            if isinstance(accs, list):
                conv = valid_pct_list(accs)
                if conv: add(fn, f"results/{method}/accuracies", top_seeds, conv, meta)
            # crossarch schema: {arch}{"Standard_per_seed": {42: {accuracy: ..}, ...}, "Standard_mean_acc":..}
            for k, v in mres.items():
                if k.endswith("_per_seed") and isinstance(v, dict):
                    scan_per_seed_dict(fn, f"results/{method}/{k}", v, meta)
                elif k.endswith("_per_seed") and isinstance(v, list):
                    conv = valid_pct_list(v)
                    if conv: add(fn, f"results/{method}/{k}", top_seeds, conv, meta)

    # summary[method]{per_seed: {42: {accuracy: ..}}, mean_accuracy, ...}
    s = data.get("summary")
    if isinstance(s, dict):
        for method, mres in s.items():
            if not isinstance(mres, dict): continue
            ps = mres.get("per_seed")
            if isinstance(ps, dict):
                scan_per_seed_dict(fn, f"summary/{method}/per_seed", ps, meta)
            elif isinstance(ps, list):
                conv = valid_pct_list(ps)
                if conv: add(fn, f"summary/{method}/per_seed", top_seeds, conv, meta)

    # summary_table[method][benchmark]{per_seed: {..}, mean, std}
    st = data.get("summary_table")
    if isinstance(st, dict):
        for method, benches in st.items():
            if not isinstance(benches, dict): continue
            for b, vals in benches.items():
                if not isinstance(vals, dict): continue
                ps = vals.get("per_seed")
                if isinstance(ps, dict):
                    scan_per_seed_dict(fn, f"summary_table/{method}/{b}/per_seed", ps, meta)

    # seed_results = [ {seed:42, Standard:..}, ... ]
    sr = data.get("seed_results")
    if isinstance(sr, list) and sr and isinstance(sr[0], dict):
        seeds = [row.get("seed") for row in sr]
        for m in [k for k in sr[0].keys() if k != "seed" and not k.endswith("_retention")]:
            vals = [row.get(m) for row in sr]
            conv = valid_pct_list(vals)
            if conv: add(fn, f"seed_results/{m}", seeds, conv, meta)

    # per_run_results[bucket]{seed_str: {accuracy: ..}}
    prr = data.get("per_run_results")
    if isinstance(prr, dict):
        for bucket, per_seed in prr.items():
            if isinstance(per_seed, dict):
                scan_per_seed_dict(fn, f"per_run_results/{bucket}", per_seed, meta)

    # flat single-seed (test_a_fixbefore)
    seed = data.get("seed")
    if isinstance(seed, int):
        for k, v in data.items():
            if isinstance(v, (int, float)) and k.endswith("_accuracy"):
                conv = to_pct(v)
                if conv is not None:
                    add(fn, f"flat/{k}", [seed], [conv], meta)

for p in FILES: scan_file(p)

# dedup
seen = set(); uniq = []
for r in records:
    key = (r["file"], r["label"], tuple(r["accs"]))
    if key in seen: continue
    seen.add(key); uniq.append(r)
records = uniq

print(f"# scanned {len(FILES)} files, {len(records)} per-seed accuracy records\n")

def matches(cand, tgt, tol=TOL):
    if not cand or len(cand) != len(tgt): return False
    rem = list(cand)
    for t in tgt:
        best, bd = None, tol + 1e-9
        for i, c in enumerate(rem):
            d = abs(c - t)
            if d < bd: bd, best = d, i
        if best is None: return False
        rem.pop(best)
    return True

def fmt_record(r):
    m = sum(r["accs"])/len(r["accs"])
    s0 = std(r["accs"], 0); s1 = std(r["accs"], 1)
    cfg = r["meta"].get("config") or {}
    es = cfg.get("eval_samples") or cfg.get("gsm8k_eval_samples") or (cfg.get("eval") or {}).get("samples")
    cm = cfg.get("compress_mode") or (cfg.get("training") or {}).get("compress_mode")
    lr = cfg.get("lr") or (cfg.get("training") or {}).get("lr")
    ep = cfg.get("epochs") or cfg.get("num_epochs") or (cfg.get("training") or {}).get("epochs")
    msl = cfg.get("max_seq_len") or (cfg.get("training") or {}).get("max_seq_len")
    lora_t = cfg.get("lora_target_modules") or ((cfg.get("lora") or {}).get("targets")) or cfg.get("target_modules")
    return (f"    seeds={r['seeds']} accs={r['accs']} mean={m:.3f} s0={s0:.3f} s1={s1:.3f}\n"
            f"    model={r['meta'].get('model')} eval={es} compress_mode={cm} lr={lr} epochs={ep} "
            f"max_seq_len={msl} lora_targets={lora_t}")

print("="*90)
print("PER-SEED MULTISET MATCHES  (tol=+-{:.2f} pp)".format(TOL))
print("="*90)
for label, target in TARGETS.items():
    print(f"\n[{label}]")
    hits = 0
    for r in records:
        if matches(r["accs"], target):
            hits += 1
            print(f"  MATCH {r['file']}  label={r['label']}")
            print(fmt_record(r))
    if hits == 0: print("  (no exact match)")

print("\n" + "="*90)
print("MEAN/STD MATCHES")
print("="*90)
for label, mt, st in STAT_TARGETS:
    print(f"\n[{label}]")
    hits = 0
    for r in records:
        if len(r["accs"]) < 2: continue
        m = sum(r["accs"])/len(r["accs"])
        s0 = std(r["accs"], 0); s1 = std(r["accs"], 1)
        if abs(m - mt) <= 0.15 and (
            (s0 is not None and abs(s0 - st) <= 0.2) or
            (s1 is not None and abs(s1 - st) <= 0.2)):
            hits += 1
            print(f"  MATCH {r['file']}  label={r['label']}")
            print(fmt_record(r))
    if hits == 0: print("  (no match)")

print("\n" + "="*90)
print("SEED-42 SINGLE MATCHES")
print("="*90)
for label, sv, needle, tv in SINGLE_TARGETS:
    print(f"\n[{label}]")
    hits = 0
    for r in records:
        seeds = r["seeds"] or []
        idx = seeds.index(sv) if sv in seeds else (0 if len(seeds)==1 and seeds[0]==sv else None)
        if idx is None or idx >= len(r["accs"]): continue
        if needle not in r["label"].lower(): continue
        v = r["accs"][idx]
        if abs(v - tv) <= 0.3:
            hits += 1
            print(f"  MATCH {r['file']}  label={r['label']}  seed42_acc={v:.3f}  accs={r['accs']}")
    if hits == 0: print("  (no match)")

print()
