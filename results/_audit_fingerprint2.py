#!/usr/bin/env python3
"""Fingerprint matcher v2: recursively harvest per-seed accuracy lists from any
schema found under results/*.json, then match them against paper targets."""
import json
import math
import os
import re
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
]
SINGLE_TARGETS = [
    ("Run-E  seed42 Standard=58.6",  42, "standard", 58.6),
    ("Run-E  seed42 OAMP-full=60.4", 42, "oamp",     60.4),
]
TOL = 0.15

def std(v, ddof):
    n = len(v)
    if n - ddof <= 0: return None
    m = sum(v)/n
    return math.sqrt(sum((x-m)**2 for x in v)/(n-ddof))

def to_pct(x):
    return x * 100.0 if isinstance(x, (int, float)) and 0.0 < x <= 1.5 else float(x)

def looks_like_acc_list(vals):
    if not isinstance(vals, list) or not (2 <= len(vals) <= 6):
        return False
    if not all(isinstance(v, (int, float)) for v in vals):
        return False
    conv = [to_pct(v) for v in vals]
    return all(1.0 <= c <= 99.9 for c in conv)

# Known per-seed keys
PERSEED_KEYS = re.compile(
    r"(?:^|_)(accuracies|acc_per_seed|per_seed|per_seed_accuracy|per_seed_accuracies|"
    r"[A-Za-z0-9\-]+_per_seed)$"
)

def walk(node, path, records, filename, top_seeds):
    """Recursively collect (label, seeds, acc_list) records."""
    if isinstance(node, dict):
        # detect single-value accuracy at this level: gsm8k_accuracy, oamp_accuracy
        for k, v in node.items():
            if isinstance(v, (int, float)) and re.search(r"(?:^|_)(accuracy|acc)$", k, re.I):
                label = "/".join(path + [k])
                records.append({
                    "file": filename,
                    "label": label,
                    "seeds": top_seeds,
                    "accs": [to_pct(v)],
                    "single": True,
                })
        # detect per-seed lists
        for k, v in node.items():
            if PERSEED_KEYS.search(k) and looks_like_acc_list(v):
                conv = [to_pct(x) for x in v]
                label = "/".join(path + [k])
                records.append({
                    "file": filename,
                    "label": label,
                    "seeds": top_seeds,
                    "accs": conv,
                    "single": False,
                })
        # detect legacy: seed_results = list of dicts keyed by seed number
        for k, v in node.items():
            walk(v, path + [str(k)], records, filename, top_seeds)
    elif isinstance(node, list):
        # skip lists of numbers; already handled via parent-key
        for i, it in enumerate(node):
            if isinstance(it, (dict, list)):
                walk(it, path + [f"[{i}]"], records, filename, top_seeds)

def scan_seed_results_lists(data, filename, records):
    """Special: seed_results = [{seed: 42, Standard: 60.0, OAMP: 58.0, ...}, ...]"""
    top_seeds = data.get("seeds") or []
    sr = data.get("seed_results")
    if not isinstance(sr, list) or not sr or not isinstance(sr[0], dict):
        return
    seeds = [row.get("seed") for row in sr]
    methods = [k for k in sr[0].keys() if k != "seed" and not k.endswith("_retention")]
    for m in methods:
        vals = [row.get(m) for row in sr]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        conv = [to_pct(v) for v in vals]
        records.append({
            "file": filename,
            "label": f"seed_results/{m}",
            "seeds": seeds,
            "accs": conv,
            "single": False,
        })

def scan_raw_results(data, filename, records):
    """benchmark_multidomain: raw_results[method][seed_str][benchmark] = acc"""
    rr = data.get("raw_results")
    if not isinstance(rr, dict):
        return
    for method, per_seed in rr.items():
        if not isinstance(per_seed, dict):
            continue
        seeds = []
        by_bench = {}
        for seed_str, benches in per_seed.items():
            if not isinstance(benches, dict):
                continue
            try: sd = int(seed_str)
            except: sd = seed_str
            seeds.append(sd)
            for b, v in benches.items():
                by_bench.setdefault(b, []).append((sd, v))
        for b, pairs in by_bench.items():
            pairs.sort(key=lambda x: seeds.index(x[0]) if x[0] in seeds else 999)
            vals = [p[1] for p in pairs]
            if not all(isinstance(v, (int, float)) for v in vals):
                continue
            records.append({
                "file": filename,
                "label": f"raw_results/{method}/{b}",
                "seeds": [p[0] for p in pairs],
                "accs": [to_pct(v) for v in vals],
                "single": False,
            })

ALL = []
for p in FILES:
    fn = os.path.basename(p)
    try:
        data = json.load(open(p))
    except Exception:
        continue
    if not isinstance(data, dict):
        continue
    top_seeds = data.get("seeds") or []
    walk(data, [], ALL, fn, top_seeds)
    scan_seed_results_lists(data, fn, ALL)
    scan_raw_results(data, fn, ALL)

# Deduplicate by (file, tuple(accs), label)
seen = set()
uniq = []
for r in ALL:
    key = (r["file"], tuple(round(x, 4) for x in r["accs"]), r["label"])
    if key in seen: continue
    seen.add(key)
    uniq.append(r)
ALL = uniq

print(f"# scanned {len(FILES)} files, {len(ALL)} accuracy records extracted\n")

def matches_multiset(cand, tgt, tol=TOL):
    if not cand or len(cand) != len(tgt): return False
    rem = list(cand)
    for t in tgt:
        best, bd = None, tol + 1e-9
        for i, c in enumerate(rem):
            d = abs(c - t)
            if d < bd: bd = d; best = i
        if best is None: return False
        rem.pop(best)
    return True

print("=" * 90)
print(f"PER-SEED MULTISET MATCHES  (tol=±{TOL} pp)")
print("=" * 90)
for label, target in TARGETS.items():
    print(f"\n[{label}]")
    hits = 0
    for r in ALL:
        if r["single"]: continue
        if matches_multiset(r["accs"], target):
            hits += 1
            s0 = std(r["accs"], 0); s1 = std(r["accs"], 1)
            print(f"  MATCH  {r['file']}")
            print(f"         label={r['label']}  seeds={r['seeds']}")
            print(f"         accs={r['accs']}  s0={s0:.3f}  s1={s1:.3f}")
    if hits == 0:
        print("  (no exact match)  near-misses (maxdiff<=0.5):")
        for r in ALL:
            if r["single"]: continue
            if len(r["accs"]) != len(target): continue
            a = sorted(r["accs"]); t = sorted(target)
            d = [abs(x-y) for x,y in zip(a,t)]
            if max(d) <= 0.5:
                print(f"    ~{r['file']}  {r['label']}  accs={r['accs']}  maxdiff={max(d):.2f}")

print("\n" + "=" * 90)
print("MEAN/STD MATCHES  (Run-B, Run-C)")
print("=" * 90)
for label, mt, st in STAT_TARGETS:
    print(f"\n[{label}]")
    hits = 0
    for r in ALL:
        if r["single"] or len(r["accs"]) < 2: continue
        m = sum(r["accs"])/len(r["accs"])
        s0 = std(r["accs"], 0); s1 = std(r["accs"], 1)
        if abs(m - mt) <= 0.15 and (
            (s0 is not None and abs(s0 - st) <= 0.2) or
            (s1 is not None and abs(s1 - st) <= 0.2)
        ):
            hits += 1
            print(f"  MATCH  {r['file']}  label={r['label']}")
            print(f"         seeds={r['seeds']}  accs={r['accs']}  mean={m:.3f} s0={s0:.3f} s1={s1:.3f}")
    if hits == 0:
        print("  candidates with mean-only match:")
        for r in ALL:
            if r["single"] or len(r["accs"]) < 2: continue
            m = sum(r["accs"])/len(r["accs"])
            if abs(m - mt) <= 0.35:
                s0 = std(r["accs"], 0); s1 = std(r["accs"], 1)
                print(f"    ~{r['file']}  {r['label']}  seeds={r['seeds']}  mean={m:.3f} s0={s0:.3f} s1={s1:.3f} accs={r['accs']}")

print("\n" + "=" * 90)
print("SEED-42 SINGLE-VALUE MATCHES  (Run-E)")
print("=" * 90)
for label, seed_val, needle, target_val in SINGLE_TARGETS:
    print(f"\n[{label}]")
    for r in ALL:
        seeds = r["seeds"] or []
        idx = seeds.index(seed_val) if seed_val in seeds else None
        if idx is None and len(seeds) == 1 and seeds[0] == seed_val:
            idx = 0
        if idx is None: continue
        if idx >= len(r["accs"]): continue
        if needle not in r["label"].lower(): continue
        v = r["accs"][idx]
        if abs(v - target_val) <= 0.3:
            print(f"  MATCH  {r['file']}  label={r['label']}  seed42_acc={v:.3f}  accs={r['accs']}")

print()
