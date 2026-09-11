#!/usr/bin/env python3
"""Fingerprint-match paper per-seed values against results/*.json.

Paper targets (from user):
  Run-A:
    Standard  [58.0, 59.4, 59.2]   (Tables 5, 6)
    OAMP      [58.2, 58.0, 59.6]   (Table 6)
    Naive-FP4 [61.2, 57.6, 58.4]   (Table 6)
  Run-D:
    Standard  [57.6, 57.8, 58.2]   (Table 15 caption; multiseed_20260414_120649 confirmed)
  Run-B: Standard mean=58.5 std=0.7    (Table 8 Llama row)
  Run-C: Standard mean=59.3 std=0.3    (Table 18)
  Run-E: seed42 Standard=58.6 / OAMP-full=60.4 (Table 17)
"""
import json
import math
import os
from glob import glob

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob(os.path.join(RESULTS_DIR, "*.json")))

TARGETS = {
    "Run-A / Standard  [58.0,59.4,59.2]": [58.0, 59.4, 59.2],
    "Run-A / OAMP      [58.2,58.0,59.6]": [58.2, 58.0, 59.6],
    "Run-A / Naive-FP4 [61.2,57.6,58.4]": [61.2, 57.6, 58.4],
    "Run-D / Standard  [57.6,57.8,58.2]": [57.6, 57.8, 58.2],
}

# mean/std targets (per-seed unknown) -> Run-B, Run-C
STAT_TARGETS = [
    ("Run-B  Standard mean=58.5 std=0.7", 58.5, 0.7, 0.15),
    ("Run-C  Standard mean=59.3 std=0.3", 59.3, 0.3, 0.15),
]

# single-seed targets -> Run-E
SINGLE_TARGETS = [
    ("Run-E  seed42 Standard=58.6",  42, "standard", 58.6),
    ("Run-E  seed42 OAMP-full=60.4", 42, "oamp",     60.4),
]

TOL = 0.15   # allow 0.10-0.15 pp fuzz on per-seed


def std(values, ddof):
    n = len(values)
    if n - ddof <= 0:
        return None
    m = sum(values) / n
    ss = sum((v - m) ** 2 for v in values)
    return math.sqrt(ss / (n - ddof))


def matches_multiset(cand, target, tol=TOL):
    if not cand or len(cand) != len(target):
        return False
    remaining = list(cand)
    for t in target:
        best = None
        best_d = tol + 1e-9
        for i, c in enumerate(remaining):
            d = abs(c - t)
            if d < best_d:
                best_d = d
                best = i
        if best is None:
            return False
        remaining.pop(best)
    return True


def iter_method_records(path):
    """Yield (filename, seeds, method, accs, losses, extra) for every method entry."""
    fn = os.path.basename(path)
    try:
        data = json.load(open(path))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    seeds = data.get("seeds") or []
    model = data.get("model")
    cfg = data.get("config") or {}
    if "results" in data and isinstance(data["results"], dict):
        for method, mres in data["results"].items():
            if not isinstance(mres, dict):
                continue
            accs = mres.get("accuracies") or mres.get("acc_per_seed")
            if accs is None or not isinstance(accs, list):
                continue
            accs_f = [float(a) for a in accs if isinstance(a, (int, float))]
            losses = mres.get("losses") or []
            yield fn, seeds, method, accs_f, losses, {"model": model, "config": cfg}
    # flat single-seed (test_a_fixbefore_*)
    seed = data.get("seed")
    for k, v in data.items():
        if isinstance(v, (int, float)) and k.endswith("_accuracy"):
            method = k[: -len("_accuracy")]
            acc = v * 100.0 if v <= 1.5 else v
            yield fn, [seed] if seed is not None else [], method, [acc], [], {"model": model, "config": cfg}


ALL_RECORDS = []
for p in FILES:
    for rec in iter_method_records(p):
        ALL_RECORDS.append(rec)

print(f"# scanned {len(FILES)} files, {len(ALL_RECORDS)} method records\n")

# per-seed multiset matching
print("=" * 78)
print("PER-SEED MULTISET MATCHES  (tol=+-{:.2f} pp)".format(TOL))
print("=" * 78)
for label, target in TARGETS.items():
    print(f"\n[{label}]")
    hits = 0
    for fn, seeds, method, accs, losses, extra in ALL_RECORDS:
        if matches_multiset(accs, target):
            hits += 1
            s0 = std(accs, 0)
            s1 = std(accs, 1)
            print(f"  MATCH  {fn}")
            print(f"         seeds={seeds}  method={method}")
            print(f"         accs={accs}  std_ddof0={s0:.3f}  std_ddof1={s1:.3f}")
    if hits == 0:
        # Show near-misses (single-value distance <= 0.3 for any element)
        print("  (no exact multiset match)  near-misses:")
        for fn, seeds, method, accs, losses, extra in ALL_RECORDS:
            if not accs or len(accs) != len(target):
                continue
            # if sorted diff <= 0.4 for each element in sorted order
            a = sorted(accs)
            t = sorted(target)
            d = [abs(x - y) for x, y in zip(a, t)]
            if max(d) <= 0.5:
                print(f"    ~{fn}  method={method}  accs={accs}  maxdiff={max(d):.2f}")

# mean/std matching (Run-B, Run-C)
print("\n" + "=" * 78)
print("MEAN/STD MATCHES  (Run-B, Run-C)")
print("=" * 78)
for label, m_target, s_target, tol in STAT_TARGETS:
    print(f"\n[{label}]")
    hits = 0
    for fn, seeds, method, accs, losses, extra in ALL_RECORDS:
        if len(accs) < 2:
            continue
        m = sum(accs) / len(accs)
        s0 = std(accs, 0)
        s1 = std(accs, 1)
        # match if mean close AND either ddof matches std_target
        if abs(m - m_target) <= tol and (
            (s0 is not None and abs(s0 - s_target) <= 0.2)
            or (s1 is not None and abs(s1 - s_target) <= 0.2)
        ):
            hits += 1
            print(f"  MATCH  {fn}")
            print(f"         seeds={seeds}  method={method}")
            print(f"         mean={m:.3f}  s0={s0:.3f}  s1={s1:.3f}  accs={accs}")
    if hits == 0:
        print("  (no exact mean/std match)  candidates with mean-only match:")
        for fn, seeds, method, accs, losses, extra in ALL_RECORDS:
            if len(accs) < 2:
                continue
            m = sum(accs) / len(accs)
            if abs(m - m_target) <= 0.3:
                s0 = std(accs, 0); s1 = std(accs, 1)
                print(f"    ~{fn}  method={method}  mean={m:.3f} s0={s0:.3f} s1={s1:.3f} accs={accs}")

# seed42 single-value (Run-E)
print("\n" + "=" * 78)
print("SEED-42 SINGLE-VALUE MATCHES  (Run-E)")
print("=" * 78)
for label, seed_val, method_needle, target_val in SINGLE_TARGETS:
    print(f"\n[{label}]")
    for fn, seeds, method, accs, losses, extra in ALL_RECORDS:
        if not accs:
            continue
        # find element at position of seed42
        idx = None
        if seed_val in seeds:
            idx = seeds.index(seed_val)
        elif len(seeds) == 1 and seeds[0] == seed_val:
            idx = 0
        # allow single-seed flat records
        if idx is None:
            continue
        if idx >= len(accs):
            continue
        v = accs[idx]
        if method_needle not in method.lower():
            continue
        if abs(v - target_val) <= 0.3:
            print(f"  MATCH  {fn}  method={method}  seed42_acc={v:.3f}  accs={accs}")

print()
