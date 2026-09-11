"""Analyse the downstream multiple-choice results against the pre-registration.

The rules this reads by are the ones written into scripts/run_mc_chain.sh before
the first cell ran, and it refuses to go beyond them: MMLU is the primary
metric, the correlation inside arm A is the primary test, arm D against arm C is
the positive control, and the other five tasks are described without a verdict.

No test here declares a result significant. Every comparison prints its
statistic and its p-value and stops there, which is this project's reporting
standard -- a p-value is a number the reader weighs, not a threshold the author
crosses on their behalf.

Tests are exact where the sample allows it. Spearman on eight points has 40,320
permutations and Mann-Whitney on four against four has 70 splits, so both are
enumerated rather than approximated; larger samples fall back to a Monte Carlo
permutation with a fixed seed, which is reported as such.

Usage:  python scripts/analyze_mc.py [--task mmlu]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import itertools
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 16.65                      # pre-registered; see scripts/run_mc_chain.sh
PRIMARY = "mmlu"
CONFIRMATORY = ("hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande")


def load_runs():
    """arm/seed/WikiText for every adapter, from the same collector the appendix uses."""
    spec = importlib.util.spec_from_file_location(
        "mk", os.path.join(ROOT, "scripts", "make_appendix_tables.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    return {ap: v for ap, v in mk.collect().items() if v["member"]}


def load_mc():
    """tag -> {task: acc}, plus the adapter path each tag was scored on."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "mc_downstream",
                                              "mc__*.json"))):
        d = json.load(open(path))
        tag = os.path.basename(path)[4:].rsplit("__", 1)[0]
        task = os.path.basename(path).rsplit("__", 1)[1][:-5]
        r = d["tasks"][task]
        e = out.setdefault(tag, {"adapter": d.get("adapter_path"), "tasks": {}})
        # Recomputed from the counts rather than read from the stored percentage,
        # which is the same rule the GSM8K accuracies are read under.
        e["tasks"][task] = {"acc": 100.0 * r["n_correct_sum"] / r["n"],
                            "acc_norm": 100.0 * r["n_correct_mean"] / r["n"],
                            "n": r["n"], "peak": r.get("peak_reserved_gb"),
                            "hash": r.get("items_hash")}
    return out


# ------------------------------------------------------------------ statistics

def _ranks(xs):
    """Average ranks, so ties are handled rather than broken arbitrarily."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def spearman_p(x, y, max_exact=200_000, n_mc=200_000, seed=20260907):
    """Two-tailed permutation p. Exact when the sample allows enumeration."""
    rho = spearman(x, y)
    n = len(x)
    fact = 1
    for k in range(2, n + 1):
        fact *= k
    if fact <= max_exact:
        hits = sum(1 for p in itertools.permutations(y)
                   if abs(spearman(x, list(p))) >= abs(rho) - 1e-12)
        return rho, hits / fact, f"exact, {fact:,} permutations"
    rnd = random.Random(seed)
    yy = list(y)
    hits = 0
    for _ in range(n_mc):
        rnd.shuffle(yy)
        if abs(spearman(x, yy)) >= abs(rho) - 1e-12:
            hits += 1
    return rho, (hits + 1) / (n_mc + 1), f"Monte Carlo, {n_mc:,} shuffles, seed {seed}"


def mannwhitney_exact(a, b):
    """Two-tailed exact Mann-Whitney. Returns (U, p, description)."""
    U = sum(1 for x in a for y in b if x > y) + \
        0.5 * sum(1 for x in a for y in b if x == y)
    pool = list(a) + list(b)
    n = len(a)
    tot = hit = 0
    for combo in itertools.combinations(range(len(pool)), n):
        aa = [pool[i] for i in combo]
        bb = [pool[i] for i in range(len(pool)) if i not in combo]
        u = sum(1 for x in aa for y in bb if x > y) + \
            0.5 * sum(1 for x in aa for y in bb if x == y)
        tot += 1
        if abs(u - len(a) * len(b) / 2) >= abs(U - len(a) * len(b) / 2) - 1e-12:
            hit += 1
    return U, hit / tot, f"exact, {tot} splits"


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="report one task only")
    a = ap.parse_args()

    runs, mc = load_runs(), load_mc()
    by_tag = {}
    for tag, e in mc.items():
        if tag == "base":
            by_tag[tag] = {"arm": "base", "seed": None, "wikitext": None,
                           "tasks": e["tasks"]}
            continue
        key = (e["adapter"] or "").replace("/app/HMA_Project/", "")
        v = runs.get(key)
        if v is None:
            print(f"  !! {tag}: adapter not in the run table, skipped"); continue
        by_tag[tag] = {"arm": v["arm"], "seed": v["seed"],
                       "wikitext": v["wikitext"], "tasks": e["tasks"]}

    tasks = [a.task] if a.task else \
        sorted({t for v in by_tag.values() for t in v["tasks"]},
               key=lambda t: (t != PRIMARY, t))

    for task in tasks:
        have = {k: v for k, v in by_tag.items() if task in v["tasks"]}
        if not have:
            continue
        role = "PRIMARY METRIC" if task == PRIMARY else "confirmatory (described, no verdict)"
        print(f"\n{'='*72}\n{task}   --   {role}\n{'='*72}")
        base = have.get("base", {}).get("tasks", {}).get(task, {}).get("acc")
        if base is not None:
            print(f"  base (no adapter): {base:.2f}")

        # arm summaries
        print(f"\n  {'arm':5s} {'n':>3s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'max':>7s}"
              f"{'   vs base' if base is not None else ''}")
        arms = {}
        for arm in "ABCDEF":
            vals = [v["tasks"][task]["acc"] for v in have.values() if v["arm"] == arm]
            if not vals:
                continue
            arms[arm] = vals
            sd = st.stdev(vals) if len(vals) > 1 else float("nan")
            d = f"   {st.mean(vals)-base:+.2f}" if base is not None else ""
            print(f"  {arm:5s} {len(vals):3d} {st.mean(vals):7.2f} {sd:6.3f} "
                  f"{min(vals):7.2f} {max(vals):7.2f}{d}")

        # primary test: correlation inside arm A
        A = [(v["wikitext"], v["tasks"][task]["acc"], v["seed"])
             for v in have.values() if v["arm"] == "A"]
        if len(A) >= 4:
            x = [p[0] for p in A]; y = [p[1] for p in A]
            rho, p, how = spearman_p(x, y)
            print(f"\n  [primary] Spearman(WikiText ppl, {task}) inside arm A, n={len(A)}")
            print(f"            rho = {rho:+.3f}   p = {p:.4f}   ({how})")
            print(f"            a negative rho is the predicted direction: higher "
                  f"perplexity, lower accuracy")

            dmg = [p_[1] for p_ in A if p_[0] > TAU]
            safe = [p_[1] for p_ in A if p_[0] <= TAU]
            if len(dmg) >= 2 and len(safe) >= 2:
                U, p2, how2 = mannwhitney_exact(dmg, safe)
                print(f"\n  [secondary] damaged (n={len(dmg)}) vs safe (n={len(safe)}) "
                      f"at tau={TAU}")
                print(f"            {st.mean(dmg):.2f} vs {st.mean(safe):.2f}   "
                      f"difference {st.mean(dmg)-st.mean(safe):+.2f}")
                print(f"            Mann-Whitney U = {U:g}   p = {p2:.4f}   ({how2})")

        # correlation across every arm, which is the stronger claim
        allpts = [(v["wikitext"], v["tasks"][task]["acc"])
                  for v in have.values() if v["wikitext"] is not None]
        if len(allpts) >= 6:
            rho, p, how = spearman_p([q[0] for q in allpts], [q[1] for q in allpts])
            print(f"\n  [across arms] Spearman over all {len(allpts)} adapters: "
                  f"rho = {rho:+.3f}   p = {p:.4f}   ({how})")

        # positive control
        if "D" in arms and "C" in arms:
            print(f"\n  [positive control] arm D vs arm C")
            print(f"            D {st.mean(arms['D']):.2f} (n={len(arms['D'])})   "
                  f"C {st.mean(arms['C']):.2f} (n={len(arms['C'])})   "
                  f"difference {st.mean(arms['D'])-st.mean(arms['C']):+.2f}")
            if len(arms["D"]) >= 2 and len(arms["C"]) >= 2:
                U, p3, how3 = mannwhitney_exact(arms["D"], arms["C"])
                print(f"            Mann-Whitney U = {U:g}   p = {p3:.4f}   ({how3})")
            else:
                print(f"            n is too small for a test; the comparison is "
                      f"descriptive")
            print(f"            if D does not separate from C, a null inside arm A "
                  f"is a limit of this metric")

    print(f"\n{'='*72}")
    print(f"cells present: {sum(len(v['tasks']) for v in by_tag.values())}")
    for task in sorted({t for v in by_tag.values() for t in v['tasks']}):
        n = sum(1 for v in by_tag.values() if task in v['tasks'])
        print(f"  {task:16s} {n:2d} / 22 adapters")


if __name__ == "__main__":
    main()
