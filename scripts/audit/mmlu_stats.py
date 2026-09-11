#!/usr/bin/env python3
"""MMLU: the pre-registered primary comparison and the MMLU cells of tab:levels.

The tests are the ones named in the header of scripts/run_mc_chain.sh (commit
27ae5b6, 2026-09-07 17:05:43 UTC), applied to results/mc_mmlu/mmlu__*.json
with the arm and the WikiText-2 perplexity of every adapter resolved by
_check_levels.collect_runs(), so arm membership and the run-level Mantel test
carry exactly the definitions the printed table uses.

Arm A, eight runs (available now):
  primary      Spearman(WikiText-2 ppl, MMLU accuracy), exact two-sided over
               8! = 40,320 relabellings; the one-sided p in the predicted
               direction (higher perplexity, lower accuracy) is printed as well
  split        the four runs above tau = 16.65 are the four above the median
  secondary    mean accuracy, damaged minus safe: pooled-variance t interval
               (df 6), the exact distribution of the difference over the
               C(8,4) = 70 splits, and the exact Mann-Whitney on the same 70
  deficit      the most negative end of the t interval as a fraction of the
               fine-tuning cost, base minus the arm-A mean
  sign test    how many of the eight runs score below the base model; exact
               binomial with p = 1/2
  item churn   fraction of the 14,042 items on which two runs answer
               differently, over the 28 pairs; the six within-damaged pairs
               against the six within-safe pairs, and the twelve within-label
               pairs against the sixteen cross-label pairs, each exact over
               the 70 labellings; and the run-level Mantel against |dppl|
  tab:levels   run column: Mantel(|dacc| x |dppl|), exact two-sided over 8!

Arms B, C, D (run when their JSONs are present, otherwise reported pending):
  tab:levels   arm column: variance ratio of accuracy A vs B, exact two-sided
               over C(16,8) = 12,870; item churn A vs B on the same splits
  control      arm D against arm C, exact Mann-Whitney

Writes results/audit/mmlu_stats_<date>.json. Needs numpy (through
_check_levels); scipy only for the t quantile, with the df-6 value as fallback.

    docker exec -w /app/HMA_Project hma-container python3 scripts/audit/mmlu_stats.py
"""
import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _check_levels as L  # noqa: E402

ROOT = L.ROOT
TAU = 16.65
PRESPEC_COMMIT = "27ae5b6"
LAUNCH_LOG = "results/mc_mmlu_launch.log"


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return {6: 2.446912}[df]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_mmlu(runs):
    cells = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "mc_mmlu", "mmlu__*.json"))):
        d = json.load(open(path))
        t = d["tasks"]["mmlu"]
        tag = os.path.basename(path)[6:-5]
        ap = L.norm(d.get("adapter_path"))
        r = runs.get(ap) if ap else None
        if ap and r is None:
            print(f"  !! {tag}: adapter {ap} is not one of the 32 runs, skipped")
            continue
        cells[tag] = {
            "file": os.path.relpath(path, ROOT), "arm": r["arm"] if r else "base",
            "seed": r["seed"] if r else None, "ppl": r["ppl"] if r else None,
            "n": t["n"], "n_correct": t["n_correct_sum"], "acc": 100.0 * t["n_correct_sum"] / t["n"],
            "items_hash": t.get("items_hash"),
            "pred": np.array([s["pred_sum"] for s in t["per_sample"]]),
            "gold": np.array([s["gold"] for s in t["per_sample"]]),
        }
    assert len({c["items_hash"] for c in cells.values()}) == 1, "items_hash differs across cells"
    assert len({c["n"] for c in cells.values()}) == 1
    g0 = next(iter(cells.values()))["gold"]
    assert all((c["gold"] == g0).all() for c in cells.values()), "gold labels differ across cells"
    for c in cells.values():
        assert int((c["pred"] == c["gold"]).sum()) == c["n_correct"], "per-item records disagree with n_correct_sum"
    return cells


def churn_matrix(cs):
    n = len(cs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(np.mean(cs[i]["pred"] != cs[j]["pred"]))
    return M


def exact_spearman(x, y):
    obs = L.spearman(x, y)
    hits2 = hits1 = total = 0
    for p in itertools.permutations(range(len(y))):
        r = L.spearman(x, [y[k] for k in p])
        total += 1
        if abs(r) >= abs(obs) - 1e-12:
            hits2 += 1
        if r <= obs + 1e-12:          # predicted direction: negative
            hits1 += 1
    return obs, hits2 / total, hits1 / total, total


def split_distribution(vals, k):
    """Difference of means, first k against the rest, over every k-subset."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    out = []
    for cb in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(cb)] = True
        out.append(vals[m].mean() - vals[~m].mean())
    return np.array(out)


def mann_whitney_exact(a, b):
    U = sum(1.0 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)
    pool = list(a) + list(b)
    na, nb = len(a), len(b)
    c = na * nb / 2.0
    hits = total = 0
    for cb in itertools.combinations(range(len(pool)), na):
        aa = [pool[i] for i in cb]
        bb = [pool[i] for i in range(len(pool)) if i not in cb]
        u = sum(1.0 for x in aa for y in bb if x > y) + 0.5 * sum(1 for x in aa for y in bb if x == y)
        total += 1
        if abs(u - c) >= abs(U - c) - 1e-12:
            hits += 1
    return U, hits / total, total


def binom_two_sided(k, n):
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return sum(p for p in pk if p <= pk[k] + 1e-15)


def prespec_record():
    rec = {"commit": PRESPEC_COMMIT}
    try:
        rec["commit_time"] = subprocess.run(["git", "show", "-s", "--format=%cI", PRESPEC_COMMIT], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        rec["commit_time"] = "(git history not available)"
    log = os.path.join(ROOT, LAUNCH_LOG)
    if os.path.exists(log):
        for line in open(log):
            if "GPU lock acquired" in line:
                rec["mmlu_chain_started"] = line.strip()
            if re.search(r"\b1/9 base\b", line):
                rec["first_mmlu_cell_done"] = line.strip()
            if re.search(r"\b2/9 A_seed42\b", line):
                rec["first_arm_a_cell_done"] = line.strip()
    return rec


def r4(x):
    return None if x is None else round(float(x), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y%m%d", time.gmtime()))
    args = ap.parse_args()

    runs = L.collect_runs()
    cells = load_mmlu(runs)
    out = {"inputs": {c["file"]: sha256(os.path.join(ROOT, c["file"])) for c in cells.values()},
           "n_items": next(iter(cells.values()))["n"], "items_hash": next(iter(cells.values()))["items_hash"],
           "tau": TAU, "prespecification": prespec_record(), "cells": {}, "arm_A": {}, "arm_level": {}}
    present = {a: sum(1 for c in cells.values() if c["arm"] == a) for a in ["base", "A", "B", "C", "D", "E", "F"]}
    print(f"cells: {', '.join(f'{a}={n}' for a, n in present.items() if n)}; "
          f"{out['n_items']} items, items_hash {out['items_hash']} in every cell")
    print(f"pre-specification: commit {PRESPEC_COMMIT} {out['prespecification'].get('commit_time')}; "
          + "; ".join(v for k, v in out["prespecification"].items() if k not in ("commit", "commit_time")))
    for tag, c in cells.items():
        out["cells"][tag] = {k: (r4(c[k]) if isinstance(c[k], float) else c[k]) for k in
                             ("file", "arm", "seed", "ppl", "n", "n_correct", "acc")}
    base = cells.get("base")

    # ---------------------------------------------------------------- arm A
    A = sorted([c for c in cells.values() if c["arm"] == "A"], key=lambda c: c["ppl"])
    if len(A) != 8:
        print(f"arm A: {len(A)}/8 cells present, arm-A statistics skipped")
    else:
        ppl = [c["ppl"] for c in A]
        acc = [c["acc"] for c in A]
        dmg = [c["ppl"] > TAU for c in A]
        print(f"\narm A, by perplexity   (base {base['acc']:.4f} = {base['n_correct']}/{base['n']})" if base else "\narm A")
        print(f"  {'seed':>5s} {'ppl':>8s} {'MMLU':>8s} {'correct':>8s}  label")
        for c, d in zip(A, dmg):
            print(f"  {c['seed']:>5} {c['ppl']:8.4f} {c['acc']:8.4f} {c['n_correct']:>8d}  {'damaged' if d else 'safe'}")
        rho, p2, p1, tot = exact_spearman(ppl, acc)
        print(f"\n  [primary] Spearman(ppl, acc) rho = {rho:+.4f}; two-sided p = {p2:.4f}, "
              f"one-sided (rho <= observed) p = {p1:.4f}; exact over {tot:,} relabellings")
        med = (ppl[3] + ppl[4]) / 2
        above_med = [c["seed"] for c in A if c["ppl"] > med]
        above_tau = [c["seed"] for c, d in zip(A, dmg) if d]
        same = set(above_med) == set(above_tau) and len(above_tau) == 4
        print(f"  [split] median ppl {med:.4f}: above median {above_med}; above tau {above_tau}; identical: {same}")
        a_d = [c["acc"] for c, d in zip(A, dmg) if d]
        a_s = [c["acc"] for c, d in zip(A, dmg) if not d]
        diff = np.mean(a_d) - np.mean(a_s)
        sp = math.sqrt((L.var(a_d) * 3 + L.var(a_s) * 3) / 6)
        se = sp * math.sqrt(1 / 4 + 1 / 4)
        tq = t975(6)
        ci = (diff - tq * se, diff + tq * se)
        dist = split_distribution(acc, 4)                      # damaged-labelled first four of each split
        p_perm2 = float(np.mean(np.abs(dist) >= abs(diff) - 1e-12))
        p_perm1 = float(np.mean(dist <= diff + 1e-12))         # predicted: damaged below safe
        lo, hi = np.sort(dist)[[1, 68]]                        # 2nd and 69th of 70: central 95.7 %
        U, p_mw, n_mw = mann_whitney_exact(a_d, a_s)
        print(f"  [secondary] damaged {np.mean(a_d):.4f} (n=4) vs safe {np.mean(a_s):.4f} (n=4): "
              f"difference {diff:+.4f} pp")
        print(f"     t interval (pooled sd {sp:.4f}, df 6, t = {tq:.4f}): [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"     70-split distribution of the difference: p two-sided {p_perm2:.4f} "
              f"({int(round(p_perm2*70))}/70), one-sided (damaged below safe) {p_perm1:.4f}; "
              f"2nd..69th of 70: [{lo:+.4f}, {hi:+.4f}]")
        print(f"     Mann-Whitney U = {U:g}, exact two-sided p = {p_mw:.4f} ({n_mw} splits)")
        cost = base["acc"] - np.mean(acc) if base else None
        if cost is not None:
            print(f"  [deficit] fine-tuning cost base - mean(A) = {base['acc']:.4f} - {np.mean(acc):.4f} = {cost:.4f} pp; "
                  f"largest damaged-run deficit inside the t interval {-ci[0]:.4f} pp = {(-ci[0])/cost:.4f} of the cost; "
                  f"observed difference {diff/cost:+.4f} of the cost")
        if base:
            below = sum(1 for c in A if c["acc"] < base["acc"])
            print(f"  [sign test] {below} of 8 arm-A runs below base; exact binomial (p = 1/2) "
                  f"two-sided {binom_two_sided(below, 8):.4f}, one-sided {0.5**8 * sum(math.comb(8, i) for i in range(below, 9)):.4f}")
        # item churn
        M = churn_matrix(A)
        I, J = np.triu_indices(8, 1)
        pairs = M[I, J]
        lab = np.array(dmg)
        within = [M[i, j] for i, j in zip(I, J) if lab[i] == lab[j]]
        cross = [M[i, j] for i, j in zip(I, J) if lab[i] != lab[j]]
        obs_c = np.mean(within) - np.mean(cross)
        hits = tot = 0
        for cb in itertools.combinations(range(8), 4):
            l2 = np.zeros(8, bool)
            l2[list(cb)] = True
            w = [M[i, j] for i, j in zip(I, J) if l2[i] == l2[j]]
            x = [M[i, j] for i, j in zip(I, J) if l2[i] != l2[j]]
            tot += 1
            if abs(np.mean(w) - np.mean(x)) >= abs(obs_c) - 1e-12:
                hits += 1
        wd = [M[i, j] for i, j in zip(I, J) if lab[i] and lab[j]]
        ws = [M[i, j] for i, j in zip(I, J) if not lab[i] and not lab[j]]
        obs_dw = np.mean(wd) - np.mean(ws)
        hits_dw = 0
        for cb in itertools.combinations(range(8), 4):
            l2 = np.zeros(8, bool)
            l2[list(cb)] = True
            d_ = [M[i, j] for i, j in zip(I, J) if l2[i] and l2[j]]
            s_ = [M[i, j] for i, j in zip(I, J) if not l2[i] and not l2[j]]
            if abs(np.mean(d_) - np.mean(s_)) >= abs(obs_dw) - 1e-12:
                hits_dw += 1
        rho_c, h_c, t_c = L.mantel(M, ppl)
        print(f"  [item churn] 28 pairs: mean {100*pairs.mean():.4f} %, range {100*pairs.min():.4f}-{100*pairs.max():.4f} %; "
              f"within damaged {100*np.mean(wd):.4f} %, within safe {100*np.mean(ws):.4f} %, cross {100*np.mean(cross):.4f} %")
        print(f"     within damaged minus within safe {100*obs_dw:+.4f} pp, exact two-sided p = {hits_dw/tot:.4f} ({hits_dw}/{tot} labellings)")
        print(f"     within-label minus cross-label {100*obs_c:+.4f} pp, exact two-sided p = {hits/tot:.4f} ({hits}/{tot} labellings)")
        print(f"     Mantel churn x |dppl|: rho {rho_c:+.4f}, p = {h_c/t_c:.4f} ({h_c}/{t_c})")
        # tab:levels run column
        dacc = np.abs(np.subtract.outer(acc, acc))
        rho_m, h_m, t_m = L.mantel(dacc, ppl)
        print(f"  [tab:levels run column, MMLU] Mantel |dacc| x |dppl|: rho {rho_m:+.4f}, "
              f"exact two-sided p = {h_m/t_m:.4f} ({h_m}/{t_m})")
        out["arm_A"] = {
            "seeds_by_ppl": [c["seed"] for c in A], "ppl": [r4(v) for v in ppl], "acc": [r4(v) for v in acc],
            "damaged": [bool(d) for d in dmg],
            "primary_spearman": {"rho": r4(rho), "p_two_sided": r4(p2), "p_one_sided_negative": r4(p1), "relabellings": tot},
            "median_split_equals_tau_split": bool(same), "median_ppl": r4(med),
            "secondary": {"mean_damaged": r4(np.mean(a_d)), "mean_safe": r4(np.mean(a_s)), "difference": r4(diff),
                          "pooled_sd": r4(sp), "t_df6": r4(tq), "ci95_t": [r4(ci[0]), r4(ci[1])],
                          "perm70_p_two_sided": r4(p_perm2), "perm70_p_one_sided": r4(p_perm1),
                          "perm70_2nd_69th": [r4(lo), r4(hi)],
                          "mann_whitney_U": U, "mann_whitney_p": r4(p_mw)},
            "deficit": None if cost is None else {"fine_tuning_cost_pp": r4(cost), "ci_lower_deficit_pp": r4(-ci[0]),
                                                  "deficit_over_cost": r4(-ci[0] / cost), "observed_over_cost": r4(diff / cost)},
            "sign_test_vs_base": None if not base else {"below_base": below, "n": 8,
                                                        "p_two_sided": r4(binom_two_sided(below, 8))},
            "item_churn": {"mean": r4(pairs.mean()), "min": r4(pairs.min()), "max": r4(pairs.max()),
                           "within_damaged": r4(np.mean(wd)), "within_safe": r4(np.mean(ws)), "cross": r4(np.mean(cross)),
                           "damaged_minus_safe": r4(obs_dw), "p_damaged_minus_safe_70": r4(hits_dw / tot),
                           "within_minus_cross": r4(obs_c), "p_within_minus_cross_70": r4(hits / tot),
                           "mantel_rho": r4(rho_c), "mantel_p": r4(h_c / t_c)},
            "levels_run_column": {"mantel_rho": r4(rho_m), "p_two_sided": r4(h_m / t_m), "hits": h_m, "relabellings": t_m},
        }

    # ------------------------------------------------------------ arm level
    B = sorted([c for c in cells.values() if c["arm"] == "B"], key=lambda c: c["ppl"])
    print()
    if len(A) == 8 and len(B) == 8:
        accA, accB = [c["acc"] for c in A], [c["acc"] for c in B]
        h, t = L.perm_var_ratio(accA, accB)
        M16 = churn_matrix(A + B)
        h2, t2 = L.perm_pair_mean(M16, list(range(8)), list(range(8, 16)))
        print(f"  [tab:levels arm column, MMLU] sd A {math.sqrt(L.var(accA)):.4f} vs B {math.sqrt(L.var(accB)):.4f}, "
              f"variance ratio {L.var(accA)/L.var(accB):.4f}, exact two-sided p = {h/t:.4f} ({h}/{t})")
        print(f"  [item churn arm level] mean pair churn A {100*L.within_mean(M16, range(8)):.4f} % vs "
              f"B {100*L.within_mean(M16, range(8, 16)):.4f} %, exact two-sided p = {h2/t2:.4f} ({h2}/{t2})")
        out["arm_level"] = {"sd_A": r4(math.sqrt(L.var(accA))), "sd_B": r4(math.sqrt(L.var(accB))),
                            "var_ratio": r4(L.var(accA) / L.var(accB)), "p_var_ratio": r4(h / t),
                            "churn_A": r4(L.within_mean(M16, range(8))), "churn_B": r4(L.within_mean(M16, range(8, 16))),
                            "p_churn": r4(h2 / t2), "mean_A": r4(np.mean(accA)), "mean_B": r4(np.mean(accB))}
    else:
        print(f"  [tab:levels arm column, MMLU] pending: arm B {len(B)}/8 cells")
    C = [c["acc"] for c in cells.values() if c["arm"] == "C"]
    D = [c["acc"] for c in cells.values() if c["arm"] == "D"]
    if len(C) == 2 and len(D) == 3:
        U, p, n = mann_whitney_exact(D, C)
        print(f"  [positive control] D {np.mean(D):.4f} (n=3) vs C {np.mean(C):.4f} (n=2): difference {np.mean(D)-np.mean(C):+.4f}; "
              f"Mann-Whitney U = {U:g}, exact two-sided p = {p:.4f} ({n} splits)")
        out["positive_control"] = {"mean_D": r4(np.mean(D)), "mean_C": r4(np.mean(C)), "U": U, "p": r4(p)}
    else:
        print(f"  [positive control] pending: D {len(D)}/3, C {len(C)}/2 cells"
              + (f"; D so far {', '.join(f'{v:.4f}' for v in D)}" if D else ""))

    dest = os.path.join(ROOT, "results", "audit", f"mmlu_stats_{args.date}.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    main()
