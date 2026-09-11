#!/usr/bin/env python3
"""Aggregate axis 2 A_4DFP4 seed-instability results (2026-08-31).

Reads all A_4DFP4 PPL JSONs from both the earlier 3 seeds and the axis-2
5 seeds, computes per-corpus damage counts against the pre-registered
threshold, and does the cross-corpus rank-concordance check the user asked
for after axis 2 completes.

Pre-registered threshold (locked before axis-2 launch):
    Standard mean + 3 * B(4D FP8) sd  =  16.08 + 3*0.19  =  16.65
    A seed with WikiText PPL > 16.65 counts as damaged.

Usage:
    python3 scripts/analyze_axis2.py
"""
import json
import glob
import os
from itertools import combinations
from statistics import mean, stdev

ROOT = '/home/yedam/HMA/HMA_Project'
THRESHOLD_WT = 16.65

# ---- Locate all A_4DFP4 PPL JSONs (both original and axis-2) ----
def load_a_ppls():
    """Returns dict[seed] = {'gsm': ..., 'wiki': ..., 'narr': ..., 'gov': ...}."""
    out = {}
    # 1. Original 3 seeds — gsm/wiki from ppl_eval, narr/gov from ppl_ood_extra
    for f in glob.glob(os.path.join(ROOT, 'results/ppl_eval/ppl__A_4DFP4__seed*.json')):
        seed = int(os.path.basename(f).replace('ppl__A_4DFP4__seed','').replace('.json',''))
        d = json.load(open(f))
        out.setdefault(seed, {})
        out[seed]['gsm']  = d.get('gsm8k_test_ppl')
        out[seed]['wiki'] = d.get('wikitext2_ppl')
    for f in glob.glob(os.path.join(ROOT, 'results/ppl_ood_extra/ppl_ood__A_4DFP4__seed*.json')):
        seed = int(os.path.basename(f).replace('ppl_ood__A_4DFP4__seed','').replace('.json',''))
        d = json.load(open(f))
        out.setdefault(seed, {})
        out[seed]['narr'] = d.get('narrativeqa_ppl')
        out[seed]['gov']  = d.get('govreport_ppl')
    # 2. Axis-2 five seeds — all four corpora in one file
    for f in glob.glob(os.path.join(ROOT, 'results/ppl_axis2/ppl_axis2__A_4DFP4__seed*.json')):
        seed = int(os.path.basename(f).replace('ppl_axis2__A_4DFP4__seed','').replace('.json',''))
        d = json.load(open(f))
        out.setdefault(seed, {})
        out[seed]['gsm']  = d.get('gsm8k_test_ppl')
        out[seed]['wiki'] = d.get('wikitext2_ppl')
        out[seed]['narr'] = d.get('narrativeqa_ppl')
        out[seed]['gov']  = d.get('govreport_ppl')
    return out


def main():
    print("=" * 90)
    print(f"Axis 2 aggregate — pre-registered threshold WikiText PPL > {THRESHOLD_WT}")
    print("=" * 90)

    ppls = load_a_ppls()
    if not ppls:
        print("  no A_4DFP4 PPL JSONs found yet")
        return

    seeds = sorted(ppls.keys())
    print()
    print(f"  {'seed':>5s}  {'GSM8K':>7s}  {'WikiText':>9s}  {'NarrQA':>8s}  {'GovRep':>7s}  {'WT damaged?':<11s}")
    print("-" * 90)
    for s in seeds:
        r = ppls[s]
        wt = r.get('wiki')
        dmg = 'DAMAGED' if wt is not None and wt > THRESHOLD_WT else 'safe'
        def f(v): return f'{v:>7.4f}' if v is not None else '   ?'
        print(f"  {s:>5d}  {f(r.get('gsm'))}  {f(wt)}  {f(r.get('narr'))}  {f(r.get('gov'))}  {dmg}")

    # ---- Damage count per corpus (WikiText: pre-registered + sensitivity sweep) ----
    B_wiki_sd = 0.19    # measured from B_4DFP8 3 seeds (user rounded 0.186→0.19)
    B_narr_sd = 0.101
    B_gov_sd  = 0.037
    C_wiki_mean = 16.08
    C_narr_mean = 26.62
    C_gov_mean  = 11.88
    thresholds = {
        'wiki': (C_wiki_mean + 3*B_wiki_sd, 'pre-registered'),
        'narr': (C_narr_mean + 3*B_narr_sd, 'descriptive (3σ B)'),
        'gov':  (C_gov_mean  + 3*B_gov_sd,  'descriptive (3σ B)'),
    }
    print()
    print("=" * 90)
    print("Damage count per corpus")
    print("=" * 90)
    print(f"  {'corpus':>8s}  {'threshold':>10s}  {'note':<20s}  damaged / total")
    for k, (th, note) in thresholds.items():
        vals = [(s, ppls[s].get(k)) for s in seeds if ppls[s].get(k) is not None]
        dmg = sum(1 for _, v in vals if v > th)
        dmg_seeds = [s for s, v in vals if v > th]
        print(f"  {k:>8s}  {th:>10.3f}  {note:<20s}  {dmg}/{len(vals)}   seeds: {dmg_seeds}")

    # ---- WikiText threshold sensitivity (2σ / 3σ / 4σ) ----
    print()
    print("=" * 90)
    print("WikiText threshold sensitivity — how robust is the damage count?")
    print("=" * 90)
    print(f"  Base: C_STD mean = {C_wiki_mean}, B_4DFP8 sd = {B_wiki_sd}")
    print()
    print(f"  {'kσ':>3s}  {'threshold':>10s}   damaged / total   seeds > threshold")
    print(f"  {'-'*3}  {'-'*10}   {'-'*15}   {'-'*30}")
    wiki_vals = [(s, ppls[s].get('wiki')) for s in seeds if ppls[s].get('wiki') is not None]
    for k_sigma in (2, 3, 4):
        th = C_wiki_mean + k_sigma * B_wiki_sd
        dmg_seeds = [(s, v) for s, v in wiki_vals if v > th]
        marker = '  ← pre-registered' if k_sigma == 3 else ''
        seed_str = ', '.join(f'{s}({v:.2f})' for s, v in dmg_seeds) or '—'
        print(f"  {k_sigma}σ   {th:>10.3f}   {len(dmg_seeds):>2d}/{len(wiki_vals):<2d}            {seed_str}{marker}")

    # ---- Cross-corpus rank concordance ----
    # Are corpora giving independent evidence, or is one seed's damage showing up
    # in every corpus? For n=8 this is a proper concordance analysis.
    print()
    print("=" * 90)
    print("Cross-corpus rank concordance (Kendall's tau)")
    print("=" * 90)
    print("  τ close to +1  =  corpora agree on seed order (redundant info)")
    print("  τ close to  0  =  independent per-corpus damage")
    print("  τ close to -1  =  anti-correlated (unlikely if all corpora reflect model quality)")
    print()
    print("  NOTE: for n=3 the null distribution puts p(τ=+1)=1/6=0.17 and")
    print("        p(τ=-1/3)=1/3, so τ is uninformative at that sample size.")
    print("        Meaningful thresholds require n>=6 (p(τ=+1)<0.01).")
    print()

    corpora = {'GSM8K':'gsm', 'WikiText':'wiki', 'NarrQA':'narr', 'GovRep':'gov'}
    complete_seeds = [s for s in seeds
                      if all(ppls[s].get(k) is not None for k in corpora.values())]
    if len(complete_seeds) < 3:
        print(f"  need ≥3 complete seeds; have {len(complete_seeds)}. Skipping.")
        return

    def rank_seeds_within_corpus(key):
        vs = [(s, ppls[s][key]) for s in complete_seeds]
        vs.sort(key=lambda kv: kv[1])
        return {s: i for i, (s, _) in enumerate(vs)}

    ranks = {name: rank_seeds_within_corpus(k) for name, k in corpora.items()}
    pairs = list(combinations(complete_seeds, 2))
    print(f"  n_seeds = {len(complete_seeds)}, {len(pairs)} seed pairs")
    print()
    # Two-sided p-value for τ from exact permutation test (small n)
    from math import factorial
    def exact_p_two_sided(n, tau_obs):
        from itertools import permutations
        # count fraction of permutations of (0..n-1) whose concordance with (0..n-1)
        # yields |τ| ≥ |τ_obs|.
        base = list(range(n))
        total = factorial(n)
        hits = 0
        for perm in permutations(base):
            c = d = 0
            for i, j in combinations(range(n), 2):
                di = base[i] - base[j]; dj = perm[i] - perm[j]
                if di * dj > 0: c += 1
                elif di * dj < 0: d += 1
            t = (c - d) / (n * (n - 1) / 2)
            if abs(t) >= abs(tau_obs) - 1e-12:
                hits += 1
        return hits / total

    print(f"  {'corpus A':>10s}  vs  {'corpus B':>10s}    τ   p(two-sided)  conc/disc")
    for (n1, r1), (n2, r2) in combinations(ranks.items(), 2):
        conc = sum(1 for a, b in pairs if (r1[a]-r1[b])*(r2[a]-r2[b]) > 0)
        disc = sum(1 for a, b in pairs if (r1[a]-r1[b])*(r2[a]-r2[b]) < 0)
        tau = (conc - disc) / len(pairs) if pairs else 0.0
        p = exact_p_two_sided(len(complete_seeds), tau)
        sig = '  *' if p < 0.05 else ''
        print(f"  {n1:>10s}  vs  {n2:>10s}   {tau:+5.2f}   p={p:.3f}{sig}   {conc}/{disc}")

    # ---- Damage overlap (which seeds are damaged in multiple corpora?) ----
    print()
    print("=" * 90)
    print("Cross-corpus damage overlap — is the same seed damaged everywhere?")
    print("=" * 90)
    print()
    damaged = {}
    for name, (k, tag) in zip(corpora.keys(), zip(corpora.values(),
                                                  ['pre-reg', '3σ', '3σ', '3σ'])):
        th = thresholds.get(k, (None, None))[0] if k in thresholds else None
        if th is None:
            # gsm doesn't have a threshold; use B_4DFP8 gsm sd 0.0013 for descriptive
            th = 2.558 + 3 * 0.0013
        damaged[name] = {s for s in complete_seeds if ppls[s][k] > th}
    for name, dset in damaged.items():
        print(f"  {name:>8s} damaged seeds: {sorted(dset)}")

    all_damaged = set(complete_seeds)
    for dset in damaged.values():
        all_damaged &= dset
    print()
    print(f"  Seeds damaged in ALL 4 corpora:  {sorted(all_damaged)}")

    # Union — seeds damaged in AT LEAST ONE corpus
    any_dmg = set()
    for dset in damaged.values():
        any_dmg |= dset
    print(f"  Seeds damaged in ≥1 corpus:      {sorted(any_dmg)}")


if __name__ == '__main__':
    main()
