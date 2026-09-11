#!/usr/bin/env python3
"""Recompute every cell of Table `tab:levels` from the primary result files.

The table reports each dispersion measurement at two levels, and its caption
fixes the test behind every cell. This script applies those definitions to the
stored results and prints the recomputed value next to the printed one, so the
caption can be checked against the numbers rather than trusted.

  arm column   Is the blockwise arm (A) more dispersed than the FP8 arm (B)?
               Exact two-sided permutation over all C(16, 8) = 12,870
               relabellings of the sixteen runs. Statistic:
                 - held-out perplexity, benchmark accuracy (mean over the four
                   benchmarks): variance ratio, tested on |log(s_A^2 / s_B^2)|
                 - item disagreement, subspace cosine, Frobenius distance:
                   |mean within-arm pair value (A) - mean within-arm pair value (B)|
  run column   Does the measurement identify which run inside arm A is damaged?
               Exact two-sided Mantel test over all 8! = 40,320 relabellings of
               the eight arm-A runs: Spearman correlation between each pair's
               value (item disagreement, subspace cosine, Frobenius distance,
               |difference in mean accuracy|) and the pair's absolute difference
               in held-out perplexity.
  footnote     Leave-one-out on the subspace Mantel test: drop one run, 7! = 5,040
               relabellings of the remaining seven.
  "no overlap in 88 pairs"  The within-arm pair sets of arms D, A, F, B and C
               (3 + 28 + 28 + 28 + 1); the [min, max] intervals are disjoint.

Sources, read directly:
  results/ppl_*/*.json                 WikiText-2 perplexity, arm resolution
                                       (method, body_encoding, pack_4d_mode),
                                       adapter path. Files tagged _rerun_ or
                                       _dup_ are second trainings of a seed and
                                       are excluded, as in the paper.
  results/mc_downstream/mc__*__*.json  per-item predictions (pred_sum) and
                                       n_correct_sum / n for the four benchmarks
  <adapter_path>/adapter_model.safetensors, adapter_config.json
                                       LoRA factors for the row-space principal
                                       angles and the Frobenius distances

Nothing here reads results/_wd or calls another script. Needs numpy, torch and
safetensors (the adapters are stored in bfloat16), so run it in the container:

    docker exec -w /app/HMA_Project hma-container python3 scripts/audit/_check_levels.py
"""
import glob
import itertools
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["arc_challenge", "arc_easy", "piqa", "winogrande"]
ARMS = {
    ("naive_fp4", "e2m1", "fp4"): "A",
    ("naive_fp4", "e2m1", "fp8"): "B",
    ("standard", None, None): "C",
    ("naive_fp4", "int4", "fp4"): "D",
    ("naive_fp4", "int4", "fp8"): "E",
    ("naive_fp4", "e2m1", "chan_int4"): "F",
}
# One adapter predates the three config fields; it is attributed by directory,
# exactly as Appendix D marks it with a double dagger.
BY_DIRECTORY = {"naive4bit_int4": "D"}
EXPECTED_N = {"A": 8, "B": 8, "C": 2, "D": 3, "E": 3, "F": 8}

# The printed table, 2026-09-08.
PAPER = {
    ("arm", "held-out perplexity"): 0.03,
    ("arm", "item-level behaviour"): 0.001,
    ("arm", "LoRA subspace direction"): 0.0002,
    ("arm", "LoRA update distance"): 0.25,
    ("arm", "benchmark accuracy"): 0.31,
    ("run", "item-level behaviour"): 0.21,
    ("run", "LoRA subspace direction"): 0.025,
    ("run", "LoRA update distance"): 0.22,
    ("run", "benchmark accuracy"): 0.93,
}
PAPER_LOO = "three of eight drops below 0.05 (0.004, 0.027, 0.042); the other five between 0.10 and 0.21"


def norm(p):
    return re.sub(r"^/app/HMA_Project/", "", (p or "").rstrip("/"))


# ----------------------------------------------------------------- collection

def collect_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "ppl_*", "*.json"))):
        base = os.path.basename(path)
        if "_rerun_" in base or "_dup_" in base:
            continue
        d = json.load(open(path))
        if d.get("wikitext2_ppl") is None:
            continue
        ap = norm(d.get("adapter_path"))
        if not ap:
            continue
        tc = d.get("training_config") or {}
        if tc:
            if "Llama-3.2-3B" not in (tc.get("model_id") or ""):
                continue
            if tc.get("n_train") != 7473 or tc.get("epochs") != 2:
                continue
            m = tc.get("method")
            key = ("standard", None, None) if m == "standard" else \
                (m, tc.get("body_encoding"), tc.get("pack_4d_mode"))
            arm = ARMS.get(key)
        else:
            arm = BY_DIRECTORY.get(os.path.basename(os.path.dirname(ap)))
        if arm is None or ap in runs:
            continue
        runs[ap] = {"arm": arm, "seed": d.get("seed"), "ppl": d["wikitext2_ppl"],
                    "ppl_file": os.path.relpath(path, ROOT)}
    counts = {a: sum(1 for r in runs.values() if r["arm"] == a) for a in EXPECTED_N}
    assert counts == EXPECTED_N, f"arm sizes {counts} != {EXPECTED_N}"
    return runs


def collect_mc(runs):
    by = {}
    for path in glob.glob(os.path.join(ROOT, "results", "mc_downstream", "mc__*__*.json")):
        d = json.load(open(path))
        task = os.path.basename(path).rsplit("__", 1)[1][:-5]
        by[(norm(d.get("adapter_path")), task)] = d["tasks"][task]
    hashes = {}
    for ap, r in runs.items():
        if r["arm"] not in ("A", "B"):
            continue
        preds, accs = [], []
        for t in TASKS:
            e = by.get((ap, t))
            assert e is not None, f"no {t} result for {ap}"
            preds += [s["pred_sum"] for s in e["per_sample"]]
            accs.append(100.0 * e["n_correct_sum"] / e["n"])
            hashes.setdefault(t, set()).add(e.get("items_hash"))
        r["pred"] = np.array(preds)
        r["acc"] = float(np.mean(accs))
    for t, h in hashes.items():
        assert len(h) == 1, f"item set differs across runs for {t}: {h}"
    n_items = len(next(r["pred"] for r in runs.values() if "pred" in r))
    return n_items


def load_lora(ap):
    import torch
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(ROOT, ap, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    A, B = {}, {}
    with safe_open(os.path.join(ROOT, ap, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if ".lora_A." in k:
                A[k.split(".lora_A.")[0]] = f.get_tensor(k).to(torch.float64).numpy()
            elif ".lora_B." in k:
                B[k.split(".lora_B.")[0]] = f.get_tensor(k).to(torch.float64).numpy()
    assert set(A) == set(B) and len(A) == 112, f"{ap}: {len(A)} adapter sites"
    return A, B, scale, cfg


def pair_matrices(runs):
    """Row-space principal-angle cosine and Frobenius distance, all adapters."""
    aps = list(runs)
    W = {ap: load_lora(ap) for ap in aps}
    cfgs = {(w[3]["r"], w[3]["lora_alpha"], tuple(sorted(w[3]["target_modules"]))) for w in W.values()}
    assert len(cfgs) == 1, f"LoRA configurations differ across adapters: {cfgs}"
    mods = sorted(W[aps[0]][0])
    n = len(aps)
    # rank check: the row space of A equals the row space of dW = s B A only if B
    # has full column rank at every site
    rank_ok = all(np.linalg.matrix_rank(W[ap][1][m]) == W[ap][3]["r"] for ap in aps for m in mods)
    Q = {(ap, m): np.linalg.qr(W[ap][0][m].T)[0] for ap in aps for m in mods}
    PA = np.zeros((n, n))
    G = np.zeros((n, n))
    for m in mods:
        for i in range(n):
            Ai, Bi, si = W[aps[i]][0][m], W[aps[i]][1][m], W[aps[i]][2]
            for j in range(i, n):
                Aj, Bj, sj = W[aps[j]][0][m], W[aps[j]][1][m], W[aps[j]][2]
                g = si * sj * float(np.trace((Bj.T @ Bi) @ (Ai @ Aj.T)))
                G[i, j] += g
                if i != j:
                    G[j, i] += g
                    c = float(np.clip(np.linalg.svd(Q[(aps[i], m)].T @ Q[(aps[j], m)],
                                                    compute_uv=False), 0, 1).mean())
                    PA[i, j] += c
                    PA[j, i] += c
    PA /= len(mods)
    Dw = np.sqrt(np.maximum(np.add.outer(np.diag(G), np.diag(G)) - 2 * G, 0.0))
    return aps, PA, Dw, rank_ok, len(mods)


# ------------------------------------------------------------------ statistics

def var(v):
    v = np.asarray(v, dtype=float)
    return float(((v - v.mean()) ** 2).sum() / (len(v) - 1))


def perm_var_ratio(a, b):
    """Two-sided exact permutation on |log(s_a^2/s_b^2)| over C(n, n_a) splits."""
    pool = np.array(list(a) + list(b), dtype=float)
    n, na = len(pool), len(a)
    obs = abs(math.log(var(a) / var(b)))
    hits = total = 0
    for cb in itertools.combinations(range(n), na):
        mask = np.zeros(n, bool)
        mask[list(cb)] = True
        total += 1
        if abs(math.log(var(pool[mask]) / var(pool[~mask]))) >= obs - 1e-12:
            hits += 1
    return hits, total


def within_mean(M, idx):
    idx = list(idx)
    return float(np.mean([M[i, j] for i, j in itertools.combinations(idx, 2)]))


def perm_pair_mean(M, ia, ib):
    """Two-sided exact permutation on |mean pair value (A) - mean pair value (B)|."""
    pool = list(ia) + list(ib)
    obs = abs(within_mean(M, ia) - within_mean(M, ib))
    hits = total = 0
    for cb in itertools.combinations(range(len(pool)), len(ia)):
        g1 = [pool[k] for k in cb]
        g2 = [pool[k] for k in range(len(pool)) if k not in cb]
        total += 1
        if abs(within_mean(M, g1) - within_mean(M, g2)) >= obs - 1e-12:
            hits += 1
    return hits, total


def avg_ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(u, v):
    ru, rv = avg_ranks(u), avg_ranks(v)
    ru -= ru.mean()
    rv -= rv.mean()
    d = math.sqrt((ru ** 2).sum() * (rv ** 2).sum())
    return float((ru * rv).sum() / d) if d else 0.0


def mantel(pair_value, ppl):
    """Exact two-sided Mantel test: Spearman(pair value, |dppl|) under all
    relabellings of the runs. pair_value is an n x n matrix, ppl a length-n
    vector; the identity permutation is one of the n! draws."""
    n = len(ppl)
    I, J = np.triu_indices(n, 1)
    u = pair_value[I, J]
    ru = avg_ranks(u)
    ru = ru - ru.mean()
    su = math.sqrt((ru ** 2).sum())
    ppl = np.asarray(ppl, dtype=float)

    def rho(p):
        v = np.abs(ppl[p][I] - ppl[p][J])
        rv = avg_ranks(v)
        rv -= rv.mean()
        return float((ru * rv).sum() / (su * math.sqrt((rv ** 2).sum())))
    obs = rho(np.arange(n))
    hits = total = 0
    for p in itertools.permutations(range(n)):
        total += 1
        if abs(rho(np.array(p))) >= abs(obs) - 1e-12:
            hits += 1
    return obs, hits, total


def fmt_match(paper, value):
    dec = len(str(paper).split(".")[1])
    return "yes" if round(value, dec) == paper else "NO"


# ------------------------------------------------------------------------ main

def main():
    runs = collect_runs()
    n_items = collect_mc(runs)
    aps, PA, Dw, rank_ok, n_mods = pair_matrices(runs)
    ix = {ap: i for i, ap in enumerate(aps)}
    arm_idx = {a: [ix[ap] for ap in aps if runs[ap]["arm"] == a] for a in EXPECTED_N}
    A, B = arm_idx["A"], arm_idx["B"]
    ppl = np.array([runs[ap]["ppl"] for ap in aps])
    acc = np.array([runs[ap].get("acc", np.nan) for ap in aps])

    # item disagreement, as a fraction of the pooled items
    churn = np.full((len(aps), len(aps)), np.nan)
    for i in A + B:
        for j in A + B:
            churn[i, j] = float(np.mean(runs[aps[i]]["pred"] != runs[aps[j]]["pred"]))

    print(f"runs: {', '.join(f'{a}={len(v)}' for a, v in arm_idx.items())}; "
          f"{n_items} benchmark items; {n_mods} adapter sites; "
          f"B full column rank at every site: {rank_ok}")
    print("arm A seeds by perplexity:",
          ", ".join(f"{runs[aps[i]]['seed']}:{ppl[i]:.4f}" for i in sorted(A, key=lambda k: ppl[k])))
    print()

    rows = []
    # --- arm column
    h, t = perm_var_ratio(ppl[A], ppl[B])
    rows.append(("arm", "held-out perplexity", h / t, f"var ratio {var(ppl[A])/var(ppl[B]):.2f}, {h}/{t}"))
    h, t = perm_pair_mean(churn, A, B)
    rows.append(("arm", "item-level behaviour", h / t,
                 f"mean churn {100*within_mean(churn, A):.3f} vs {100*within_mean(churn, B):.3f} pp, {h}/{t}"))
    h, t = perm_pair_mean(PA, A, B)
    rows.append(("arm", "LoRA subspace direction", h / t,
                 f"mean cos {within_mean(PA, A):.5f} vs {within_mean(PA, B):.5f}, {h}/{t}"))
    h, t = perm_pair_mean(Dw, A, B)
    rows.append(("arm", "LoRA update distance", h / t,
                 f"mean dist {within_mean(Dw, A):.3f} vs {within_mean(Dw, B):.3f}, {h}/{t}"))
    h, t = perm_var_ratio(acc[A], acc[B])
    rows.append(("arm", "benchmark accuracy", h / t,
                 f"sd {math.sqrt(var(acc[A])):.3f} vs {math.sqrt(var(acc[B])):.3f}, var ratio {var(acc[A])/var(acc[B]):.2f}, {h}/{t}"))
    # --- run column (arm A only)
    sub = np.ix_(A, A)
    dacc = np.abs(np.subtract.outer(acc[A], acc[A]))
    for name, M in [("item-level behaviour", churn[sub]), ("LoRA subspace direction", PA[sub]),
                    ("LoRA update distance", Dw[sub]), ("benchmark accuracy", dacc)]:
        r, h, t = mantel(M, ppl[A])
        rows.append(("run", name, h / t, f"rho {r:+.3f}, {h}/{t}"))

    print(f"{'level':5s} {'measurement':26s} {'paper':>8s} {'recomputed':>11s} {'match':>6s}   detail")
    all_ok = True
    for lvl, name, p, detail in rows:
        paper = PAPER[(lvl, name)]
        m = fmt_match(paper, p)
        all_ok &= (m == "yes")
        print(f"{lvl:5s} {name:26s} {paper:>8} {p:>11.4f} {m:>6s}   {detail}")

    # --- leave-one-out on the subspace Mantel test
    print("\nleave-one-out, subspace cosine x |dppl| (7 runs, 7! = 5,040 relabellings):")
    loo = []
    for drop in A:
        keep = [i for i in A if i != drop]
        r, h, t = mantel(PA[np.ix_(keep, keep)], ppl[keep])
        loo.append(h / t)
        print(f"  drop seed {runs[aps[drop]]['seed']:<5} rho {r:+.3f}  p {h/t:.4f}  ({h}/{t})")
    below = sorted(p for p in loo if p < 0.05)
    above = sorted(p for p in loo if p >= 0.05)
    print(f"  below 0.05: {len(below)} of 8 -> {', '.join(f'{p:.4f}' for p in below)}")
    print(f"  others: {', '.join(f'{p:.4f}' for p in above)}")
    print(f"  footnote says: {PAPER_LOO}")

    # --- the 88-pair no-overlap statement
    print("\nwithin-arm pair sets behind \"no overlap in 88 pairs\":")
    ranges = {}
    for a in ["D", "A", "F", "B", "C", "E"]:
        vals = [PA[i, j] for i, j in itertools.combinations(arm_idx[a], 2)]
        ranges[a] = (min(vals), max(vals), len(vals))
        print(f"  arm {a}: {len(vals):2d} pairs  mean {np.mean(vals):.5f}  range {min(vals):.5f}-{max(vals):.5f}")
    n88 = sum(ranges[a][2] for a in "DAFBC")
    disjoint = lambda arms_: all(ranges[x][1] < ranges[y][0] or ranges[y][1] < ranges[x][0]
                                 for x, y in itertools.combinations(arms_, 2))
    print(f"  D+A+F+B+C = {n88} pairs, intervals disjoint: {disjoint('DAFBC')}; "
          f"with E: {n88 + ranges['E'][2]} pairs, disjoint: {disjoint('DAFBCE')}")
    print("\nall ten cells match the printed table:", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
