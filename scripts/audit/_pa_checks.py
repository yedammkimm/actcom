"""Robustness for the row-space principal-angle result."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math, random, os
import numpy as np
S = ROOT + 'results/_wd/'
tags = json.load(open(S + 'weight_tags.json'))
PA = np.load(S + 'princ_angle.npy')
Dw = np.load(S + 'weight_dist.npy')
i = {t: k for k, t in enumerate(tags)}
A = [t for t in tags if t.startswith('A_')]; B = [t for t in tags if t.startswith('B_')]
mean = lambda v: sum(v) / len(v)
ga = lambda a, b: PA[i[a]][i[b]]
def armtest(f, members_A, members_B):
    pool = members_A + members_B; n = len(members_A)
    wm = lambda g: mean([f(x, y) for x, y in itertools.combinations(g, 2)])
    o = wm(members_A) - wm(members_B); c = t = 0
    for cb in itertools.combinations(range(len(pool)), n):
        g1 = [pool[k] for k in cb]; g2 = [pool[k] for k in range(len(pool)) if k not in cb]
        t += 1
        if abs(wm(g1) - wm(g2)) >= abs(o) - 1e-12: c += 1
    return wm(members_A), wm(members_B), c / t

print('1) random-subspace baseline: what does an unrelated pair of 16-dim')
print('   subspaces of R^3072 give?  (100 draws)')
rng = np.random.default_rng(0); vals = []
for _ in range(100):
    Q1 = np.linalg.qr(rng.standard_normal((3072, 16)))[0]
    Q2 = np.linalg.qr(rng.standard_normal((3072, 16)))[0]
    vals.append(np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), 0, 1).mean())
print(f'   random mean cos = {np.mean(vals):.5f} +- {np.std(vals):.5f}')
a, b, p = armtest(ga, A, B)
print(f'   observed  arm A {a:.5f}   arm B {b:.5f}   (both well above random)')
print(f'   arm difference exact permutation p = {p:.4f}')

print('\n2) leave-one-adapter-out')
worst = 0
for drop in A + B:
    aa = [x for x in A if x != drop]; bb = [x for x in B if x != drop]
    x, y, q = armtest(ga, aa, bb); worst = max(worst, q)
    print(f'   drop {drop:12s} A {x:.5f}  B {y:.5f}  ratio {x/y:.4f}  p {q:.4f}')
print(f'   worst p over 16 drops: {worst:.4f}')

print('\n3) Mantel with the principal-angle matrix instead of the distance')
def ranks(x):
    o = sorted(range(len(x)), key=lambda k: x[k]); rk = [0.0]*len(x); k = 0
    while k < len(o):
        j = k
        while j+1 < len(o) and x[o[j+1]] == x[o[k]]: j += 1
        for m in range(k, j+1): rk[o[m]] = (k+j)/2+1
        k = j+1
    return rk
def pear(u, v):
    n=len(u); mu,mv=mean(u),mean(v)
    su=math.sqrt(sum((x-mu)**2 for x in u)); sv=math.sqrt(sum((x-mv)**2 for x in v))
    return sum((u[k]-mu)*(v[k]-mv) for k in range(n))/(su*sv) if su*sv else 0.0
rows = json.load(open(S + 'rows.json')); by = {r['tag']: r for r in rows}
T = ['arc_challenge','arc_easy','piqa','winogrande']
preds = {}
for t in tags:
    p = []
    for tk in T:
        d = json.load(open(f'{ROOT}results/mc_downstream/mc__{t}__{tk}.json'))['tasks'][tk]
        p += [s['pred_sum'] for s in d['per_sample']]
    preds[t] = p
N = len(preds[tags[0]])
DIS = {(x, y): sum(1 for k in range(N) if preds[x][k] != preds[y][k])/N
       for x, y in itertools.combinations(tags, 2)}
gd = lambda x, y: DIS[(x, y)] if (x, y) in DIS else DIS[(y, x)]
gp = lambda x, y: abs(by[x]['ppl'] - by[y]['ppl'])
def mantel(tg, f1, f2, seed=0, NP=20000):
    prs = list(itertools.combinations(tg, 2))
    u = [f1(x, y) for x, y in prs]; v = [f2(x, y) for x, y in prs]
    r0 = pear(ranks(u), ranks(v)); rnd = random.Random(seed); c = 0
    for _ in range(NP):
        pm = tg[:]; rnd.shuffle(pm); m = dict(zip(tg, pm))
        if abs(pear(ranks(u), ranks([f2(m[x], m[y]) for x, y in prs]))) >= abs(r0)-1e-12: c += 1
    return r0, (c+1)/(NP+1)
for lbl, tg in [('arm A only', A), ('arm B only', B), ('A+B', A+B)]:
    r1, p1 = mantel(tg, ga, gp); r2, p2 = mantel(tg, ga, gd)
    print(f'   {lbl:12s} angle x ppl  rho {r1:+.3f} p {p1:.3f}     angle x disagree  rho {r2:+.3f} p {p2:.3f}')
