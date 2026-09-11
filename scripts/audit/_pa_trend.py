"""Trend test on the arm-level subspace dispersion.

The unit of exchangeability is the adapter, not the pair: each adapter appears
in seven within-arm pairs, so a permutation over pairs would treat dependent
observations as independent. The null here reassigns the 27 compressed adapters
to arms of the observed sizes and recomputes every arm mean from scratch.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math, random
import numpy as np
S=ROOT+'results/_wd/'
tags=json.load(open(S+'tags_all.json')); PA=np.load(S+'pa_all.npy')
idx={t:i for i,t in enumerate(tags)}
ARMS=json.load(open(S+'arms.json'))
COS={'D':0.371,'A':0.587,'F':0.910,'B':0.992}
BITS={'D':4.125,'A':4.125,'F':4.105,'B':8.125}
order=['D','A','F','B']
members={a:[f'{a}_{s}' for s in ARMS[a]] for a in order}
mean=lambda v: sum(v)/len(v)
def armmean(tg): return mean([PA[idx[x]][idx[y]] for x,y in itertools.combinations(tg,2)])
obs={a:armmean(members[a]) for a in order}
def ranks(x):
    o=sorted(range(len(x)),key=lambda k:x[k]); rk=[0.0]*len(x); k=0
    while k<len(o):
        j=k
        while j+1<len(o) and x[o[j+1]]==x[o[k]]: j+=1
        for m in range(k,j+1): rk[o[m]]=(k+j)/2+1
        k=j+1
    return rk
def pear(u,v):
    n=len(u); mu,mv=mean(u),mean(v)
    su=math.sqrt(sum((x-mu)**2 for x in u)); sv=math.sqrt(sum((x-mv)**2 for x in v))
    return sum((u[k]-mu)*(v[k]-mv) for k in range(n))/(su*sv) if su*sv else 0.0
pool=[t for a in order for t in members[a]]; sizes=[len(members[a]) for a in order]
def stat(assign,score):
    ms=[armmean(g) for g in assign]
    return pear(ranks([score[a] for a in order]),ranks(ms))
obs_c=stat([members[a] for a in order],COS)
obs_b=stat([members[a] for a in order],BITS)
print('arm-level subspace dispersion vs the two candidate dose axes')
print(f"  {'arm':4s}{'cos(g,g*)':>11s}{'bits':>8s}{'subspace cos':>14s}{'n runs':>8s}")
for a in order: print(f"  {a:4s}{COS[a]:11.3f}{BITS[a]:8.3f}{obs[a]:14.5f}{len(members[a]):8d}")
print(f"\n  Spearman(gradient cosine, subspace cos) = {obs_c:+.3f}")
print(f"  Spearman(bits/elem,       subspace cos) = {obs_b:+.3f}")
rnd=random.Random(0); N=100000; cc=cb=0
for _ in range(N):
    p=pool[:]; rnd.shuffle(p); a=[]; k=0
    for sz in sizes: a.append(p[k:k+sz]); k+=sz
    if abs(stat(a,COS))>=abs(obs_c)-1e-12: cc+=1
    if abs(stat(a,BITS))>=abs(obs_b)-1e-12: cb+=1
print(f"  permutation p (adapters reassigned to arms of the same sizes, {N:,} draws)")
print(f"    gradient cosine  p = {(cc+1)/(N+1):.5f}")
print(f"    bits per element p = {(cb+1)/(N+1):.5f}")
# how often does a random reassignment separate the arms with no overlap?
def sep(assign):
    rngs=[(min(PA[idx[x]][idx[y]] for x,y in itertools.combinations(g,2)),
           max(PA[idx[x]][idx[y]] for x,y in itertools.combinations(g,2))) for g in assign]
    rngs=sorted(rngs)
    return all(rngs[i][1]<rngs[i+1][0] for i in range(len(rngs)-1))
rnd=random.Random(1); N2=100000; cs=0
for _ in range(N2):
    p=pool[:]; rnd.shuffle(p); a=[]; k=0
    for sz in sizes: a.append(p[k:k+sz]); k+=sz
    if sep(a): cs+=1
print(f"\n  complete separation of all four arms' pair ranges: observed yes;")
print(f"  under random reassignment {cs}/{N2:,} = p = {(cs+1)/(N2+1):.5f}")
