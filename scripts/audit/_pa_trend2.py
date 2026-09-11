"""Higher-power version: correlate each within-arm pair's subspace cosine with
its arm's dose score, over all 87 pairs, under the same adapter-level null.

Using the four arm means throws away the within-arm values and leaves only four
ranks, which a random reassignment reproduces about one time in six. Using the
pair values keeps the magnitudes; the null is unchanged, so the dependence
between pairs sharing an adapter is still handled correctly.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math, random
import numpy as np
S=ROOT+'results/_wd/'
tags=json.load(open(S+'tags_all.json')); PA=np.load(S+'pa_all.npy')
idx={t:i for i,t in enumerate(tags)}; ARMS=json.load(open(S+'arms.json'))
COS={'D':0.371,'A':0.587,'F':0.910,'B':0.992}
BITS={'D':4.125,'A':4.125,'F':4.105,'B':8.125}
RANK4={'D':0,'A':0,'F':1,'B':1}
order=['D','A','F','B']; members={a:[f'{a}_{s}' for s in ARMS[a]] for a in order}
sizes=[len(members[a]) for a in order]; pool=[t for a in order for t in members[a]]
mean=lambda v: sum(v)/len(v)
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
def pairstat(assign,score):
    xs,ys=[],[]
    for a,g in zip(order,assign):
        for x,y in itertools.combinations(g,2):
            xs.append(score[a]); ys.append(PA[idx[x]][idx[y]])
    return pear(ranks(xs),ranks(ys)),len(xs)
print('pair-level trend, all within-arm pairs')
print(f"  {'dose axis':22s}{'rho':>8s}{'pairs':>7s}{'perm p':>10s}")
obs={}; 
for nm,sc in [('gradient cosine',COS),('bits per element',BITS),('rank-4 rule (binary)',RANK4)]:
    r,npair=pairstat([members[a] for a in order],sc); obs[nm]=(r,sc,npair)
rnd=random.Random(0); N=100000; cnt={k:0 for k in obs}
for _ in range(N):
    p=pool[:]; rnd.shuffle(p); a=[]; k=0
    for sz in sizes: a.append(p[k:k+sz]); k+=sz
    for nm,(r0,sc,_) in obs.items():
        if abs(pairstat(a,sc)[0])>=abs(r0)-1e-12: cnt[nm]+=1
for nm,(r0,sc,npair) in obs.items():
    print(f"  {nm:22s}{r0:+8.3f}{npair:7d}{(cnt[nm]+1)/(N+1):10.5f}")
print(f"\n  ({N:,} reassignments of the 27 adapters into arms of sizes {sizes})")
