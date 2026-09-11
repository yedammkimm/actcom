"""Separate the norm effect from the direction effect.

Row-space principal angles are scale-invariant: the row space of (alpha/r)BA
does not depend on the scaling, so norm cannot explain a subspace-cosine
difference. Frobenius distance is not scale-invariant and can. This checks both
claims directly and reports the norm-normalised distance, which is the quantity
that would be comparable across arms of different update size.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math, os
import numpy as np, torch
from safetensors import safe_open
S=ROOT+'results/_wd/'
ARMS=json.load(open(S+'arms.json')); s2=(32/16)**2
def load(ap):
    out={}
    with safe_open(os.path.join(ROOT,ap,'adapter_model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                b=k.split('.lora_')[0]; sd='A' if '.lora_A.' in k else 'B'
                out.setdefault(b,{})[sd]=f.get_tensor(k).to(torch.float64).numpy()
    return out
want=['A','B','D','F','C','E']
W={f'{a}_{s}':load(p) for a in want for s,p in ARMS[a].items()}
mods=sorted(set.intersection(*[set(v) for v in W.values()]))
def ip(x,y):
    return sum(s2*float(np.trace((W[y][m]['B'].T@W[x][m]['B'])@(W[x][m]['A']@W[y][m]['A'].T))) for m in mods)
nrm={t:math.sqrt(ip(t,t)) for t in W}

print('scale-invariance of the subspace measure, checked directly')
t=list(W)[0]; m=mods[0]
Q1=np.linalg.qr(W[t][m]['A'].T)[0]
Ascaled=W[t][m]['A']*7.3
Q2=np.linalg.qr(Ascaled.T)[0]
sv=np.linalg.svd(Q1.T@Q2,compute_uv=False)
print(f"  same A scaled by 7.3: mean principal-angle cosine with itself = {sv.mean():.10f}")
print("  (1.0 confirms the row space, and therefore every angle, ignores scale)")

print('\nFrobenius distance, raw and normalised by the pair\'s geometric-mean norm')
print(f"  {'arm':4s}{'runs':>6s}{'pairs':>7s}{'mean norm':>12s}{'raw dist':>11s}{'dist/norm':>12s}")
for a in want:
    tg=[f'{a}_{s}' for s in ARMS[a]]; prs=list(itertools.combinations(tg,2))
    raw=[math.sqrt(max(ip(x,x)+ip(y,y)-2*ip(x,y),0.0)) for x,y in prs]
    nd=[d/math.sqrt(nrm[x]*nrm[y]) for d,(x,y) in zip(raw,prs)]
    mn=sum(nrm[t] for t in tg)/len(tg)
    print(f"  {a:4s}{len(tg):6d}{len(prs):7d}{mn:12.3f}{sum(raw)/len(raw):11.3f}{sum(nd)/len(nd):12.4f}")
