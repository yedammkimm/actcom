"""Why does arm E sit lower on both axes at once?

Lower row-space cosine means the updates point in more different directions;
lower Frobenius distance means they sit closer together. At equal norm those
cannot both happen -- worse alignment forces a larger distance -- so either the
norms differ or the two statistics are weighting the spectrum differently.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math, os
import numpy as np, torch
from safetensors import safe_open
S=ROOT+'results/_wd/'
ARMS=json.load(open(S+'arms.json')); ALPHA,R=32,16; s2=(ALPHA/R)**2
def load(ap):
    out={}
    with safe_open(os.path.join(ROOT,ap,'adapter_model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                b=k.split('.lora_')[0]; sd='A' if '.lora_A.' in k else 'B'
                out.setdefault(b,{})[sd]=f.get_tensor(k).to(torch.float64).numpy()
    return out
want=['B','E','C','A']
W={f'{a}_{s}':load(p) for a in want for s,p in ARMS[a].items()}
mods=sorted(set.intersection(*[set(v) for v in W.values()]))

print('CHECK 1 -- per-adapter update norm')
print(f"  {'adapter':12s}{'||dW||_F':>12s}")
norms={}
for t in W:
    n2=sum(s2*float(np.trace((W[t][m]['B'].T@W[t][m]['B'])@(W[t][m]['A']@W[t][m]['A'].T))) for m in mods)
    norms[t]=math.sqrt(n2)
for a in want:
    v=[norms[f'{a}_{s}'] for s in ARMS[a]]
    print(f"  arm {a}: " + '  '.join(f'{x:.3f}' for x in v) + f"   mean {sum(v)/len(v):.3f}")
mb=sum(norms[f'B_{s}'] for s in ARMS['B'])/8; me=sum(norms[f'E_{s}'] for s in ARMS['E'])/3
mc=sum(norms[f'C_{s}'] for s in ARMS['C'])/2
print(f"\n  E/B norm ratio = {me/mb:.4f}     observed pair-distance ratio = 31.814/37.128 = {31.814/37.128:.4f}")
print(f"  C/B norm ratio = {mc/mb:.4f}")
print("  If the two ratios agree, the distance gap is a norm effect and nothing else.")

print('\nCHECK 2 -- the 16 principal angles individually, not averaged')
def angles(a):
    prs=list(itertools.combinations([f'{a}_{s}' for s in ARMS[a]],2))
    acc=np.zeros(16)
    for x,y in prs:
        for m in mods:
            Q1=np.linalg.qr(W[x][m]['A'].T)[0]; Q2=np.linalg.qr(W[y][m]['A'].T)[0]
            acc+=np.sort(np.clip(np.linalg.svd(Q1.T@Q2,compute_uv=False),0,1))[::-1]
    return acc/(len(prs)*len(mods))
aB,aE=angles('B'),angles('E')
print(f"  {'k':>3s}{'B':>10s}{'E':>10s}{'E-B':>10s}")
for k in range(16): print(f"  {k+1:3d}{aB[k]:10.5f}{aE[k]:10.5f}{aE[k]-aB[k]:+10.5f}")
print(f"  top-4 mean   B {aB[:4].mean():.5f}  E {aE[:4].mean():.5f}  diff {aE[:4].mean()-aB[:4].mean():+.5f}")
print(f"  bottom-8 mean B {aB[8:].mean():.5f}  E {aE[8:].mean():.5f}  diff {aE[8:].mean()-aB[8:].mean():+.5f}")
