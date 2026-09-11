"""Row-space principal angles for all five arms, and the dose-response curve.

Same construction as weight_distance.py: the comparison is on dW = (alpha/r)BA
via 16x16 traces, and subspace comparison uses the row space of A. Every one of
the 29 adapters carries the same LoRA configuration (r 16, alpha 32, q/k/v/o),
which is checked before anything is computed.
"""
import json, os, itertools, math
import numpy as np, torch
from safetensors import safe_open
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'; S=ROOT+'results/_wd/'
ARMS=json.load(open(S+'arms.json'))
COS={'D':0.3712,'A':0.5868,'F':0.9103,'B':0.9919,'E':0.9919,'C':None}   # C is uncompressed
ALPHA,R=32,16; s2=(ALPHA/R)**2

def load(ap):
    out={}
    with safe_open(os.path.join(ROOT,ap,'adapter_model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                b=k.split('.lora_')[0]; sd='A' if '.lora_A.' in k else 'B'
                out.setdefault(b,{})[sd]=f.get_tensor(k).to(torch.float64).numpy()
    return out

tags=[f'{a}_{s}' for a in ARMS for s in ARMS[a]]
print(f'loading {len(tags)} adapters ...',flush=True)
W={f'{a}_{s}':load(p) for a in ARMS for s,p in ARMS[a].items()}
mods=sorted(set.intersection(*[set(v) for v in W.values()]))
print(f'{len(mods)} common modules')
n=len(tags); idx={t:i for i,t in enumerate(tags)}
PA=np.zeros((n,n)); G=np.zeros((n,n))
for m in mods:
    Q={t:np.linalg.qr(W[t][m]['A'].T)[0] for t in tags}
    A={t:W[t][m]['A'] for t in tags}; B={t:W[t][m]['B'] for t in tags}
    for i in range(n):
        for j in range(i,n):
            ti,tj=tags[i],tags[j]
            v=s2*float(np.trace((B[tj].T@B[ti])@(A[ti]@A[tj].T)))
            G[i,j]+=v
            if i!=j:
                G[j,i]+=v
                c=float(np.clip(np.linalg.svd(Q[ti].T@Q[tj],compute_uv=False),0,1).mean())
                PA[i,j]+=c; PA[j,i]+=c
PA/=len(mods)
Dw=np.zeros((n,n))
for i in range(n):
    for j in range(n): Dw[i,j]=math.sqrt(max(G[i,i]+G[j,j]-2*G[i,j],0.0))
np.save(S+'pa_all.npy',PA); np.save(S+'dw_all.npy',Dw); json.dump(tags,open(S+'tags_all.json','w'))
mean=lambda v: sum(v)/len(v)
print()
print('within-arm mean row-space cosine  (lower = more different subspaces)')
print(f"  {'arm':4s}{'cos(g,g*)':>11s}{'runs':>6s}{'pairs':>7s}{'mean':>10s}{'min':>9s}{'max':>9s}{'Frobenius':>11s}")
res={}
for a in ARMS:
    tg=[f'{a}_{s}' for s in ARMS[a]]
    pr=[(x,y) for x,y in itertools.combinations(tg,2)]
    v=[PA[idx[x]][idx[y]] for x,y in pr]; d=[Dw[idx[x]][idx[y]] for x,y in pr]
    res[a]=(mean(v),len(pr))
    c=COS[a]; cs=f'{c:.3f}' if c is not None else '  --'
    print(f"  {a:4s}{cs:>11s}{len(tg):6d}{len(pr):7d}{mean(v):10.5f}{min(v):9.5f}{max(v):9.5f}{mean(d):11.3f}")
print()
print('is the ordering monotone in gradient cosine?')
o=[(COS[a],res[a][0],a) for a in ARMS if COS[a] is not None]
o.sort()
print('  '+'   '.join(f'{a} (cos {c:.3f}) -> {v:.5f}' for c,v,a in o))
mono=all(o[i][1]<=o[i+1][1] for i in range(len(o)-1))
print(f'  monotone increasing with gradient cosine: {mono}')
print(f"  uncompressed reference C: {res['C'][0]:.5f} from {res['C'][1]} pair")
