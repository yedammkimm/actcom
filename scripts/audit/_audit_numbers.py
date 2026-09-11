"""Recompute the paper's headline numbers from stored results and diff them.

Completion criterion 1: every number in the body reproduces from a JSON on disk.
This covers the statistics added or changed during the 2026-09-08 reframing; the
older tables were audited earlier against their own sources.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, glob, re, math, itertools, os, sys
R=ROOT
def norm(p): return re.sub(r'^/app/HMA_Project/','',(p or '').rstrip('/'))
fails=[]
def chk(name, claimed, got, tol=5e-4):
    """Compare against the unrounded value with a half-ulp tolerance.

    Comparing round(x, 4) to the printed figure double-rounds: arm F's mean is
    0.1231496, which is 0.12315 at five places and then rounds up to 0.1232 at
    four. The paper prints one rounding of the stored number, so the check has
    to be against the number and not against a rounding of it.
    """
    ok = got is not None and abs(claimed-got) <= tol*max(1.0,abs(claimed))
    print(f"  {'OK ' if ok else 'FAIL':5s} {name:52s} paper {claimed:<12} recomputed {got}")
    if not ok: fails.append(name)

# ---- perplexity by arm, joined on adapter path
M={}
for f in glob.glob(R+'results/ppl*/*.json'):
    d=json.load(open(f)); a=norm(d.get('adapter_path'))
    if a and d.get('wikitext2_ppl') is not None: M.setdefault(a,d['wikitext2_ppl'])
ARMS=json.load(open(R+'results/_wd/arms.json'))
def vals(arm): return [M[p] for p in ARMS[arm].values() if p in M]
def var(v):
    m=sum(v)/len(v); return sum((x-m)**2 for x in v)/(len(v)-1)
A,B=vals('A'),vals('B')
print('section 5.3 -- perplexity dispersion')
chk('arm A n', 8, len(A), 0); chk('arm B n', 8, len(B), 0)
chk('variance ratio 33.7', 33.7, round(var(A)/var(B),1), 1e-9)
chk('mean difference +0.58', 0.58, round(sum(A)/8-sum(B)/8,2), 1e-9)
def welch(a,b): return (sum(a)/len(a)-sum(b)/len(b))/math.sqrt(var(a)/len(a)+var(b)/len(b))
chk('Welch t 1.86', 1.86, round(welch(A,B),2), 1e-9)
pool=A+B; c=t=0
for cb in itertools.combinations(range(16),8):
    g1=[pool[i] for i in cb]; g2=[pool[i] for i in range(16) if i not in cb]; t+=1
    f=var(g1)/var(g2)
    if abs(math.log(f))>=abs(math.log(var(A)/var(B)))-1e-12: c+=1
chk('permutation p 0.0303', 0.0303, round(c/t,4), 1e-9)

# ---- subspace, from the saved matrices
import struct, array
def loadnpy(p):
    with open(p,'rb') as fh:
        assert fh.read(6)==b'\x93NUMPY'; maj,_=fh.read(2)
        hl=struct.unpack('<H',fh.read(2))[0] if maj==1 else struct.unpack('<I',fh.read(4))[0]
        h=eval(fh.read(hl).decode()); sh=h['shape']; n=sh[0]*sh[1]
        d=array.array('d'); d.fromfile(fh,n)
        return [[d[i*sh[1]+j] for j in range(sh[1])] for i in range(sh[0])]
tags=json.load(open(R+'results/_wd/tags_all.json')); PA=loadnpy(R+'results/_wd/pa_all.npy')
ix={t_:i for i,t_ in enumerate(tags)}
def sub(arm):
    tg=[t_ for t_ in tags if t_.startswith(arm+'_')]
    v=[PA[ix[x]][ix[y]] for x,y in itertools.combinations(tg,2)]
    return sum(v)/len(v), min(v), max(v)
print('\nsection 5.3 -- subspace')
for arm,claim in [('D',0.0829),('A',0.1121),('F',0.1231),('B',0.1339),('C',0.1391),('E',0.1314)]:
    chk(f'arm {arm} mean row-space cosine', claim, sub(arm)[0], 5e-5)
chk('arm B range lower 0.13297', 0.13297, sub('B')[1], 5e-5)
chk('arm B range upper 0.13495', 0.13495, sub('B')[2], 5e-5)

# ---- item disagreement, from mc_downstream
print('\nsection 5.4 -- item disagreement')
T=['arc_challenge','arc_easy','piqa','winogrande']
rows=json.load(open(R+'results/_wd/rows.json'))
pr={}
for r in rows:
    p=[]
    for tk in T:
        d=json.load(open(R+f"results/mc_downstream/mc__{r['tag']}__{tk}.json"))['tasks'][tk]
        p+=[s['pred_sum'] for s in d['per_sample']]
    pr[r['tag']]=p
N=len(pr['base']); chk('item count 6653', 6653, N, 0)
arm={r['tag']:r['arm'] for r in rows}
def churn(a,b): return sum(1 for i in range(N) if pr[a][i]!=pr[b][i])/N*100
for a,claim in [('C',6.478),('B',6.594),('A',7.387),('D',9.079)]:
    tg=[t_ for t_ in pr if arm[t_]==a]
    v=[churn(x,y) for x,y in itertools.combinations(tg,2)]
    chk(f'arm {a} churn', claim, sum(v)/len(v), 5e-4)
chk('base vs arm A churn 13.24', 13.24,
    round(sum(churn('base',t_) for t_ in pr if arm[t_]=='A')/8,2), 1e-9)
print(f"\n{len(fails)} failures" + (': '+', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)
