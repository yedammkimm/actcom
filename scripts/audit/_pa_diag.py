import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, numpy as np
S=ROOT+'results/_wd/'
tags=json.load(open(S+'weight_tags.json')); PA=np.load(S+'princ_angle.npy')
i={t:k for k,t in enumerate(tags)}
A=[t for t in tags if t.startswith('A_')]; B=[t for t in tags if t.startswith('B_')]
ga=lambda a,b: PA[i[a]][i[b]]
aa=sorted(ga(x,y) for x,y in itertools.combinations(A,2))
bb=sorted(ga(x,y) for x,y in itertools.combinations(B,2))
ab=sorted(ga(x,y) for x in A for y in B)
f=lambda v: f"min {v[0]:.4f}  q1 {v[len(v)//4]:.4f}  med {v[len(v)//2]:.4f}  max {v[-1]:.4f}  mean {sum(v)/len(v):.5f}"
print('pairwise mean row-space cosine')
print('  A-A (28):', f(aa)); print('  B-B (28):', f(bb)); print('  A-B (64):', f(ab))
print(f'  overlap: A-A values above B-B min = {sum(1 for v in aa if v>bb[0])}/28;'
      f'  B-B below A-A max = {sum(1 for v in bb if v<aa[-1])}/28')
mean=lambda v: sum(v)/len(v)
def test(gA,gB,label):
    pool=gA+gB; n=len(gA)
    wm=lambda g: mean([ga(x,y) for x,y in itertools.combinations(g,2)])
    o=wm(gA)-wm(gB); c=t=0
    for cb in itertools.combinations(range(len(pool)),n):
        g1=[pool[k] for k in cb]; g2=[pool[k] for k in range(len(pool)) if k not in cb]; t+=1
        if abs(wm(g1)-wm(g2))>=abs(o)-1e-12: c+=1
    print(f'  {label:34s} {len(gA)}v{len(gB)}  diff {o:+.5f}  p {c/t:.4f}  ({t:,} splits)')
print('\nis the collapse a group-balance artifact?')
test(A,B,'full 8v8')
test(A[:-1],B,'drop one from A (unbalanced)')
test(A,B[:-1],'drop one from B (unbalanced)')
test(A[:-1],B[:-1],'drop one from EACH (balanced 7v7)')
test(A[:-2],B[:-2],'drop two from each (balanced 6v6)')
print('\nsame diagnostic on the item-disagreement statistic, for comparison')
rows=json.load(open(S+'rows.json')); T=['arc_challenge','arc_easy','piqa','winogrande']
pr={}
for t in tags:
    p=[]
    for tk in T:
        d=json.load(open(f'{ROOT}results/mc_downstream/mc__{t}__{tk}.json'))['tasks'][tk]
        p+=[s['pred_sum'] for s in d['per_sample']]
    pr[t]=p
N=len(pr[tags[0]])
D={(x,y):sum(1 for k in range(N) if pr[x][k]!=pr[y][k])/N for x,y in itertools.combinations(tags,2)}
gd=lambda x,y: D[(x,y)] if (x,y) in D else D[(y,x)]
def test2(gA,gB,label):
    pool=gA+gB; n=len(gA)
    wm=lambda g: mean([gd(x,y) for x,y in itertools.combinations(g,2)])
    o=wm(gA)-wm(gB); c=t=0
    for cb in itertools.combinations(range(len(pool)),n):
        g1=[pool[k] for k in cb]; g2=[pool[k] for k in range(len(pool)) if k not in cb]; t+=1
        if abs(wm(g1)-wm(g2))>=abs(o)-1e-12: c+=1
    print(f'  {label:34s} {len(gA)}v{len(gB)}  diff {o:+.5f}  p {c/t:.4f}')
test2(A,B,'full 8v8'); test2(A[:-1],B,'drop one from A (unbalanced)')
test2(A[:-1],B[:-1],'drop one from EACH (balanced 7v7)')
