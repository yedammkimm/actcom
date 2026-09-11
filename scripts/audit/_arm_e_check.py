"""Is arm E's subspace value explainable by n=3 versus n=8 alone?

The 28 within-arm pair values are not independent: a pair shares a run with six
others. Model d_ij = mu + a_i + a_j + eps_ij, so

    Var(mean over C(n,2) pairs) = (4/n) sigma_a^2 + sigma_eps^2 / C(n,2)

which is not sigma^2/C(n,2). Components come from the pair variance and the
covariance between pairs sharing exactly one run:

    Var(d_ij)             = 2 sigma_a^2 + sigma_eps^2
    Cov(d_ij, d_ik)       =   sigma_a^2
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, itertools, math
import numpy as np
S=ROOT+'results/_wd/'
tags=json.load(open(S+'tags_all.json')); PA=np.load(S+'pa_all.npy')
i={t:k for k,t in enumerate(tags)}
def arm(a): return [t for t in tags if t.startswith(a+'_')]
g=lambda x,y: float(PA[i[x]][i[y]])

def components(members):
    prs=list(itertools.combinations(members,2))
    d={p:g(*p) for p in prs}
    v=list(d.values()); n=len(members)
    mu=sum(v)/len(v)
    var=sum((x-mu)**2 for x in v)/(len(v)-1)
    co=[]
    for p,q in itertools.combinations(prs,2):
        if len(set(p)&set(q))==1: co.append((d[p]-mu)*(d[q]-mu))
    cov=sum(co)/(len(co)-1)
    sa2=max(cov,0.0); se2=max(var-2*sa2,0.0)
    return mu,var,cov,sa2,se2,n
def var_mean(sa2,se2,n): return (4.0/n)*sa2 + se2/(n*(n-1)/2)

B=arm('B'); E=arm('E'); A=arm('A'); F=arm('F')
print('variance components from the eight-run arms')
print(f"  {'arm':4s}{'mean':>10s}{'Var(pair)':>12s}{'Cov(share1)':>13s}{'sigma_a':>10s}{'sigma_eps':>11s}")
comp={}
for nm,m in [('B',B),('A',A),('F',F)]:
    mu,var,cov,sa2,se2,n=components(m); comp[nm]=(sa2,se2)
    print(f"  {nm:4s}{mu:10.5f}{var:12.3e}{cov:13.3e}{math.sqrt(sa2):10.3e}{math.sqrt(se2):11.3e}")
muE=sum(g(*p) for p in itertools.combinations(E,2))/3
muB=components(B)[0]
print(f"\n  arm E mean (n=3, 3 pairs) = {muE:.5f}")
print(f"  arm B mean (n=8, 28 pairs) = {muB:.5f}     difference = {muB-muE:+.5f}")
print()
print('  is that difference within sampling noise?  (B components used for both)')
for src in ['B','A','F']:
    sa2,se2=comp[src]
    v3=var_mean(sa2,se2,3); v8=var_mean(sa2,se2,8)
    z=(muB-muE)/math.sqrt(v3+v8)
    print(f"    components from arm {src}:  sd(n=3) {math.sqrt(v3):.5f}   sd(n=8) {math.sqrt(v8):.5f}   z = {z:.2f}")
print()
print('  reference: the naive subset distribution, which understates the noise')
sub=[sum(g(*p) for p in itertools.combinations(c,2))/3 for c in itertools.combinations(B,3)]
sub.sort()
print(f"    {len(sub)} three-run subsets of arm B: min {sub[0]:.5f}  max {sub[-1]:.5f}  sd {np.std(sub,ddof=1):.5f}")
print(f"    arm E's {muE:.5f} is below all of them: {sum(1 for x in sub if x<=muE)}/{len(sub)}")
print("    (every subset reuses the same eight runs, so this is variation in")
print("     choosing three of eight, not in drawing three new runs)")
