"""One run that fixes every Brown-Forsythe trend number the paper quotes.

Earlier figures were computed at different draw counts and, worse, on different
arm sets: a z-shuffle p from the four-arm set was divided by an arm-relabel p
from the five-arm set. Everything here is the five-arm set (30 runs) at 10^6
draws, so the ratios compare like with like.
"""
import math, random, itertools, sys
ARMS={'D':(0.3712,4.125,[17.694,17.263,33.747]),
      'A':(0.5868,4.125,[15.7796178,18.4320070,16.9549848,15.9558043,
                         16.1015878,16.0627072,17.0183249,16.9350069]),
      'F':(0.9103,4.105,[16.3664,16.2876,16.1257,16.6579,
                         16.5298,16.5877,15.9908,16.2414]),
      'B':(0.9919,8.125,[15.9330655,16.0007507,16.2828162,16.0926890,
                         15.8886977,16.3028794,16.0370449,16.0253090]),
      'E':(0.9919,8.125,[15.9250,16.4676,16.0782])}
def med(v):
    s=sorted(v); n=len(s); return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
def ranks(x):
    o=sorted(range(len(x)),key=lambda i:x[i]); rk=[0.]*len(x); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and x[o[j+1]]==x[o[i]]: j+=1
        for k in range(i,j+1): rk[o[k]]=(i+j)/2+1
        i=j+1
    return rk
def pear(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    sa=math.sqrt(sum((v-ma)**2 for v in a)); sb=math.sqrt(sum((v-mb)**2 for v in b))
    return sum((a[i]-ma)*(b[i]-mb) for i in range(n))/(sa*sb) if sa*sb else 0.
def zvals(sc,gs):
    xs,zs=[],[]
    for c,g in zip(sc,gs):
        m=med(g)
        for v in g: xs.append(c); zs.append(abs(v-m))
    return xs,zs
def relabel(keys,axis,M,seed):
    sc=[ARMS[k][axis] for k in keys]; gs=[ARMS[k][2] for k in keys]
    sizes=[len(g) for g in gs]; pool=[x for g in gs for x in g]
    def stat(g): 
        x,z=zvals(sc,g); return pear(ranks(x),ranks(z))
    r0=stat(gs); rnd=random.Random(seed); c=0
    for _ in range(M):
        p=pool[:]; rnd.shuffle(p); gg=[]; k=0
        for s in sizes: gg.append(p[k:k+s]); k+=s
        if stat(gg)<=r0+1e-12: c+=1
    return r0,c,M,sum(sizes)
def shuffle_z(keys,axis,M,seed):
    sc=[ARMS[k][axis] for k in keys]; gs=[ARMS[k][2] for k in keys]
    xs,zs=zvals(sc,gs); rx=ranks(xs); r0=pear(rx,ranks(zs))
    rnd=random.Random(seed); c=0; z=zs[:]
    for _ in range(M):
        rnd.shuffle(z)
        if pear(rx,ranks(z))<=r0+1e-12: c+=1
    return r0,c,M
M=int(sys.argv[1]) if len(sys.argv)>1 else 1000000
FIVE=['D','A','F','B','E']; FOUR=['D','A','F','B']; BITSFIX=['D','A','F']
print(f'all p one-sided, arm-relabel with group sizes preserved, M = {M:,}\n')
print(f"  {'set':26s}{'axis':>9s}{'N':>4s}{'rho':>9s}{'hits':>8s}{'p':>12s}")
rows=[]
for lbl,keys in [('five arms',FIVE),('four arms (no E)',FOUR),('bits held fixed',BITSFIX)]:
    for ax,an in [(0,'cosine'),(1,'bits')]:
        r,c,m,N=relabel(keys,ax,M,ax)
        rows.append((lbl,an,r,c,m))
        print(f"  {lbl:26s}{an:>9s}{N:4d}{r:+9.4f}{c:8d}{c/m:12.3e}")
print()
r0,c,m=shuffle_z(FIVE,0,M,9)
p_rel=[x for x in rows if x[0]=='five arms' and x[1]=='cosine'][0][3]/M
print(f"  z-shuffle on the SAME five-arm set: rho {r0:+.4f}  p {c/m:.3e}")
print(f"  conservatism factor = {(c/m)/p_rel:.1f}x   (both on 30 runs, both at M={M:,})")
