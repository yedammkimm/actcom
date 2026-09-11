"""Do the two re-runs of a blockwise configuration separate faster than the FP8 pair?

Raw |dloss| is unreadable: both runs see the same batch at step t, but batch
loss itself swings by more than the difference between runs, so the difference
inherits that swing. What is plotted is a windowed RMS of |dloss|, against a
baseline of the within-run step-to-step variation over the same window. Below
that baseline two runs are not distinguishable from batch noise; the crossing is
where they begin to differ macroscopically, and where each pair crosses is the
content of the figure.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, math, sys
PAIRS=[('A_seed123','blockwise',
        'results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002.json',
        'results/axis4_rerun/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260903_010833.json'),
       ('A_seed4042','blockwise',
        'results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260901_144026.json',
        'results/axis4_rerun/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260902_105622.json'),
       ('B_seed789','FP8',
        'results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260901_202010.json',
        'results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260825_130253.json')]
W=100
def rms(v): return math.sqrt(sum(x*x for x in v)/len(v))
def sd(v):
    m=sum(v)/len(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
out={}
for tag,kind,f1,f2 in PAIRS:
    a=json.load(open(ROOT+f1))['results']['step_losses']; b=json.load(open(ROOT+f2))['results']['step_losses']
    n=min(len(a),len(b)); d=[abs(a[i]-b[i]) for i in range(n)]
    xs,ys,base=[],[],[]
    for s in range(0,n-W,W):
        w=slice(s,s+W)
        xs.append(s+W//2); ys.append(rms(d[w]))
        # within-run step-to-step variation, averaged over the two runs
        base.append((sd([a[i+1]-a[i] for i in range(s,s+W-1)])+
                     sd([b[i+1]-b[i] for i in range(s,s+W-1)]))/2)
    cross=next((xs[k] for k in range(len(xs)) if ys[k]>base[k]), None)
    out[tag]=dict(kind=kind,x=xs,y=ys,base=base,cross=cross,n=n,
                  final=abs(a[-1]-b[-1]),
                  plateau=rms(d[int(n*0.8):]))
    print(f"{tag:12s} {kind:10s} n={n}  first window above the noise floor: "
          f"{'step '+str(cross) if cross is not None else 'never'}")
    print(f"             |dloss| RMS: first window {ys[0]:.5f}  last {ys[-1]:.5f}  "
          f"noise floor first {base[0]:.5f} last {base[-1]:.5f}")
    print(f"             ratio to floor: first {ys[0]/base[0]:.3f}  last {ys[-1]/base[-1]:.3f}")
json.dump(out,open(ROOT+'results/_wd/loss_divergence.json','w'))
print()
print('growth from the first window to the last, as a ratio')
for tag in out:
    o=out[tag]; print(f"  {tag:12s} {o['kind']:10s} {o['y'][-1]/o['y'][0]:8.2f}x")
