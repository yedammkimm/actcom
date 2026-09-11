"""The three rerun pairs: same seed, same configuration, trained twice.

These are the only run-level contrast in the project where the seed is held
fixed, so they separate "the seed decides" from "the run decides". Two are
blockwise (arm A) and one is FP8 routing (arm B). Compared against the
cross-arm same-seed pairs (0.82, which share an initialisation but differ in
compression) and the within-arm different-seed pairs (0.11-0.13).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import json, os, itertools, math
import numpy as np, torch
from safetensors import safe_open
# ROOT (repository root) is derived from this file's location above
PAIRS=[('A_seed123','results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002_adapter',
                    'results/axis4_rerun/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260903_010833_adapter',
                    18.432,16.271),
       ('A_seed4042','results/axis2_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260901_144026_adapter',
                     'results/axis4_rerun/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed4042__20260902_105622_adapter',
                     16.935,16.351),
       ('B_seed789','results/axis3_b4dfp8/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260901_202010_adapter',
                    'results/pilot_e2m1_seeds/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed789__20260825_130253_adapter',
                    16.093,15.999)]
ALPHA,R=32,16; s2=(ALPHA/R)**2
def load(ap):
    out={}
    with safe_open(os.path.join(ROOT,ap,'adapter_model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                b=k.split('.lora_')[0]; sd='A' if '.lora_A.' in k else 'B'
                out.setdefault(b,{})[sd]=f.get_tensor(k).to(torch.float64).numpy()
    return out
print(f"{'pair':12s}{'ppl 1':>9s}{'ppl 2':>9s}{'|dppl|':>9s}{'row-space cos':>15s}{'||dW1-dW2||':>13s}")
res=[]
for tag,p1,p2,x1,x2 in PAIRS:
    W1,W2=load(p1),load(p2)
    mods=sorted(set(W1)&set(W2)); acc=[]; g11=g22=g12=0.0
    for m in mods:
        Q1=np.linalg.qr(W1[m]['A'].T)[0]; Q2=np.linalg.qr(W2[m]['A'].T)[0]
        acc.append(float(np.clip(np.linalg.svd(Q1.T@Q2,compute_uv=False),0,1).mean()))
        A1,B1,A2,B2=W1[m]['A'],W1[m]['B'],W2[m]['A'],W2[m]['B']
        g11+=s2*float(np.trace((B1.T@B1)@(A1@A1.T)))
        g22+=s2*float(np.trace((B2.T@B2)@(A2@A2.T)))
        g12+=s2*float(np.trace((B2.T@B1)@(A1@A2.T)))
    c=sum(acc)/len(acc); d=math.sqrt(max(g11+g22-2*g12,0.0))
    res.append((tag,c,d))
    print(f"  {tag:10s}{x1:9.3f}{x2:9.3f}{abs(x1-x2):9.3f}{c:15.5f}{d:13.3f}")
print()
print("reference scales from the earlier run")
print("  same seed, different compression (A_sX x B_sX)   0.819 - 0.820")
print("  different seed, same arm                          0.111 - 0.135")
print("  two unrelated 16-d subspaces of R^3072            0.061")
print()
A=[c for t,c,_ in res if t.startswith('A')]; B=[c for t,c,_ in res if t.startswith('B')]
print(f"  blockwise reruns (n=2): {A[0]:.5f}, {A[1]:.5f}   mean {sum(A)/2:.5f}")
print(f"  FP8 rerun       (n=1): {B[0]:.5f}")
print(f"  difference: {sum(A)/2-B[0]:+.5f}")
print("  n = 2 against 1. No test is possible; this is a description.")
