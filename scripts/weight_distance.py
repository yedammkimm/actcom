"""Weight-space distance between LoRA endpoints, and its relation to
perplexity distance and to item-level behavioural disagreement.

The comparison is on the product dW = (alpha/r) B A, never on A and B
separately: LoRA factors are identified only up to an invertible G acting as
(GA, B G^-1), so a distance between factors is not a distance between models.

dW is 3072x3072 at 112 sites, so it is never materialised. For the Frobenius
inner product,

    <dW_i, dW_j> = s^2 tr( (B_j^T B_i) (A_i A_j^T) )

and both bracketed factors are 16x16, which makes the whole matrix cheap.

Row-space principal angles use the row space of A, which is the row space of BA
whenever B has full column rank; the reported rank check confirms that here.
"""
import json, os, itertools, math, random
import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
S = ROOT + 'results/_wd/'
rows = json.load(open(S + 'rows.json'))
sub = [r for r in rows if r['arm'] in ('A', 'B')]
tags = [r['tag'] for r in sub]
ALPHA, R = 32, 16
s2 = (ALPHA / R) ** 2

def load(ap):
    out = {}
    # stored in bfloat16; promote through torch, since numpy has no bf16 dtype
    with safe_open(os.path.join(ROOT, ap, 'adapter_model.safetensors'), framework='pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                base = k.split('.lora_')[0]
                side = 'A' if '.lora_A.' in k else 'B'
                out.setdefault(base, {})[side] = f.get_tensor(k).to(torch.float64).numpy()
    return out

print('loading 16 adapters ...', flush=True)
W = {t: load(r['ap']) for t, r in zip(tags, sub)}
mods = sorted(set.intersection(*[set(v) for v in W.values()]))
print(f'{len(mods)} common modules; example shapes '
      f'A {W[tags[0]][mods[0]]["A"].shape} B {W[tags[0]][mods[0]]["B"].shape}')

ranks = [np.linalg.matrix_rank(W[tags[0]][m]['B']) for m in mods[:8]]
print(f'B column rank on the first 8 modules: {ranks} (r = {R})')

# Gram matrix of dW over all 16 adapters, accumulated module by module
n = len(tags)
G = np.zeros((n, n))
for m in mods:
    A = [W[t][m]['A'] for t in tags]
    B = [W[t][m]['B'] for t in tags]
    for i in range(n):
        for j in range(i, n):
            v = s2 * float(np.trace((B[j].T @ B[i]) @ (A[i] @ A[j].T)))
            G[i, j] += v
            if i != j:
                G[j, i] += v
nrm = np.sqrt(np.diag(G))
Dw = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Dw[i, j] = math.sqrt(max(G[i, i] + G[j, j] - 2 * G[i, j], 0.0))
np.save(S + 'weight_dist.npy', Dw)
np.save(S + 'weight_norm.npy', nrm)
json.dump(tags, open(S + 'weight_tags.json', 'w'))
print('\n||dW|| per adapter (Frobenius, all 112 sites)')
for t, v in zip(tags, nrm):
    print(f'  {t:12s} {v:10.3f}')

# principal angles between row spaces, pooled over modules
print('\nrow-space principal angles (mean cos over 112 modules x 16 angles)')
Q = {}
for m in mods:
    for t in tags:
        Q[(t, m)] = np.linalg.qr(W[t][m]['A'].T)[0]
PA = np.zeros((n, n))
for m in mods:
    for i in range(n):
        for j in range(i + 1, n):
            sv = np.linalg.svd(Q[(tags[i], m)].T @ Q[(tags[j], m)], compute_uv=False)
            v = float(np.clip(sv, 0, 1).mean())
            PA[i, j] += v; PA[j, i] += v
PA /= len(mods)
np.save(S + 'princ_angle.npy', PA)
print('  saved')
