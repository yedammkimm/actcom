"""Diagnose which 4-D tensors reach PackHooks.pack in a single fresh
Llama-3.2-3B forward+backward. Answers the question: does attn_4d filter
compress attention weights (B, H, L, L)? If yes -> MATH backend, softmax
output is materialised and packed; if not -> FLASH backend (expected).

Usage:
  docker exec hma-container bash -c "cd /app/HMA_Project && \
    python oamp_train_engine/audit/probe_pack_4d_shapes.py"

Output: shape histogram sorted by total elements packed, annotated with
the module-attribution role (from Linear .in/.out hooks). Unknown-role
entries are the "internal" tensors born inside the attention block.
"""
import os
import sys
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/app/hf_cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

import torch

from oamp.pack_hooks import make_pack_hooks
from oamp_train_engine.audit.gradient_error_probe import (
    load_arm4_fresh, register_attribution, _bucket_role, B, L, DEV,
)


def main():
    torch.manual_seed(0)
    ids = torch.randint(0, 128256, (B, L), device=DEV)

    m, tok, param_ptrs = load_arm4_fresh()
    print(f"[env] transformers _attn_implementation = "
          f"{m.base_model.model.config._attn_implementation}", flush=True)
    print(f"[env] SDPA backend = default (auto-select)", flush=True)

    vocab = m.config.vocab_size
    ctx = make_pack_hooks('oamp', fp8_ratio=0.20, group_size=128, min_numel=1024,
                         skip_last_dims={vocab}, param_ptrs=param_ptrs)
    ptr_map = {}
    _handles = register_attribution(m, ptr_map)  # keep alive

    hist_4d = defaultdict(lambda: {'count': 0, 'numel': 0,
                                    'roles': defaultdict(int)})
    hist_3d = defaultdict(lambda: {'count': 0, 'numel': 0})
    original_pack = ctx.pack

    def probe_pack(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:
                shape = tuple(tensor.shape)
                ptr = tensor.untyped_storage().data_ptr()
                role = _bucket_role(ptr_map.get(ptr, 'unknown'))
                hist_4d[shape]['count'] += 1
                hist_4d[shape]['numel'] += tensor.numel()
                hist_4d[shape]['roles'][role] += 1
            elif tensor.dim() == 3:
                hist_3d[tuple(tensor.shape)]['count'] += 1
                hist_3d[tuple(tensor.shape)]['numel'] += tensor.numel()
        return original_pack(tensor)

    ctx.pack = probe_pack

    with ctx:
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()

    print("\n=== 4-D tensor shape histogram (packed in one forward+backward) ===")
    print(f"{'shape':<32} {'count':>6} {'numel':>14}  roles")
    for shape, stats in sorted(hist_4d.items(), key=lambda x: -x[1]['numel']):
        roles_str = ', '.join(f'{r}:{c}'
                              for r, c in sorted(stats['roles'].items()))
        print(f"  {str(shape):<30} {stats['count']:>6}  {stats['numel']:>12,}"
              f"   {{ {roles_str} }}")

    total_4d_count = sum(s['count'] for s in hist_4d.values())
    total_4d_numel = sum(s['numel'] for s in hist_4d.values())
    print(f"\n  TOTAL 4D:  count={total_4d_count:>6}  numel={total_4d_numel:,}")

    # Check for BxHxLxL (attention weights = softmax output).
    seq_len_sq_shapes = [s for s in hist_4d
                         if len(s) == 4 and s[-1] == s[-2]]
    if seq_len_sq_shapes:
        print("\n[!] Found (*, *, L, L) shapes = softmax attention weights:")
        for s in seq_len_sq_shapes:
            print(f"    {s}  -> MATH backend (FLASH does NOT materialise this)")
        print("\n  Implication: attn_4d filter DID compress softmax output.")
        print("               Experiment 1's attn_4d cos=0.34 includes softmax weights.")
    else:
        print("\n[OK] No (*, *, L, L) shapes found -> FLASH backend confirmed.")
        print("     softmax output is fused into SDPA and never materialised.")
        print("     Experiment 1's attn_4d cos=0.34 is due to Q/K only (+ rotary neg).")

    print(f"\n=== 3-D tensor histogram (top-5 by numel) ===")
    for shape, stats in sorted(hist_3d.items(), key=lambda x: -x[1]['numel'])[:5]:
        print(f"  {str(shape):<30} count={stats['count']:>4}  "
              f"numel={stats['numel']:>12,}")


if __name__ == '__main__':
    main()
