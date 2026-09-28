"""Which kernels of the training loop return different bits on repeated calls?

Each operation the 3B training step uses is run on identical inputs `repeats`
times and every output (forward value and every gradient) is compared bitwise
with the first call. An operation that accumulates with atomics gives a
different result within a few repeats; one with a fixed reduction order never
does. The probe is run twice, as two processes, because deterministic mode has
to be set before CUDA initialises:

    python scripts/audit/determinism_ops_probe.py --mode default \
        --output results/determinism/ops_probe__default.json
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/audit/determinism_ops_probe.py \
        --mode deterministic --output results/determinism/ops_probe__deterministic.json

Under deterministic mode an operation that has no deterministic implementation
raises, and the error text is stored in the record: that is the name of a
nondeterminism source. Shapes are those of Llama-3.2-3B at L = 512, batch 1,
LoRA rank 16: 24 query heads, 8 key/value heads, head dimension 128, hidden
3072, MLP 8192, vocabulary 128256.
"""
import argparse
import json
import os
import platform
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['default', 'deterministic'], required=True)
parser.add_argument('--repeats', type=int, default=30)
parser.add_argument('--output', required=True)
args = parser.parse_args()

if args.mode == 'deterministic':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch  # noqa: E402  (after the environment variable)
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

if args.mode == 'deterministic':
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DEV = 'cuda'
B, H, HKV, L, D, HID, MLP, VOCAB, R = 1, 24, 8, 512, 128, 3072, 8192, 128256, 16
BF = torch.bfloat16


def repeat_op(name, make, run):
    """Run `run(*make())` repeats+1 times; compare every output with the first."""
    rec = {'name': name, 'status': 'ok', 'mismatching_repeats': 0, 'first_mismatch': None,
           'max_abs_diff': 0.0, 'error': None}
    try:
        inputs = make()
        ref = [t.detach().clone() for t in run(*inputs)]
        torch.cuda.synchronize()
        for i in range(args.repeats):
            out = [t.detach() for t in run(*inputs)]
            torch.cuda.synchronize()
            if any(not torch.equal(a, b) for a, b in zip(ref, out)):
                rec['mismatching_repeats'] += 1
                if rec['first_mismatch'] is None:
                    rec['first_mismatch'] = i + 1
                rec['max_abs_diff'] = max(rec['max_abs_diff'], max(
                    (a.float() - b.float()).abs().max().item() for a, b in zip(ref, out)))
    except Exception as e:                       # noqa: BLE001
        rec['status'] = 'raised'
        rec['error'] = f'{type(e).__name__}: {e}'[:600]
    tag = rec['status'] if rec['status'] != 'ok' else f"{rec['mismatching_repeats']}/{args.repeats} mismatching"
    print(f"  {name:48s} {tag}" + (f"  (first at {rec['first_mismatch']}, max|diff| {rec['max_abs_diff']:.3e})"
                                   if rec['mismatching_repeats'] else '') +
          (f"  {rec['error'][:120]}" if rec['error'] else ''), flush=True)
    return rec


def mk_qkv():
    return (torch.randn(B, H, L, D, device=DEV, dtype=BF),
            torch.randn(B, HKV, L, D, device=DEV, dtype=BF),
            torch.randn(B, HKV, L, D, device=DEV, dtype=BF))


def sdpa_run(backend):
    def run(q, k, v):
        q = q.clone().requires_grad_(True); k = k.clone().requires_grad_(True); v = v.clone().requires_grad_(True)
        ctx = sdpa_kernel([backend]) if backend is not None else torch.enable_grad()
        with ctx:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o.float().sum().backward()
        return [o, q.grad, k.grad, v.grad]
    return run


def mk_mm():
    return (torch.randn(L, HID, device=DEV, dtype=BF), torch.randn(HID, HID, device=DEV, dtype=BF))


def mm_run(a, w):
    a = a.clone().requires_grad_(True); w = w.clone().requires_grad_(True)
    o = a @ w; o.float().sum().backward(); return [o, a.grad, w.grad]


def mk_lora():
    return (torch.randn(L, HID, device=DEV, dtype=BF), torch.randn(HID, R, device=DEV, dtype=BF),
            torch.randn(R, HID, device=DEV, dtype=BF))


def lora_run(x, A, Bm):
    x = x.clone().requires_grad_(True); A = A.clone().requires_grad_(True); Bm = Bm.clone().requires_grad_(True)
    o = (x @ A) @ Bm; o.float().sum().backward(); return [o, x.grad, A.grad, Bm.grad]


def mk_ce():
    return (torch.randn(L, VOCAB, device=DEV, dtype=torch.float32), torch.randint(0, VOCAB, (L,), device=DEV))


def ce_run(x, y):
    x = x.clone().requires_grad_(True); l = F.cross_entropy(x, y); l.backward(); return [l, x.grad]


def mk_rms():
    return (torch.randn(B, L, HID, device=DEV, dtype=BF), torch.ones(HID, device=DEV, dtype=BF))


def rms_run(x, w):
    x = x.clone().requires_grad_(True); w = w.clone().requires_grad_(True)
    o = F.rms_norm(x, (HID,), w, 1e-5); o.float().sum().backward(); return [o, x.grad, w.grad]


def mk_silu():
    return (torch.randn(B, L, MLP, device=DEV, dtype=BF), torch.randn(B, L, MLP, device=DEV, dtype=BF))


def silu_run(g, u):
    g = g.clone().requires_grad_(True); u = u.clone().requires_grad_(True)
    o = F.silu(g) * u; o.float().sum().backward(); return [o, g.grad, u.grad]


def mk_bnb():
    import bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=BF, quant_type='nf4',
                            compress_statistics=True).to(DEV)
    return (lin, torch.randn(B, L, HID, device=DEV, dtype=BF))


def bnb_run(lin, x):
    x = x.clone().requires_grad_(True); o = lin(x); o.float().sum().backward(); return [o, x.grad]


def mk_opt():
    p0 = torch.randn(HID, R, device=DEV, dtype=BF)
    return (p0, torch.randn_like(p0))


def opt_run(p0, g):
    import bitsandbytes as bnb
    p = torch.nn.Parameter(p0.clone())
    opt = bnb.optim.PagedAdamW8bit([p], lr=2e-4, weight_decay=0.01)
    for _ in range(3):
        p.grad = g.clone(); opt.step()
    return [p.detach().clone()]


def mk_clip():
    return tuple(torch.randn(HID, R, device=DEV, dtype=BF) for _ in range(8))


def clip_run(*gs):
    ps = [torch.nn.Parameter(torch.zeros_like(g)) for g in gs]
    for p, g in zip(ps, gs):
        p.grad = g.clone()
    n = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    return [n] + [p.grad for p in ps]


def mk_dropout():
    return (torch.randn(B, L, HID, device=DEV, dtype=BF),)


def dropout_run(x):
    # Same Philox seed and offset on every call: identical mask expected.
    torch.manual_seed(1234)
    x = x.clone().requires_grad_(True); o = F.dropout(x, 0.05, True); o.float().sum().backward()
    return [o, x.grad]


def mk_pack():
    from oamp.pack_hooks import make_pack_hooks
    hooks = make_pack_hooks('naive_fp4', fp8_ratio=0.0, group_size=128, min_numel=1024,
                            skip_last_dims=set(), param_ptrs=set(), mask_seed=None, dedupe=False,
                            stochastic_rounding=False, sr_seed=None, pack_4d_mode='fp4',
                            body_encoding='e2m1')
    return (hooks, torch.randn(B, L, HID, device=DEV, dtype=BF), torch.randn(B, H, L, D, device=DEV, dtype=BF))


def pack_run(hooks, x3, x4):
    return [hooks.unpack(hooks.pack(x3)), hooks.unpack(hooks.pack(x4))]


def mk_pack_fp8():
    from oamp.pack_hooks import make_pack_hooks
    hooks = make_pack_hooks('naive_fp4', fp8_ratio=0.0, group_size=128, min_numel=1024,
                            skip_last_dims=set(), param_ptrs=set(), mask_seed=None, dedupe=False,
                            stochastic_rounding=False, sr_seed=None, pack_4d_mode='fp8',
                            body_encoding='e2m1')
    return (hooks, torch.randn(B, H, L, D, device=DEV, dtype=BF))


def pack_fp8_run(hooks, x4):
    return [hooks.unpack(hooks.pack(x4))]


torch.manual_seed(0)
print(f"[ops probe] mode={args.mode} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
      f"deterministic={torch.are_deterministic_algorithms_enabled()} "
      f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
t0 = time.time()
ops = [
    ('sdpa default dispatch, causal GQA 24/8, fwd+bwd', mk_qkv, sdpa_run(None)),
    ('sdpa FLASH_ATTENTION fwd+bwd', mk_qkv, sdpa_run(SDPBackend.FLASH_ATTENTION)),
    ('sdpa MATH fwd+bwd', mk_qkv, sdpa_run(SDPBackend.MATH)),
    ('sdpa EFFICIENT_ATTENTION fwd+bwd', mk_qkv, sdpa_run(SDPBackend.EFFICIENT_ATTENTION)),
    ('sdpa CUDNN_ATTENTION fwd+bwd', mk_qkv, sdpa_run(SDPBackend.CUDNN_ATTENTION)),
    ('bf16 matmul (512x3072)@(3072x3072) fwd+bwd', mk_mm, mm_run),
    ('LoRA (x@A)@B, rank 16, fwd+bwd', mk_lora, lora_run),
    ('cross_entropy 512x128256 fwd+bwd', mk_ce, ce_run),
    ('rms_norm fwd+bwd', mk_rms, rms_run),
    ('silu(gate)*up fwd+bwd', mk_silu, silu_run),
    ('dropout p=0.05, fixed Philox seed, fwd+bwd', mk_dropout, dropout_run),
    ('bitsandbytes Linear4bit NF4 fwd+bwd', mk_bnb, bnb_run),
    ('bitsandbytes PagedAdamW8bit, 3 steps', mk_opt, opt_run),
    ('clip_grad_norm_ over 8 tensors', mk_clip, clip_run),
    ('oamp pack/unpack: E2M1 body (3-D) and 4-D FP4', mk_pack, pack_run),
    ('oamp pack/unpack: 4-D FP8', mk_pack_fp8, pack_fp8_run),
]
records = [repeat_op(n, mk, run) for n, mk, run in ops]
out = {
    'mode': args.mode,
    'repeats': args.repeats,
    'shapes': dict(B=B, heads=H, kv_heads=HKV, L=L, head_dim=D, hidden=HID, mlp=MLP, vocab=VOCAB, lora_r=R),
    'env': {
        'hostname': platform.node(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    },
    'elapsed_s': time.time() - t0,
    'ops': records,
}
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
print(f"[ops probe] wrote {args.output} in {out['elapsed_s']:.1f}s", flush=True)
