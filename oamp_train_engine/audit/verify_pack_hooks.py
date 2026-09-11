"""Spec v1 §10 verification for oamp.pack_hooks.PackHooks.

Runs four gates:
  V5      : fp8_ratio=0.0 (naive_fp4) matches legacy NaiveFP4AllHooks on peak+loss.
  V6      : parity check — forward loss is bit-exact with/without pack context.
  ERR-1   : routing='random' + mask_seed=None -> ValueError.
  ERR-2   : missing skip_last_dims / param_ptrs -> TypeError.

Runs on a small toy config (B=2, L=512) so it finishes in <2 minutes.
Uses meta-llama/Llama-3.2-3B-Instruct in BF16 with LoRA (matches audit setup).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/app/hf_cache")

# Make legacy hooks importable for the V5 comparison — we do NOT ship any
# runtime dependency on legacy; this is a one-off verification.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "legacy", "benchmarks"))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from oamp.pack_hooks import PackHooks, make_pack_hooks, collect_param_ptrs
from benchmark_native_packing_vram import NaiveFP4AllHooks, NativeOAMPHooks

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "/app/hf_cache"
DEVICE = torch.device("cuda:0")
B, L = 2, 512


def build_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      bias='none')
    m = get_peft_model(m, lcfg)
    return m, tok


def run_one_step(model, hooks_ctx):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device=DEVICE)
    if hooks_ctx is None:
        out = model(input_ids=ids, labels=ids)
    else:
        with hooks_ctx:
            out = model(input_ids=ids, labels=ids)
    loss = out.loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    model.zero_grad(set_to_none=True)
    return float(loss.item()), peak


def clone_model(m):
    """Reload fresh so seeded RNG produces identical LoRA init."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    return build_model()


# ------------------------------------------------------------
# ERR-1 / ERR-2
# ------------------------------------------------------------

def check_input_validation():
    print("\n[ERR-1] routing='random' without mask_seed -> ValueError")
    try:
        PackHooks(
            fp8_ratio=0.2, routing='random',
            group_size=128, min_numel=1024,
            skip_last_dims=set(), param_ptrs=set(),
        )
    except ValueError as e:
        print(f"  OK: raised ValueError: {e}")
    else:
        raise AssertionError("PackHooks did NOT raise ValueError for random+no seed.")

    print("\n[ERR-2a] missing skip_last_dims -> TypeError")
    try:
        PackHooks(
            fp8_ratio=0.2, routing='maxabs',
            group_size=128, min_numel=1024,
            param_ptrs=set(),
        )
    except TypeError as e:
        print(f"  OK: raised TypeError: {e}")
    else:
        raise AssertionError("PackHooks did NOT raise TypeError for missing skip_last_dims.")

    print("\n[ERR-2b] missing param_ptrs -> TypeError")
    try:
        PackHooks(
            fp8_ratio=0.2, routing='maxabs',
            group_size=128, min_numel=1024,
            skip_last_dims=set(),
        )
    except TypeError as e:
        print(f"  OK: raised TypeError: {e}")
    else:
        raise AssertionError("PackHooks did NOT raise TypeError for missing param_ptrs.")


# ------------------------------------------------------------
# V5: naive_fp4 (fp8_ratio=0.0) parity with legacy NaiveFP4AllHooks
# ------------------------------------------------------------

def check_v5():
    print("\n[V5] naive_fp4: PackHooks(fp8_ratio=0.0) vs legacy NaiveFP4AllHooks")
    vocab = None

    # New PackHooks path
    m, _ = build_model()
    vocab = m.config.vocab_size
    ptrs = collect_param_ptrs(m)
    new_ctx = make_pack_hooks('naive_fp4',
                              group_size=128, min_numel=1024,
                              skip_last_dims={vocab}, param_ptrs=ptrs)
    loss_new, peak_new = run_one_step(m, new_ctx)
    stats_new = dict(new_ctx.pack_stats)
    del m, new_ctx
    torch.cuda.empty_cache()

    # Legacy path (same seed + fresh model)
    m, _ = build_model()
    ptrs = collect_param_ptrs(m)
    legacy_ctx = NaiveFP4AllHooks(group_size=128, min_numel=1024,
                                  skip_last_dims={vocab}, param_ptrs=ptrs)
    loss_legacy, peak_legacy = run_one_step(m, legacy_ctx)
    stats_legacy = dict(legacy_ctx.pack_stats)
    del m, legacy_ctx
    torch.cuda.empty_cache()

    d_loss = abs(loss_new - loss_legacy)
    d_peak = abs(peak_new - peak_legacy)
    print(f"  new     loss={loss_new:.6f}  peak={peak_new:.3f} GB  stats={stats_new}")
    print(f"  legacy  loss={loss_legacy:.6f}  peak={peak_legacy:.3f} GB  stats={stats_legacy}")
    print(f"  |Δloss|={d_loss:.2e}   |Δpeak|={d_peak*1024:.1f} MB")
    # Loss must be bit-exact (backward FP4 dequant is deterministic).
    assert d_loss <= 1e-6, f"V5 FAIL: loss mismatch {d_loss:.2e}"
    # Peak may differ by allocator noise; require <50 MB.
    assert d_peak * 1024 < 50.0, f"V5 FAIL: peak drift {d_peak*1024:.1f} MB"
    # Filter counts must match on all shared branches.
    for k in ('kept', 'skip_small', 'skip_head', 'skip_non_float',
              'skip_misaligned', 'skip_param'):
        assert stats_new[k] == stats_legacy[k], (
            f"V5 FAIL: {k} differs new={stats_new[k]} legacy={stats_legacy[k]}")
    # kept must map 1:1 to uniform (new) and fp4_all (legacy).
    assert stats_new['uniform'] == stats_legacy['fp4_all'], (
        f"V5 FAIL: uniform={stats_new['uniform']} vs fp4_all={stats_legacy['fp4_all']}")
    assert stats_new['bilevel'] == 0, "V5 FAIL: naive_fp4 should not use bilevel"
    print("  V5 PASS")


# ------------------------------------------------------------
# V6: parity check — forward loss identical with/without pack context
# ------------------------------------------------------------

def check_v6():
    print("\n[V6] parity: forward loss with/without OAMP pack context (bit-exact)")
    m, _ = build_model()
    vocab = m.config.vocab_size
    ptrs = collect_param_ptrs(m)
    ctx = make_pack_hooks('oamp', fp8_ratio=0.20,
                          group_size=128, min_numel=1024,
                          skip_last_dims={vocab}, param_ptrs=ptrs)

    # eval() disables dropout; keep grad enabled so hooks actually fire.
    m.eval()
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (B, L), device=DEVICE)
    # Forward only — no backward. γ is backward-only so forward must be bit-exact.
    loss_off = m(input_ids=ids, labels=ids).loss
    with ctx:
        loss_on = m(input_ids=ids, labels=ids).loss
    d = abs(float(loss_off.item()) - float(loss_on.item()))
    print(f"  loss_off={loss_off.item():.10f}")
    print(f"  loss_on ={loss_on.item():.10f}")
    print(f"  |Δloss| ={d:.2e}   pack_stats={dict(ctx.pack_stats)}")
    assert ctx.pack_stats['kept'] > 0, "V6 FAIL: pack context did not fire (pack_stats=0)"
    assert d == 0.0, f"V6 FAIL: forward not bit-exact (Δ={d:.2e})"
    print("  V6 PASS")


# ------------------------------------------------------------
# U1: uniform_fp8 round-trip precision
# ------------------------------------------------------------

def check_u1_uniform_fp8_roundtrip():
    print("\n[U1] uniform_fp8 pack -> unpack round-trip precision")
    ctx = PackHooks(
        fp8_ratio=1.0, routing='none',
        group_size=128, min_numel=1024,
        skip_last_dims=set(), param_ptrs=set(),
    )
    # Realistic activation-scale bf16 tensor.
    torch.manual_seed(0)
    x = torch.randn(4, 4096, 128, dtype=torch.bfloat16, device=DEVICE) * 0.5
    packed = ctx._pack_uniform(x, bits=8)
    y = ctx._unpack_uniform_fp8(packed)
    assert y.shape == x.shape, f"U1 FAIL: shape {y.shape} != {x.shape}"
    assert y.dtype == x.dtype, f"U1 FAIL: dtype {y.dtype} != {x.dtype}"
    err = (y.float() - x.float()).abs()
    signal_energy = float((x.float() ** 2).mean().sqrt())
    noise_energy  = float((err ** 2).mean().sqrt())
    sqnr_db = 20.0 * (torch.log10(torch.tensor(signal_energy / max(noise_energy, 1e-12)))).item()
    max_abs = float(err.max())
    print(f"  numel={x.numel()}  max_abs={max_abs:.4e}  "
          f"signal_rms={signal_energy:.4f}  noise_rms={noise_energy:.4e}  SQNR={sqnr_db:.1f} dB")
    # FP8 e4m3 with per-group amax scale on gaussian data: SQNR ≥ 30 dB.
    assert sqnr_db >= 30.0, f"U1 FAIL: SQNR={sqnr_db:.1f} dB below FP8 tolerance"
    assert noise_energy < 0.10 * signal_energy, "U1 FAIL: RMS noise exceeds 10% of signal"
    print("  U1 PASS")


# ------------------------------------------------------------
# B1: bilevel degenerate branch corrects pack_stats
# ------------------------------------------------------------

def check_b1_bilevel_degenerate():
    print("\n[B1] bilevel degenerate (K >= num_groups) reroutes stats to 'uniform'")
    ctx = PackHooks(
        fp8_ratio=0.99, routing='maxabs',
        group_size=128, min_numel=128,
        skip_last_dims=set(), param_ptrs=set(),
    )
    # Small tensor: 1 group only -> K = max(1, int(1*0.99)) = 1 >= num_groups.
    x = torch.randn(128, dtype=torch.bfloat16, device=DEVICE)
    _ = ctx.pack(x)
    print(f"  stats={dict(ctx.pack_stats)}")
    assert ctx.pack_stats['kept'] == 1, f"B1 FAIL: kept={ctx.pack_stats['kept']}"
    assert ctx.pack_stats['bilevel'] == 0, (
        f"B1 FAIL: bilevel={ctx.pack_stats['bilevel']} (should be corrected to 0)")
    assert ctx.pack_stats['uniform'] == 1, (
        f"B1 FAIL: uniform={ctx.pack_stats['uniform']} (should be 1 after correction)")
    print("  B1 PASS")


# ------------------------------------------------------------
# DP1: dedupe cache
# ------------------------------------------------------------

def check_dp1_dedupe():
    print("\n[DP1] dedupe=True: same storage packed twice -> one miss + one hit")

    ctx_off = PackHooks(fp8_ratio=0.20, routing='maxabs',
                        group_size=128, min_numel=1024,
                        skip_last_dims=set(), param_ptrs=set(), dedupe=False)
    ctx_on  = PackHooks(fp8_ratio=0.20, routing='maxabs',
                        group_size=128, min_numel=1024,
                        skip_last_dims=set(), param_ptrs=set(), dedupe=True)

    # Realistic Q/K/V-style share: three views of the same storage.
    x = torch.randn(4, 4096, 3072, dtype=torch.bfloat16, device=DEVICE)
    v1, v2, v3 = x, x.view_as(x), x[:, :, :]     # all same storage_ptr + shape

    for c in (ctx_off, ctx_on):
        c._dedupe_cache = {}     # forward-pass reset would do this
        for v in (v1, v2, v3):
            _ = c.pack(v)

    print(f"  off  stats={dict(ctx_off.pack_stats)}")
    print(f"  on   stats={dict(ctx_on.pack_stats)}")

    # Both must have kept=3 and the same bilevel/uniform totals.
    assert ctx_off.pack_stats['kept'] == 3
    assert ctx_on.pack_stats['kept'] == 3
    assert ctx_on.pack_stats['bilevel'] == ctx_off.pack_stats['bilevel']
    assert ctx_on.pack_stats['uniform'] == ctx_off.pack_stats['uniform']
    # Off has no dedupe counters.
    assert ctx_off.pack_stats['dedupe_hit'] == 0
    assert ctx_off.pack_stats['dedupe_miss'] == 0
    # On must show 1 miss + 2 hits.
    assert ctx_on.pack_stats['dedupe_miss'] == 1, (
        f"DP1 FAIL: miss={ctx_on.pack_stats['dedupe_miss']} (expected 1)")
    assert ctx_on.pack_stats['dedupe_hit'] == 2, (
        f"DP1 FAIL: hit={ctx_on.pack_stats['dedupe_hit']} (expected 2)")

    # Different views (transpose) must not hit even with same storage_ptr.
    x2 = torch.randn(128, 128, dtype=torch.bfloat16, device=DEVICE)
    c = PackHooks(fp8_ratio=0.20, routing='maxabs',
                  group_size=128, min_numel=128,
                  skip_last_dims=set(), param_ptrs=set(), dedupe=True)
    c._dedupe_cache = {}
    _ = c.pack(x2)
    _ = c.pack(x2.t())   # same storage, different stride
    assert c.pack_stats['dedupe_miss'] == 2, (
        f"DP1 FAIL: transposed view collided (miss={c.pack_stats['dedupe_miss']})")
    assert c.pack_stats['dedupe_hit'] == 0, (
        f"DP1 FAIL: transposed view false-hit (hit={c.pack_stats['dedupe_hit']})")
    print("  DP1 PASS")


# ------------------------------------------------------------
# SR1: stochastic rounding is unbiased + sr_seed required
# ------------------------------------------------------------

def check_sr1_stochastic_rounding():
    print("\n[SR1] stochastic_rounding=True: E[q(x)] ~= x (unbiased)")

    # Missing sr_seed -> ValueError
    try:
        PackHooks(fp8_ratio=0.20, routing='maxabs',
                  group_size=128, min_numel=1024,
                  skip_last_dims=set(), param_ptrs=set(),
                  stochastic_rounding=True)
    except ValueError as e:
        print(f"  OK: ValueError without sr_seed: {str(e)[:80]}...")
    else:
        raise AssertionError("SR1 FAIL: no ValueError for missing sr_seed")

    # Actual quantization statistics.
    ctx = PackHooks(fp8_ratio=0.20, routing='maxabs',
                    group_size=128, min_numel=1024,
                    skip_last_dims=set(), param_ptrs=set(),
                    stochastic_rounding=True, sr_seed=0)
    ctx_det = PackHooks(fp8_ratio=0.20, routing='maxabs',
                        group_size=128, min_numel=1024,
                        skip_last_dims=set(), param_ptrs=set())

    torch.manual_seed(1)
    x = torch.randn(8, 128, dtype=torch.bfloat16, device=DEVICE) * 0.5

    # Deterministic path — repeatable single quant.
    p1, s1 = ctx_det._quantize_fp4_groups(x)
    p2, s2 = ctx_det._quantize_fp4_groups(x)
    assert torch.equal(p1, p2), "SR1 FAIL: deterministic path not repeatable"

    # SR path: sample many quantizations, average dequantized value.
    from oamp.pack_hooks import unpack_uint4, _FP4_MAX
    x_f = x.float()
    N = 64
    acc = torch.zeros_like(x_f)
    for _ in range(N):
        p, s = ctx._quantize_fp4_groups(x)
        # dequantize
        x_u = unpack_uint4(p).reshape(-1, 128).to(torch.float32) - _FP4_MAX
        acc += (x_u * s.float().unsqueeze(-1)).view_as(x_f)
    mean_dq = acc / N
    bias = (mean_dq - x_f).abs().mean().item()
    xmax = x_f.abs().max().item()
    print(f"  |E[q(x)] - x|_mean = {bias:.4e}   x max abs = {xmax:.3f}")
    # Deterministic quantizer's per-element error is bounded by scale/2. With
    # scale = absmax/7, per-group err is roughly xmax/14. Averaging N times an
    # unbiased quantizer, mean bias should be ~ xmax/(14 * sqrt(N)).
    tol = xmax / 14.0 / (N ** 0.5) * 3.0    # 3x factor for bf16 rounding noise
    assert bias < tol, f"SR1 FAIL: |bias|={bias:.2e} exceeds {tol:.2e}"

    # Boundary test: post-round clamp would bias abs-max elements. Build a
    # group whose max equals 7.0 exactly after bf16 cast, so scale = 1.0 and
    # y contains y=±7.0 exactly. With clamp-before-SR (prob=0 at ±7), E[q] at
    # those positions must equal ±7.0 exactly. Also include an interior value
    # (5.5, exact in bf16 at that magnitude) that DOES need rounding to
    # confirm SR is stochastic.
    row = [7.0, -7.0, 5.5, -5.5, 3.5, -3.5, 0.0, 0.0] * 16   # len=128, bf16-exact
    xb = torch.tensor([row], dtype=torch.bfloat16, device=DEVICE)
    xb_f = xb.float()
    N2 = 1000
    acc_b = torch.zeros_like(xb_f)
    for _ in range(N2):
        p, s = ctx._quantize_fp4_groups(xb)
        x_u = unpack_uint4(p).reshape(-1, 128).to(torch.float32) - _FP4_MAX
        acc_b += (x_u * s.float().unsqueeze(-1)).view_as(xb_f)
    mean_b = acc_b / N2

    boundary_mask = (xb_f.abs() == 7.0)
    interior_mask = (xb_f.abs() == 5.5)                       # requires rounding
    boundary_err = (mean_b[boundary_mask] - xb_f[boundary_mask]).abs().max().item()
    interior_err = (mean_b[interior_mask] - xb_f[interior_mask]).abs().max().item()
    print(f"  boundary (y=±7)   max|E-x| = {boundary_err:.4e}   (fp32 ULP noise expected)")
    print(f"  interior (y=±5.5) max|E-x| = {interior_err:.4e}   "
          f"(~1/sqrt({N2})={1/N2**0.5:.3f})")
    # Post-round clamp bug would give O(0.5) systematic bias at the boundary;
    # fp32 accumulator ULP over N2 sums is ~1e-6.
    assert boundary_err < 1e-3, \
        f"SR1 FAIL: boundary bias {boundary_err:.2e} — post-round clamp bug back?"
    # Interior expected variance: sqrt(p(1-p)/N) at p=0.5 is 1/(2*sqrt(N))=0.016;
    # bf16 noise in scale/floor can widen this to ~5%. Fail if grossly wider.
    assert interior_err < 0.10, \
        f"SR1 FAIL: interior bias {interior_err:.2e} — SR not converging"
    print("  SR1 PASS")


# ------------------------------------------------------------
# SR2: SR-on OAMP forward is still bit-exact (γ is backward-only)
#      and the mask/SR generators are independent instances.
# ------------------------------------------------------------

def check_sr2_sr_on_parity():
    print("\n[SR2] SR-on OAMP forward parity + generator isolation")
    m, _ = build_model()
    vocab = m.config.vocab_size
    ptrs = collect_param_ptrs(m)

    # random_mixed exercises BOTH generators (mask + SR).
    ctx = make_pack_hooks('random_mixed', fp8_ratio=0.20,
                          group_size=128, min_numel=1024,
                          skip_last_dims={vocab}, param_ptrs=ptrs,
                          mask_seed=42,
                          stochastic_rounding=True, sr_seed=42)

    m.eval()
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (B, L), device=DEVICE)
    loss_off = m(input_ids=ids, labels=ids).loss
    with ctx:
        loss_on = m(input_ids=ids, labels=ids).loss
    d = abs(float(loss_off.item()) - float(loss_on.item()))
    print(f"  Δloss = {d:.2e}   (γ is backward-only; forward must be bit-exact)")
    assert d == 0.0, f"SR2 FAIL: forward not bit-exact with SR on (Δ={d:.2e})"

    # Force both generators to materialize, then verify isolation.
    x = torch.randn(32, 128, dtype=torch.bfloat16, device=DEVICE) * 0.3
    ctx._quantize_fp4_groups(x)                            # creates _sr_gen
    ctx._select_anchor_mask(x, num_groups=32, K=6)         # creates _mask_gen
    assert ctx._sr_gen is not None and ctx._mask_gen is not None, \
        "SR2 FAIL: generators not materialized"
    assert ctx._sr_gen is not ctx._mask_gen, "SR2 FAIL: mask & SR share Generator"
    print(f"  mask_gen id={id(ctx._mask_gen)}  sr_gen id={id(ctx._sr_gen)}  (independent)")
    print("  SR2 PASS")


# ------------------------------------------------------------
# D4-1/2/3: pack_4d_mode dispatches correctly
# ------------------------------------------------------------

def check_d4_pack_4d_mode():
    print("\n[D4] pack_4d_mode dispatches correctly + parity bit-exact")
    from oamp.pack_hooks import PackHooks

    # 4-D tensor that survives all skip filters.
    torch.manual_seed(0)
    x4 = torch.randn(4, 24, 128, 128, dtype=torch.bfloat16, device=DEVICE)

    # (D4-1) invalid mode -> ValueError
    try:
        PackHooks(fp8_ratio=0.20, routing='maxabs',
                  group_size=128, min_numel=1024,
                  skip_last_dims=set(), param_ptrs=set(),
                  pack_4d_mode='fp16')
    except ValueError as e:
        print(f"  OK: ValueError on invalid mode: {str(e)[:70]}...")
    else:
        raise AssertionError("D4-1 FAIL: expected ValueError for pack_4d_mode='fp16'")

    # (D4-2) each mode increments the right counter and returns the right shape.
    for mode, want_uniform, want_fp8, want_skip in [
        ('fp4',  True,  False, False),
        ('fp8',  True,  True,  False),
        ('skip', False, False, True),
    ]:
        ctx = PackHooks(fp8_ratio=0.20, routing='maxabs',
                        group_size=128, min_numel=1024,
                        skip_last_dims=set(), param_ptrs=set(),
                        pack_4d_mode=mode)
        packed = ctx.pack(x4)
        st = ctx.pack_stats
        print(f"  mode={mode!s:<5}  kept={st['kept']}  uniform={st['uniform']}  "
              f"override_fp8_4d={st['override_fp8_4d']}  skip_by_dim={st['skip_by_dim']}")
        if want_skip:
            assert packed is x4, f"D4-2 FAIL: skip mode did not return original tensor"
            assert st['skip_by_dim'] == 1, "D4-2 FAIL: skip_by_dim counter"
            assert st['kept'] == 0, "D4-2 FAIL: skip should not count as kept"
        else:
            assert st['kept'] == 1, "D4-2 FAIL: kept counter"
            assert st['uniform'] == 1, "D4-2 FAIL: uniform branch"
            recovered = ctx.unpack(packed)
            assert recovered.shape == x4.shape, \
                f"D4-2 FAIL: unpack shape {recovered.shape} != {x4.shape}"
        if want_fp8:
            assert st['override_fp8_4d'] == 1, "D4-2 FAIL: override_fp8_4d not incremented"
        else:
            assert st['override_fp8_4d'] == 0, \
                "D4-2 FAIL: override_fp8_4d incremented outside fp8 mode"

    # (D4-3) parity: forward loss identical to gamma_full (fp4) whether
    # pack_4d_mode is 'fp8' or 'skip' (γ is backward-only).
    m, _ = build_model()
    vocab = m.config.vocab_size
    ptrs = collect_param_ptrs(m)
    m.eval()
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (B, L), device=DEVICE)
    loss_off = m(input_ids=ids, labels=ids).loss.item()
    for mode in ('fp4', 'fp8', 'skip'):
        ctx = make_pack_hooks('oamp', fp8_ratio=0.20,
                              group_size=128, min_numel=1024,
                              skip_last_dims={vocab}, param_ptrs=ptrs,
                              pack_4d_mode=mode)
        with ctx:
            loss_on = m(input_ids=ids, labels=ids).loss.item()
        d = abs(loss_off - loss_on)
        print(f"  parity mode={mode!s:<5}  |Δloss|={d:.2e}  "
              f"pack_stats={{'kept':{ctx.pack_stats['kept']}, "
              f"'override_fp8_4d':{ctx.pack_stats['override_fp8_4d']}, "
              f"'skip_by_dim':{ctx.pack_stats['skip_by_dim']}}}")
        assert d == 0.0, f"D4-3 FAIL: forward parity broken under pack_4d_mode={mode}"

    # (D4-4) pack_4d_mode governs 4-D across ALL methods (not just OAMP).
    # naive_fp4 (fp8_ratio=0.0) + pack_4d_mode='fp8' MUST route 4-D to FP8.
    # uniform_fp8 (fp8_ratio=1.0) + pack_4d_mode='fp4' MUST route 4-D to FP4.
    # This is the cross-method implementation principle (2026-08-16).
    combos = [
        ('naive_fp4',    'fp4', 0),        # 4-D FP4 through new branch; counter untouched
        ('naive_fp4',    'fp8', 224),      # 4-D promoted to FP8
        ('uniform_fp8',  'fp4', 0),        # 4-D forced back to FP4 while 3-D stays FP8
        ('uniform_fp8',  'fp8', 224),      # both branches FP8; counter increments
        ('random_mixed', 'fp8', 224),
        ('oamp',         'fp8', 224),      # regression check on prior D4-2 result
    ]
    for method, mode, want_override in combos:
        kwargs = dict(fp8_ratio=0.20, group_size=128, min_numel=1024,
                      skip_last_dims={vocab}, param_ptrs=ptrs,
                      pack_4d_mode=mode)
        if method == 'random_mixed':
            kwargs['mask_seed'] = 0
        ctx = make_pack_hooks(method, **kwargs)
        with ctx:
            _ = m(input_ids=ids, labels=ids).loss.item()
        got = ctx.pack_stats['override_fp8_4d']
        print(f"  D4-4 method={method:<13} mode={mode:<4}  "
              f"override_fp8_4d={got:>3} (want {want_override})")
        assert got == want_override, \
            f"D4-4 FAIL: {method} + {mode}: override_fp8_4d={got}, want {want_override}"
    print("  D4 PASS")


if __name__ == '__main__':
    print("=" * 70)
    print("PackHooks verification (spec v1 §10)")
    print("=" * 70)
    check_input_validation()
    check_v5()
    check_v6()
    check_u1_uniform_fp8_roundtrip()
    check_b1_bilevel_degenerate()
    check_dp1_dedupe()
    check_sr1_stochastic_rounding()
    check_sr2_sr_on_parity()
    check_d4_pack_4d_mode()
    print("\nALL CHECKS PASSED")
