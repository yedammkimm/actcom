"""Log every tensor that hits NativeOAMPHooks.pack for one forward+backward step.

Answers questions:
  Q1: What axis does _pack_bilevel group along? (implicit from flattened-order pack)
  Q2: What tensors reach pack()? (shape / dim / numel / branch)
  Q3: How does the 4D path handle non-divisible sizes?

Sweep dimensions:
  --sdpa_backend {flash,math,default}  — force SDPA backend
  --skip_head                           — exclude classification head (last dim == vocab_size)
"""

import argparse
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NativeOAMPHooks lives in legacy until oamp/pack_hooks.py lands.
# TODO(pack_hooks): delete this insert + switch to `from oamp.pack_hooks import PackHooks`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'benchmarks')))

from benchmark_native_packing_vram import NativeOAMPHooks  # noqa: E402
from oamp.sdpa_utils import probe_backend_availability, normalize_attention_mask  # noqa: E402

CACHE_DIR = "/app/hf_cache"


def load_nf4(model_name):
    """Load NF4 base after prepare_model_for_kbit_training (all-fp32 non-quant modules).
    Callers do post-get_peft_model bf16 casting themselves so LoRA is initialized in fp32
    just like the 3-way dtype policy study."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0},
        cache_dir=CACHE_DIR, local_files_only=True,
    )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    return m


def apply_arm3_dtype(peft_model):
    """Arm 3 policy: cast RMSNorm/embed/lm_head/LoRA to bf16 after get_peft_model."""
    import torch.nn as nn
    for name, module in peft_model.named_modules():
        cls_name = type(module).__name__
        if 'RMSNorm' in cls_name or 'LayerNorm' in cls_name:
            for p in module.parameters(recurse=False):
                p.data = p.data.to(torch.bfloat16)
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                p.data = p.data.to(torch.bfloat16)
        if name.endswith('lm_head') and isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
    for n, p in peft_model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.data = p.data.to(torch.bfloat16)


class LoggingNativeOAMPHooks(NativeOAMPHooks):
    """Wraps pack() to record every tensor's classification.

    Adds two features on top of NativeOAMPHooks:
      - Splits the ambiguous "small or non-float" label so we can tell which
        condition caught each tensor.
      - Optional parameter filter: if the tensor's storage pointer is in
        ``param_ptrs``, we label it as 'skip (param)' and never pack.
    """

    def __init__(self, *args, param_ptrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []
        self.param_ptrs = param_ptrs if param_ptrs is not None else set()
        # GACT §5.3: pack_hook fires multiple times for aliased tensors (q/k/v from same input).
        self._seen_ptrs = set()

    def pack(self, tensor):
        entry = {
            'shape': tuple(tensor.shape),
            'dim': tensor.dim(),
            'numel': tensor.numel(),
            'dtype': str(tensor.dtype),
            'is_float': tensor.is_floating_point(),
            'last_dim': int(tensor.shape[-1]) if tensor.dim() > 0 else None,
        }
        try:
            entry['storage_ptr'] = tensor.untyped_storage().data_ptr()
        except Exception:
            entry['storage_ptr'] = tensor.data_ptr() if tensor.numel() > 0 else 0

        entry['is_duplicate'] = entry['storage_ptr'] in self._seen_ptrs
        self._seen_ptrs.add(entry['storage_ptr'])

        is_param = entry['storage_ptr'] in self.param_ptrs

        if is_param:
            entry['branch'] = 'skip (param)'
        elif tensor.numel() < self.min_numel:
            entry['branch'] = 'skip (small)'
        elif not tensor.is_floating_point():
            entry['branch'] = 'skip (non-float)'
        elif tensor.dim() > 0 and tensor.shape[-1] in self.skip_last_dims:
            entry['branch'] = 'skip (head/vocab)'
        elif tensor.dim() > 0 and tensor.shape[-1] % self.group_size != 0:
            entry['branch'] = 'skip (misaligned)'
        elif tensor.dim() <= 3:
            entry['branch'] = 'bilevel (2D/3D)'
        else:
            entry['branch'] = 'fp4 (4D+)'
        gs = self.group_size
        entry['numel_div_gs'] = entry['numel'] % gs == 0
        self.log.append(entry)

        if is_param:
            return tensor
        return super().pack(tensor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--sdpa_backend', choices=['flash', 'math', 'default'], default='default',
                   help='Force SDPA backend via sdpa_kernel context.')
    p.add_argument('--pass_attn_mask', action='store_true', default=False,
                   help='If set, pass explicit all-ones attention_mask (Transformers 4D path).')
    p.add_argument('--skip_head', action='store_true', default=False,
                   help='Exclude tensors whose last dim == vocab_size from packing (GACT §6.1 style).')
    p.add_argument('--restore_bf16', action='store_true', default=False,
                   help='Revert prepare_model_for_kbit_training fp32 cast on RMSNorm/embed/lm_head. LoRA stays fp32.')
    p.add_argument('--no_pack', action='store_true', default=False,
                   help='Run forward+backward WITHOUT NativeOAMPHooks context. Baseline peak alloc measurement.')
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda')

    cfg = AutoConfig.from_pretrained(args.model, cache_dir=CACHE_DIR, local_files_only=True)
    vocab_size = cfg.vocab_size

    base = load_nf4(args.model)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none',
    )
    peft_model = get_peft_model(base, lora)

    if args.restore_bf16:
        apply_arm3_dtype(peft_model)

    peft_model.train()
    if hasattr(peft_model, 'gradient_checkpointing_disable'):
        peft_model.gradient_checkpointing_disable()

    input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids, dtype=torch.long) if args.pass_attn_mask else None

    from torch.nn.attention import SDPBackend, sdpa_kernel
    import contextlib

    def do_forward():
        if args.pass_attn_mask:
            return peft_model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        return peft_model(input_ids=input_ids, labels=input_ids)

    skip_last_dims = [vocab_size] if args.skip_head else []

    # Snapshot every parameter's storage pointer BEFORE the forward pass so we
    # can detect if a saved-for-backward tensor is actually a model parameter.
    param_ptrs = set()
    for p in peft_model.parameters():
        try:
            param_ptrs.add(p.untyped_storage().data_ptr())
        except Exception:
            if p.numel() > 0:
                param_ptrs.add(p.data_ptr())
    print(f"[env] param_ptrs collected: {len(param_ptrs)} unique storage ptrs", flush=True)

    hooks = LoggingNativeOAMPHooks(fp8_ratio=0.20, skip_last_dims=skip_last_dims, param_ptrs=param_ptrs)
    if args.sdpa_backend == 'flash':
        ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    elif args.sdpa_backend == 'math':
        ctx = sdpa_kernel([SDPBackend.MATH])
    else:
        ctx = contextlib.nullcontext()

    backend_probe = probe_backend_availability()
    print(f"[env] probe_backend_availability() (causal, no mask) = {backend_probe}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    if args.no_pack:
        with ctx:
            out = do_forward()
            loss = out.loss
            loss.backward()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[NO_PACK L={args.seq_len} sdpa={args.sdpa_backend} pass_attn_mask={args.pass_attn_mask}]", flush=True)
        print(f"  loss={loss.item():.4f}  peak_alloc={peak_gb:.2f} GB  (baseline; hooks disabled)", flush=True)
        return

    with ctx, hooks:
        out = do_forward()
        loss = out.loss
        loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n[L={args.seq_len} sdpa={args.sdpa_backend} pass_attn_mask={args.pass_attn_mask} skip_head={args.skip_head} vocab={vocab_size}]", flush=True)
    print(f"  loss={loss.item():.4f}  n_pack_calls={len(hooks.log)}  peak_alloc={peak_gb:.2f} GB", flush=True)

    from collections import Counter, defaultdict
    by_signature = defaultdict(list)
    for e in hooks.log:
        sig = (e['dim'], e['branch'], e['shape'], e['dtype'], e['numel'], e['numel_div_gs'])
        by_signature[sig].append(e)

    # dtype -> bytes/element on GPU
    _DTYPE_BYTES = {
        'torch.float32': 4, 'torch.float': 4,
        'torch.float16': 2, 'torch.half': 2,
        'torch.bfloat16': 2,
        'torch.float64': 8, 'torch.double': 8,
        'torch.uint8': 1, 'torch.int8': 1,
        'torch.int16': 2, 'torch.short': 2,
        'torch.int32': 4, 'torch.int': 4,
        'torch.int64': 8, 'torch.long': 8,
        'torch.bool': 1,
    }
    def _bpe(dtype_str):
        return _DTYPE_BYTES.get(dtype_str, 2)

    print(f"\n{'=' * 128}")
    print(f"{'dim':>4} {'branch':<28} {'shape':<36} {'dtype':<15} {'numel':>12} {'div':>4} {'n':>4}")
    print(f"{'-' * 128}")
    for sig, entries in sorted(by_signature.items(), key=lambda kv: (kv[0][0], kv[0][4])):
        dim, branch, shape, dtype, numel, div = sig
        short_dtype = dtype.replace('torch.', '')
        print(f"{dim:>4} {branch:<28} {str(shape):<36} {short_dtype:<15} {numel:>12} {str(div):>4} {len(entries):>4}")
    print(f"{'=' * 128}")

    total = len(hooks.log)
    branches = Counter(e['branch'] for e in hooks.log)
    print(f"\nTotal pack calls: {total}")
    for b, n in branches.most_common():
        print(f"  {b:<32} {n:>5} ({n/total*100:.1f}%)")

    # dtype breakdown for all intercepted tensors
    dtype_bytes = Counter()
    dtype_counts = Counter()
    for e in hooks.log:
        b = e['numel'] * _bpe(e['dtype'])
        dtype_bytes[e['dtype']] += b
        dtype_counts[e['dtype']] += 1
    print(f"\nDtype distribution across intercepted tensors:")
    for dt, cnt in dtype_counts.most_common():
        print(f"  {dt:<20} n={cnt:>4}  bytes={dtype_bytes[dt]/1e9:.3f} GB")

    def bytes_of(branch, unique_only=True):
        seen = set()
        total = 0
        for e in hooks.log:
            if e['branch'] != branch:
                continue
            if unique_only:
                if e['storage_ptr'] in seen:
                    continue
                seen.add(e['storage_ptr'])
            total += e['numel'] * _bpe(e['dtype'])
        return total

    branches_all = ['bilevel (2D/3D)', 'fp4 (4D+)', 'skip (head/vocab)',
                    'skip (misaligned)', 'skip (small)', 'skip (non-float)', 'skip (param)']
    b_unique = {b: bytes_of(b, unique_only=True)  for b in branches_all}
    b_raw    = {b: bytes_of(b, unique_only=False) for b in branches_all}
    b_bilevel = b_unique['bilevel (2D/3D)']
    b_fp4     = b_unique['fp4 (4D+)']
    b_head    = b_unique['skip (head/vocab)']
    b_misal   = b_unique['skip (misaligned)']
    b_small   = b_unique['skip (small)']
    b_nonf    = b_unique['skip (non-float)']
    b_param   = b_unique['skip (param)']
    b_total_unique = sum(b_unique.values())
    b_total_raw    = sum(b_raw.values())

    print(f"\nApprox intercepted activation bytes  "
          f"[unique={b_total_unique/1e9:.2f} GB | raw={b_total_raw/1e9:.2f} GB "
          f"| dup_ratio={b_total_raw/max(1e-9,b_total_unique):.2f}x]:")
    for b in branches_all:
        u, r = b_unique[b], b_raw[b]
        pct_u = u/b_total_unique*100 if b_total_unique>0 else 0
        dup = r/max(1e-9,u)
        print(f"  {b:<18} unique={u/1e9:5.2f} GB ({pct_u:4.1f}%)  raw={r/1e9:5.2f} GB  x{dup:.2f}")

    b_total = b_total_unique

    # Effective bits per element uses ACTUAL dtype size as the source bits.
    # bilevel: 4.8 bits/elem (fp8_ratio*8 + (1-fp8_ratio)*4).
    # fp4:     4.0 bits/elem.
    # skipped tensors retain their source bit-width.
    bits_bilevel = 0.20 * 8 + 0.80 * 4
    bits_fp4     = 4.0
    if b_total > 0:
        def _unique_entries():
            seen = set()
            for e in hooks.log:
                if e['storage_ptr'] in seen:
                    continue
                seen.add(e['storage_ptr'])
                yield e
        uniq = list(_unique_entries())
        total_numel = sum(e['numel'] for e in uniq)
        weighted_src_bits = sum(e['numel'] * _bpe(e['dtype']) * 8 for e in uniq) / max(1, total_numel)
        packed_numel = sum(e['numel'] for e in uniq if e['branch'] in ('bilevel (2D/3D)', 'fp4 (4D+)'))
        packed_target_bits = (
            (sum(e['numel'] for e in uniq if e['branch'] == 'bilevel (2D/3D)') * bits_bilevel +
             sum(e['numel'] for e in uniq if e['branch'] == 'fp4 (4D+)') * bits_fp4)
            / packed_numel if packed_numel > 0 else float('nan')
        )
        stored_bytes = (
            sum(e['numel'] for e in uniq if e['branch'] == 'bilevel (2D/3D)') * bits_bilevel / 8 +
            sum(e['numel'] for e in uniq if e['branch'] == 'fp4 (4D+)') * bits_fp4 / 8 +
            b_head + b_misal + b_small + b_nonf + b_param
        )
        savings_pct = (1 - stored_bytes / b_total) * 100 if b_total > 0 else 0
        print(f"\n[unique-dedup] Weighted source bits/elem across ALL intercepted tensors: {weighted_src_bits:.3f}")
        print(f"[unique-dedup] Target bits/elem across PACKED tensors only (4.8 for bilevel, 4.0 for fp4): {packed_target_bits:.3f}")
        print(f"[unique-dedup] Approx stored bytes after packing: {stored_bytes/1e9:.2f} GB "
              f"({savings_pct:.1f}% saved vs original dtype)")

    print("\nNon-divisible numels (would have crashed pre-fix):")
    any_nd = False
    for e in hooks.log:
        if not e['numel_div_gs'] and e['branch'] in ('bilevel (2D/3D)', 'fp4 (4D+)'):
            print(f"  shape={e['shape']}  numel={e['numel']}  numel%128={e['numel']%128}  branch={e['branch']}")
            any_nd = True
    if not any_nd:
        print("  (none)")


if __name__ == '__main__':
    main()
