"""PackHooks: the saved_tensors_hooks context that compresses what autograd keeps for backward.

The forward pass is untouched. When autograd saves a tensor, pack() decides
whether and how to compress it, and unpack() dequantizes it when backward
needs it. run_experiment.py checks before every run that the loss with the
hooks on equals the loss with them off.

Which tensors are compressed is decided by a fixed chain of filters. The
order matters:

    1.  parameters (storage pointer in param_ptrs)         stored as is
    2.  tensors with fewer than min_numel elements          stored as is
    3.  non-floating-point tensors                          stored as is
    4.  tensors whose last dim is in skip_last_dims         stored as is
        (the logits, (B*L, vocab); packing them corrupts the gradient,
        which is why this test precedes the rank and alignment tests)
    5.  last dim not a multiple of group_size               stored as is
    5b. 4-D tensors, when pack_4d_mode is 'skip'            stored as is
    6.  4-D tensors, the attention head views and the rotary tables:
        pack_4d_mode 'fp8' stores them as blockwise FP8, 'chan_int4' as
        per-channel INT4, 'fp4' on the 4-bit body grid
    7.  everything else, by method: fp8_ratio <= 0 is uniform 4-bit,
        fp8_ratio >= 1 is uniform FP8, anything between is the anchor mix

Step 6 precedes step 7 so that pack_4d_mode governs the head views whatever
the method is. The paper's main finding is about step 6: compressing the
Q/K head views on a blockwise 4-bit grid drives the gradient cosine to
about 0.4, and FP8 restores it to 0.99.

The method names of run_experiment.py map onto (fp8_ratio, routing):

    naive_fp4      0.0       'none'      the paper's configuration
    uniform_fp8    1.0       'none'
    oamp           e.g. 0.2  'maxabs'    max-abs anchor variant, Section 5.2
    random_mixed   e.g. 0.2  'random'    needs mask_seed

fp8_ratio <= 0 and >= 1 go straight to the uniform packers, so naive_fp4
and uniform_fp8 are never clipped by the max(1, ...) guard of the anchor
code.

body_encoding chooses the 4-bit grid: 'int4' (symmetric integers -7..7),
'e2m1' (the FP4 grid 0, 0.5, 1, 1.5, 2, 3, 4, 6 with a sign bit) or
'gact_affine' (the 4-bit affine quantizer of GACT, for comparison;
'gact_affine_det' is its deterministic-rounding variant).

dedupe caches packed tensors by (storage pointer, shape, stride, offset) so
that aliased saves are packed once. It is off by default and should stay
off in training: the CUDA allocator can hand the same storage pointer to an
unrelated tensor in a later step, and a cache hit then returns stale data
(NaN at step 7, 2026-08-14). The single-forward parity check cannot catch
this.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

try:
    from torchao.prototype.mx_formats.nvfp4_tensor import pack_uint4, unpack_uint4
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PackHooks requires torchao.prototype.mx_formats for pack_uint4/unpack_uint4"
    ) from e


_SENTINEL = object()
_FP8_MAX = 448.0     # torch.float8_e4m3fn absmax
_FP4_MAX = 7.0       # signed 4-bit integer quant range used for INT4 body payload
_E2M1_MAX = 6.0      # signed E2M1 FP4 max magnitude (nvfp4 grid)
_GACT_MAX = 15.0     # unsigned 4-bit range for GACT affine [0, 15]

# E2M1 grid (positive magnitudes; sign is stored as a separate nibble bit).
# Boundaries are midpoints between adjacent levels for nearest-neighbour rounding.
_E2M1_LEVELS_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # 7 midpoints

_ROUTING_CHOICES = ('maxabs', 'random', 'none')
_PACK_4D_MODES = ('fp4', 'fp8', 'skip', 'chan_int4')   # 4-D saved-tensor precision routing
_BODY_ENCODINGS = ('int4', 'e2m1', 'gact_affine', 'gact_affine_det')


class PackHooks:
    """The hooks context. Build it, run forward and backward inside `with hooks:`,
    and read hooks.pack_stats for what happened.

    All arguments are keyword-only. fp8_ratio and routing select the method as
    described in the module docstring; group_size is the block size along the
    last axis and min_numel the size below which a tensor is stored as is.
    skip_last_dims and param_ptrs have no default on purpose: pass
    {config.vocab_size} and the pointer set from apply_dtype_policy, or an
    empty set to opt out explicitly. pack_4d_mode ('fp4', 'fp8', 'chan_int4' or
    'skip') sets the treatment of 4-D saved tensors and body_encoding ('int4',
    'e2m1', 'gact_affine' or 'gact_affine_det') the 4-bit grid. mask_seed is
    required for routing='random' and sr_seed for stochastic_rounding=True;
    each seeds its own CUDA generator so the global RNG stream, which drives
    dropout, is untouched.

    stochastic_rounding replaces round() in the 4-bit quantizers with
    per-element Bernoulli rounding, so the quantizer is unbiased in expectation
    (E[q(x)] = x). GACT argues this is needed for compressed training to remain
    a valid SGD algorithm, since deterministic rounding can accumulate error in
    the same direction step after step. It applies to the 4-bit body only; FP8
    keeps deterministic rounding, because the e4m3 mantissa is fine enough that
    the bias is negligible.

    dedupe is the experimental cache described in the module docstring. Off by
    default; leave it off in training.
    """

    def __init__(self, *,
                 fp8_ratio: float,
                 routing: str,
                 group_size: int,
                 min_numel: int,
                 skip_last_dims=_SENTINEL,
                 param_ptrs=_SENTINEL,
                 mask_seed: Optional[int] = None,
                 dedupe: bool = False,
                 stochastic_rounding: bool = False,
                 sr_seed: Optional[int] = None,
                 pack_4d_mode: str = 'fp4',
                 body_encoding: str = 'e2m1'):
        # The sentinel forces callers to pass the filter sets explicitly.
        if skip_last_dims is _SENTINEL:
            raise TypeError(
                "PackHooks: skip_last_dims is required. "
                "Pass skip_last_dims={config.vocab_size} to exclude the LM head, "
                "or skip_last_dims=set() to opt out explicitly.")
        if param_ptrs is _SENTINEL:
            raise TypeError(
                "PackHooks: param_ptrs is required. "
                "Pass a set of trainable-parameter storage pointers, "
                "or param_ptrs=set() to opt out explicitly.")
        if routing not in _ROUTING_CHOICES:
            raise ValueError(
                f"PackHooks: routing must be one of {_ROUTING_CHOICES}, got {routing!r}.")
        # Random routing needs its own seed.
        if routing == 'random' and mask_seed is None:
            raise ValueError(
                "PackHooks: routing='random' requires mask_seed (dedicated RNG stream).")
        if stochastic_rounding and sr_seed is None:
            raise ValueError(
                "PackHooks: stochastic_rounding=True requires sr_seed (dedicated RNG stream).")
        # gact_affine has its own stochastic rounding, separate from the INT4 body's.
        # Fall back to a fixed seed when the caller set none, so probes can
        # construct the hooks without one.
        if body_encoding == 'gact_affine' and sr_seed is None:
            sr_seed = 42
        if pack_4d_mode not in _PACK_4D_MODES:
            raise ValueError(
                f"PackHooks: pack_4d_mode must be one of {_PACK_4D_MODES}, "
                f"got {pack_4d_mode!r}.")
        if body_encoding not in _BODY_ENCODINGS:
            raise ValueError(
                f"PackHooks: body_encoding must be one of {_BODY_ENCODINGS}, "
                f"got {body_encoding!r}.")

        self.fp8_ratio = float(fp8_ratio)
        self.routing = routing
        self.group_size = int(group_size)
        self.min_numel = int(min_numel)
        self.skip_last_dims = set(skip_last_dims)
        self.param_ptrs = set(param_ptrs)
        self.mask_seed = mask_seed
        self._mask_gen: Optional[torch.Generator] = None    # created lazily on first CUDA call

        self.dedupe = bool(dedupe)
        self._dedupe_cache: dict = {}

        self.stochastic_rounding = bool(stochastic_rounding)
        self.sr_seed = sr_seed
        self._sr_gen: Optional[torch.Generator] = None      # created lazily
        self.pack_4d_mode = pack_4d_mode
        self.body_encoding = body_encoding
        # E2M1 constants are materialised once on __enter__ to avoid a host-to-device copy per tensor.
        self._e2m1_levels_gpu: Optional[torch.Tensor] = None
        self._e2m1_bounds_gpu: Optional[torch.Tensor] = None

        self.pack_stats = {
            'kept': 0,
            'skip_small': 0,
            'skip_head': 0,
            'skip_non_float': 0,
            'skip_misaligned': 0,
            'skip_param': 0,
            'skip_by_dim': 0,           # opt-in via pack_4d_mode='skip'
            'bilevel': 0,
            'uniform': 0,
            'override_fp8_4d': 0,       # opt-in via pack_4d_mode='fp8'
            # Detail counters that decompose `uniform` by tensor rank and bits
            # (2026-08-18). uniform == uniform_fp4_3d + uniform_fp8_3d
            #                     + uniform_fp4_4d + uniform_fp8_4d.
            'uniform_fp4_3d': 0,
            'uniform_fp8_3d': 0,
            'uniform_fp4_4d': 0,
            'uniform_fp8_4d': 0,
            'chan_int4_4d': 0,          # opt-in via pack_4d_mode='chan_int4'
            # Element-count decomposition for effective_bits() (2026-08-19).
            # numel_kept == numel_uniform_fp4_3d + numel_uniform_fp8_3d
            #             + numel_uniform_fp4_4d + numel_uniform_fp8_4d
            #             + numel_bilevel.
            'numel_kept': 0,
            'numel_uniform_fp4_3d': 0,
            'numel_uniform_fp8_3d': 0,
            'numel_uniform_fp4_4d': 0,
            'numel_uniform_fp8_4d': 0,
            'numel_chan_int4_4d': 0,
            'numel_bilevel': 0,
            # Scale-element count, accumulated per branch so effective_bits()
            # can report real overhead instead of assuming every branch uses a
            # group_size-sized block. Blockwise branches contribute
            # ceil(numel/group_size); the per-channel branch contributes D
            # (= H*head_dim) regardless of sequence length.
            'n_scale_elems': 0,
            'dedupe_hit': 0,     # zero when dedupe=False
            'dedupe_miss': 0,
        }

    # Hook entry points

    def pack(self, tensor: torch.Tensor):
        # 1. parameters (storage pointer from untyped_storage)
        if self._is_param(tensor):
            self.pack_stats['skip_param'] += 1
            return tensor
        # 2. small
        if tensor.numel() < self.min_numel:
            self.pack_stats['skip_small'] += 1
            return tensor
        # 3. non-float
        if not tensor.is_floating_point():
            self.pack_stats['skip_non_float'] += 1
            return tensor
        # 4. head / vocab; must precede the rank and alignment checks
        if tensor.dim() > 0 and tensor.shape[-1] in self.skip_last_dims:
            self.pack_stats['skip_head'] += 1
            return tensor
        # 5. misaligned
        if tensor.dim() > 0 and tensor.shape[-1] % self.group_size != 0:
            self.pack_stats['skip_misaligned'] += 1
            return tensor
        # 5b. pack_4d_mode='skip': 4-D saved tensors bypass compression entirely.
        #     Introduced after gradient probe showed head-view compression to FP4
        #     corrupts backward (cos ~0.4, norm x3). Runs before kept++/dedupe so
        #     the pass-through is not counted as compressed.
        if tensor.dim() == 4 and self.pack_4d_mode == 'skip':
            self.pack_stats['skip_by_dim'] += 1
            return tensor

        self.pack_stats['kept'] += 1
        self.pack_stats['numel_kept'] += tensor.numel()

        # Dedupe cache lookup; aliased saves (the shared Q/K/V input) hit here.
        # Key: (storage_ptr, shape, stride, storage_offset) so different views
        # of the same storage are treated as distinct tensors.
        cache_key = None
        if self.dedupe:
            cache_key = self._dedupe_key(tensor)
            hit = self._dedupe_cache.get(cache_key)
            if hit is not None:
                self.pack_stats['dedupe_hit'] += 1
                # branch counters still bump so bilevel+uniform == kept.
                if isinstance(hit, tuple) and hit and hit[0] == 'bilevel':
                    self.pack_stats['bilevel'] += 1
                else:
                    self.pack_stats['uniform'] += 1
                return hit
            self.pack_stats['dedupe_miss'] += 1

        # 6-7. dispatch
        # The 4-D branch comes first so pack_4d_mode governs the head views whatever
        # method the caller is running: compressing the head views to four bits
        # corrupts the backward pass under every method (gradient probe, 2026-08-16),
        # so the FP8 route is a property of the tensor, not of the method.
        if tensor.dim() == 4:
            self.pack_stats['uniform'] += 1
            if self.pack_4d_mode == 'fp8':
                self.pack_stats['override_fp8_4d'] += 1
                self.pack_stats['uniform_fp8_4d'] += 1
                self.pack_stats['numel_uniform_fp8_4d'] += tensor.numel()
                self.pack_stats['n_scale_elems'] += self._n_block_scales(tensor)
                packed = self._pack_uniform(tensor, bits=8)
            elif self.pack_4d_mode == 'chan_int4':
                # Per-channel INT4: the scale axis moves from a group of
                # `group_size` contiguous elements to the channel (H*head_dim),
                # reducing over the token axis instead. Gradient cosine on Q/K
                # rises from 0.37 (blockwise INT4) to ~0.90 at the same four
                # bits, which is what this mode exists to test.
                self.pack_stats['chan_int4_4d'] += 1
                self.pack_stats['numel_chan_int4_4d'] += tensor.numel()
                self.pack_stats['n_scale_elems'] += tensor.shape[1] * tensor.shape[3]
                packed = self._pack_chan_int4_4d(tensor)
            else:  # 'fp4' (default legacy behaviour; 'skip' already returned)
                self.pack_stats['uniform_fp4_4d'] += 1
                self.pack_stats['numel_uniform_fp4_4d'] += tensor.numel()
                self.pack_stats['n_scale_elems'] += self._n_block_scales(tensor)
                packed = self._pack_uniform(tensor, bits=4)
        elif self.fp8_ratio <= 0.0:
            self.pack_stats['uniform'] += 1
            self.pack_stats['uniform_fp4_3d'] += 1
            self.pack_stats['numel_uniform_fp4_3d'] += tensor.numel()
            self.pack_stats['n_scale_elems'] += self._n_block_scales(tensor)
            packed = self._pack_uniform(tensor, bits=4)
        elif self.fp8_ratio >= 1.0:
            self.pack_stats['uniform'] += 1
            self.pack_stats['uniform_fp8_3d'] += 1
            self.pack_stats['numel_uniform_fp8_3d'] += tensor.numel()
            self.pack_stats['n_scale_elems'] += self._n_block_scales(tensor)
            packed = self._pack_uniform(tensor, bits=8)
        else:
            # dim <= 3 with fp8_ratio in (0, 1): bilevel body + anchor mix.
            self.pack_stats['bilevel'] += 1
            self.pack_stats['numel_bilevel'] += tensor.numel()
            self.pack_stats['n_scale_elems'] += self._n_block_scales(tensor)
            packed = self._pack_bilevel(tensor)

        if cache_key is not None:
            self._dedupe_cache[cache_key] = packed
        return packed

    def unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        tag = packed[0]
        if tag == 'bilevel':
            return self._unpack_bilevel(packed)
        if tag == 'uniform_fp4':
            return self._unpack_uniform_fp4(packed)
        if tag == 'uniform_fp8':
            return self._unpack_uniform_fp8(packed)
        if tag == 'chan_int4':
            return self._unpack_chan_int4_4d(packed)
        return packed

    # Context manager

    def __enter__(self):
        # Clear the dedupe cache per forward pass so a storage_ptr reused later
        # in a different training step can't return stale packed data.
        self._dedupe_cache = {}
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        self._ctx.__exit__(*args)

    # Effective bits accounting

    def effective_bits(self) -> dict:
        """Element-weighted bits per element over everything that was packed.

        Nominal bits by branch: 4 for uniform_fp4_{3d,4d}, 8 for uniform_fp8_{3d,4d},
        and 4 + 4 * fp8_ratio for the anchor mix. `element_bits` is the payload
        alone; `byte_bits` adds one fp16 scale (16 bits) per group_size elements,
        the overhead that shows up in checkpoint bytes.
        """
        s = self.pack_stats
        n_fp4  = s['numel_uniform_fp4_3d'] + s['numel_uniform_fp4_4d']
        n_fp8  = s['numel_uniform_fp8_3d'] + s['numel_uniform_fp8_4d']
        n_chan = s.get('numel_chan_int4_4d', 0)
        n_bi   = s['numel_bilevel']
        total = n_fp4 + n_fp8 + n_chan + n_bi
        payload = {
            'element_bits':    None,
            'byte_bits':       None,
            'scale_bits':      None,
            'group_size':      self.group_size,
            'fp8_ratio':       self.fp8_ratio,
            'total_numel':     int(total),
            'numel_fp4':       int(n_fp4),
            'numel_fp8':       int(n_fp8),
            'numel_chan_int4': int(n_chan),
            'numel_bilevel':   int(n_bi),
            'n_scale_elems':   int(s.get('n_scale_elems', 0)),
        }
        if total == 0:
            return payload
        bilevel_bits = 4.0 + 4.0 * self.fp8_ratio
        weighted = (n_fp4 * 4.0 + n_fp8 * 8.0 + n_chan * 4.0
                    + n_bi * bilevel_bits)
        elem = weighted / total
        payload['element_bits'] = float(elem)
        # Scale overhead is measured, not assumed: n_scale_elems is accumulated
        # per branch at pack time, so a per-channel branch (one scale per
        # channel, independent of sequence length) is accounted correctly
        # instead of being charged the blockwise 16/group_size.
        scale_bytes = (2.0 if self.body_encoding in ('gact_affine', 'gact_affine_det')
                       else 1.0)
        n_scales = s.get('n_scale_elems', 0)
        scale_bits = scale_bytes * 16.0 * n_scales / total if n_scales else 0.0
        payload['scale_bits'] = float(scale_bits)
        payload['byte_bits']  = float(elem + scale_bits)
        payload['body_encoding'] = self.body_encoding
        return payload

    # Filters

    def _is_param(self, tensor: torch.Tensor) -> bool:
        if not self.param_ptrs:
            return False
        try:
            return tensor.untyped_storage().data_ptr() in self.param_ptrs
        except Exception:
            # 0-numel tensors don't have a storage
            return tensor.data_ptr() in self.param_ptrs if tensor.numel() > 0 else False

    def _dedupe_key(self, tensor: torch.Tensor):
        try:
            sp = tensor.untyped_storage().data_ptr()
        except Exception:
            sp = tensor.data_ptr() if tensor.numel() > 0 else 0
        return (sp, tuple(tensor.shape), tuple(tensor.stride()), int(tensor.storage_offset()))

    # Uniform packers (the fp8_ratio boundaries)

    def _n_block_scales(self, tensor: torch.Tensor) -> int:
        """Number of scales of a blockwise branch: one per group_size elements, with
        the tail group counted, since the quantizers zero-pad up to a full group.
        """
        gs = self.group_size
        return (tensor.numel() + gs - 1) // gs

    def _pack_chan_int4_4d(self, tensor: torch.Tensor):
        """Per-channel symmetric INT4 for 4-D head views (B, H, L, head_dim).

        The tensor is viewed as (B*L, H*head_dim), channels last and tokens
        leading, and absmax reduces over the token axis, giving one scale per
        channel. The audit probe's version keeps int8 codes and fp32 scales because
        it only measures gradient error; this one stores nibble-packed codes and
        fp16 scales so that the mode really costs four bits per element.

        The scale overhead is 16 * D / (B*L*D) = 16/(B*L) bits per element, against
        16/group_size for the blockwise branches. With this project's GSM8K
        sequences (harmonic-mean length 152 at batch size 1) that is 0.105 against
        0.125, so the two are nearly identical in memory. The mode exists to test
        whether four bits can be made safe, not cheap.
        """
        assert tensor.dim() == 4, "_pack_chan_int4_4d expects (B, H, L, head_dim)"
        B, H, L, dh = tensor.shape
        x_ch = tensor.transpose(1, 2).reshape(B * L, H * dh)        # (BL, D)
        absmax = x_ch.abs().amax(dim=0).clamp(min=1e-8)             # (D,)
        scale = (absmax / _FP4_MAX).to(torch.float16)               # (D,)
        # Clamp before rounding, matching _quantize_int4_groups: a post-round
        # clamp reintroduces bias exactly at the boundary.
        y = (x_ch / scale.to(x_ch.dtype)).clamp(-_FP4_MAX, _FP4_MAX)
        x_u = (y.round() + _FP4_MAX).to(torch.uint8)
        packed = pack_uint4(x_u)
        return ('chan_int4', packed, scale, tensor.shape, tensor.dtype)

    def _unpack_chan_int4_4d(self, packed):
        _, data, scale, shape, dtype = packed
        B, H, L, dh = shape
        x_u = unpack_uint4(data).reshape(B * L, H * dh)
        x = (x_u.to(torch.float32) - _FP4_MAX) * scale.to(torch.float32)
        return x.reshape(B, L, H, dh).transpose(1, 2).to(dtype)

    def _pack_uniform(self, tensor: torch.Tensor, *, bits: int):
        if bits == 4:
            packed, scale, n = self._quantize_fp4(tensor)
            return ('uniform_fp4', packed, scale, tensor.shape, tensor.dtype, n)
        if bits == 8:
            data, scale, n = self._quantize_fp8(tensor)
            return ('uniform_fp8', data, scale, tensor.shape, tensor.dtype, n)
        raise ValueError(f"_pack_uniform: unsupported bits={bits}")

    def _unpack_uniform_fp4(self, packed):
        _, data, scale, shape, dtype, n = packed
        return self._dequantize_fp4(data, scale, n, dtype).reshape(shape)

    def _unpack_uniform_fp8(self, packed):
        _, data, scale, shape, dtype, n = packed
        x = (data.to(torch.bfloat16).float() * scale.float()).to(dtype)
        return x.reshape(-1)[:n].reshape(shape)

    # Anchor packer (0 < fp8_ratio < 1)

    def _pack_bilevel(self, tensor: torch.Tensor):
        orig_shape = tensor.shape
        dtype = tensor.dtype
        gs = self.group_size

        x_flat = tensor.reshape(-1)
        orig_numel = x_flat.numel()
        pad = (gs - (orig_numel % gs)) % gs
        if pad > 0:
            x_flat = F.pad(x_flat, (0, pad))
        x = x_flat.reshape(-1, gs)
        num_groups = x.shape[0]

        # 0 < fp8_ratio < 1 by construction (boundaries handled in pack()).
        K = max(1, int(num_groups * self.fp8_ratio))
        if K >= num_groups:
            # Degenerate: whole tensor becomes an FP8 anchor. pack() already
            # bumped 'bilevel'; correct the stats since we route to uniform.
            self.pack_stats['bilevel'] -= 1
            self.pack_stats['uniform'] += 1
            self.pack_stats['numel_bilevel'] -= orig_numel
            self.pack_stats['numel_uniform_fp8_3d'] += orig_numel
            return self._pack_uniform(tensor, bits=8)

        with torch.no_grad():
            fp8_mask = self._select_anchor_mask(x, num_groups, K)

        # FP8 anchor groups (absmax taken in the source dtype)
        anchor_groups = x[fp8_mask]
        anchor_absmax = anchor_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).float()
        anchor_scale = anchor_absmax / _FP8_MAX
        anchor_fp8 = (anchor_groups.float() / anchor_scale
                      ).to(torch.bfloat16).to(torch.float8_e4m3fn)

        # FP4 body groups
        body = x[~fp8_mask]
        body_packed, body_scale = self._quantize_fp4_groups(body)
        body_n = body.numel()   # already (G_body, gs); no padding to unwind
        del body

        return ('bilevel',
                anchor_fp8, anchor_scale.to(torch.float16),
                body_packed, body_scale, fp8_mask,
                orig_shape, dtype, body_n, orig_numel)

    def _unpack_bilevel(self, packed):
        (_, anchor_fp8, anchor_scale, body_packed, body_scale, fp8_mask,
         orig_shape, dtype, body_n, orig_numel) = packed
        gs = self.group_size
        num_groups = fp8_mask.shape[0]
        out = torch.empty(num_groups, gs, dtype=dtype, device=anchor_fp8.device)
        out[fp8_mask] = (anchor_fp8.to(torch.bfloat16).float()
                         * anchor_scale.float()).to(dtype)
        body = self._dequantize_fp4(body_packed, body_scale, body_n, dtype)
        out[~fp8_mask] = body.reshape(-1, gs)
        out_flat = out.reshape(-1)[:orig_numel]
        return out_flat.reshape(orig_shape)

    def _select_anchor_mask(self, x: torch.Tensor, num_groups: int, K: int) -> torch.Tensor:
        if self.routing == 'maxabs':
            # absmax in the source dtype; only the reduced result is upcast.
            importance = x.abs().amax(dim=-1).float()
        elif self.routing == 'random':
            gen = self._get_mask_generator(x.device)
            importance = torch.rand(num_groups, device=x.device, generator=gen)
        else:  # 'none' should not reach here; the uniform boundary is handled earlier
            importance = x.abs().amax(dim=-1).float()
        _, top_idx = importance.topk(K)
        mask = torch.zeros(num_groups, dtype=torch.bool, device=x.device)
        mask.scatter_(0, top_idx, True)
        return mask

    def _get_mask_generator(self, device: torch.device) -> torch.Generator:
        # Lazy so PackHooks can be constructed before CUDA is ready.
        if self._mask_gen is None:
            self._mask_gen = torch.Generator(device=device).manual_seed(int(self.mask_seed))
        return self._mask_gen

    # 4-bit / FP8 group quantizers

    def _quantize_fp4(self, tensor: torch.Tensor):
        """Dispatch on self.body_encoding: INT4 (symmetric integers), E2M1 (the FP4
        grid) or the GACT affine quantizer. Returns (packed_uint8, scale,
        pre_pad_numel).
        """
        gs = self.group_size
        x = tensor.reshape(-1)
        n = x.numel()
        n_padded = ((n + gs - 1) // gs) * gs
        if n_padded > n:
            x = F.pad(x, (0, n_padded - n))
        if self.body_encoding == 'e2m1':
            packed, scale = self._quantize_e2m1_groups(x.reshape(-1, gs))
        elif self.body_encoding == 'gact_affine':
            packed, scale = self._quantize_gact_affine_groups(x.reshape(-1, gs), stochastic=True)
        elif self.body_encoding == 'gact_affine_det':
            packed, scale = self._quantize_gact_affine_groups(x.reshape(-1, gs), stochastic=False)
        else:
            packed, scale = self._quantize_int4_groups(x.reshape(-1, gs))
        return packed, scale, n

    def _quantize_fp4_groups(self, x_groups: torch.Tensor):
        """Dispatcher used by the anchor packer's body groups; see _quantize_fp4."""
        if self.body_encoding == 'e2m1':
            return self._quantize_e2m1_groups(x_groups)
        if self.body_encoding == 'gact_affine':
            return self._quantize_gact_affine_groups(x_groups, stochastic=True)
        if self.body_encoding == 'gact_affine_det':
            return self._quantize_gact_affine_groups(x_groups, stochastic=False)
        return self._quantize_int4_groups(x_groups)

    def _quantize_int4_groups(self, x_groups: torch.Tensor):
        """Symmetric INT4 with a per-group absmax scale, the grid the earlier code
        called FP4.

        No element count is returned because callers want different ones:
        uniform_fp4 wants the pre-padding count and the anchor body wants G * gs.
        Each computes its own.
        """
        absmax = x_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / _FP4_MAX
        # Clamp BEFORE SR: fp roundoff can push abs-max elements to y=±7.0000...1,
        # and a post-round clamp reintroduces bias exactly at the boundary,
        # defeating E[q(x)] = x. With y clamped to [-7, 7], floor lands in [-7, 6] and q in [-7, 7].
        y = (x_groups / scale).clamp(-_FP4_MAX, _FP4_MAX)
        if self.stochastic_rounding:
            floor_y = torch.floor(y)
            prob = y - floor_y
            gen = self._get_sr_generator(y.device)
            noise = torch.rand(prob.shape, device=y.device, dtype=prob.dtype, generator=gen)
            x_q = floor_y + (noise < prob).to(y.dtype)
        else:
            x_q = y.round()
        x_u = (x_q + _FP4_MAX).to(torch.uint8)
        packed = pack_uint4(x_u)
        return packed, scale.squeeze(-1).to(torch.float16)

    def _quantize_e2m1_groups(self, x_groups: torch.Tensor):
        """E2M1 FP4 with a per-group absmax scale.

        Each nibble is [sign bit | 3-bit magnitude index]; the magnitudes come from
        _E2M1_LEVELS_POS, and sign bit 0 is positive. Both signed zeros map to
        level 0 and dequantize to 0, a harmless duplicate encoding.
        """
        if self.stochastic_rounding:
            raise NotImplementedError(
                "PackHooks: stochastic_rounding is currently only implemented for "
                "body_encoding='int4'.")
        absmax = x_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / _E2M1_MAX
        y = (x_groups / scale).clamp(-_E2M1_MAX, _E2M1_MAX)
        # Materialise E2M1 lookup tables on the target device on first use.
        if self._e2m1_bounds_gpu is None or self._e2m1_bounds_gpu.device != y.device:
            self._e2m1_bounds_gpu = torch.tensor(
                _E2M1_BOUNDARIES, device=y.device, dtype=y.dtype)
            self._e2m1_levels_gpu = torch.tensor(
                _E2M1_LEVELS_POS, device=y.device, dtype=y.dtype)
        sign = torch.signbit(y).to(torch.uint8)
        mag = y.abs()
        idx = torch.bucketize(mag, self._e2m1_bounds_gpu).to(torch.uint8)   # 0..7
        code = (sign << 3) | idx                                            # 0..15
        packed = pack_uint4(code)
        return packed, scale.squeeze(-1).to(torch.float16)

    def _quantize_gact_affine_groups(self, x_groups: torch.Tensor, *, stochastic: bool):
        """The 4-bit affine quantizer of GACT (Chen et al. 2021).

        Per group, (mn, mx) map to an unsigned code in 0..15. Stochastic rounding
        is the paper's default; the deterministic variant is exposed for a
        controlled comparison. Each group stores two fp16 scalars, scale and mn,
        packed into one (G, 2) fp16 tensor so the two-tuple return signature of the
        INT4 and E2M1 quantizers is preserved.
        """
        mn = x_groups.min(dim=-1, keepdim=True).values                      # (G, 1)
        mx = x_groups.max(dim=-1, keepdim=True).values                      # (G, 1)
        scale = (mx - mn).clamp(min=1e-8) / _GACT_MAX                       # (G, 1)
        y = (x_groups - mn) / scale                                         # ∈ [0, 15]
        if stochastic:
            floor_y = torch.floor(y)
            prob = y - floor_y
            gen = self._get_sr_generator(y.device)
            noise = torch.rand(prob.shape, device=y.device, dtype=prob.dtype, generator=gen)
            q = floor_y + (noise < prob).to(y.dtype)
        else:
            q = y.round()
        q = q.clamp(0, _GACT_MAX).to(torch.uint8)
        packed = pack_uint4(q)
        # Stack scale and mn along the last dim into a (G, 2) fp16 tensor. Downstream sees an
        # opaque "scale" tensor of shape (G, 2); dequant reads (scale, mn) via
        # unbind(-1) so the 2-tuple call signature is preserved.
        scale_and_mn = torch.stack(
            [scale.squeeze(-1), mn.squeeze(-1)], dim=-1).to(torch.float16)
        return packed, scale_and_mn

    def _get_sr_generator(self, device: torch.device) -> torch.Generator:
        if self._sr_gen is None:
            self._sr_gen = torch.Generator(device=device).manual_seed(int(self.sr_seed))
        return self._sr_gen

    def _dequantize_fp4(self, packed, scale, original_n: int, dtype):
        """Dispatcher paired with _quantize_fp4."""
        gs = self.group_size
        if self.body_encoding == 'e2m1':
            return self._dequantize_e2m1(packed, scale, original_n, dtype)
        if self.body_encoding in ('gact_affine', 'gact_affine_det'):
            return self._dequantize_gact_affine(packed, scale, original_n, dtype)
        # INT4 path
        x_u = unpack_uint4(packed).reshape(-1, gs)
        x_q = x_u.to(dtype) - _FP4_MAX
        x = x_q * scale.to(dtype).unsqueeze(-1)
        return x.reshape(-1)[:original_n]

    def _dequantize_e2m1(self, packed, scale, original_n: int, dtype):
        gs = self.group_size
        code = unpack_uint4(packed).reshape(-1, gs)               # uint8, 0..15
        if self._e2m1_levels_gpu is None or self._e2m1_levels_gpu.device != code.device:
            self._e2m1_levels_gpu = torch.tensor(
                _E2M1_LEVELS_POS, device=code.device, dtype=dtype)
        levels = self._e2m1_levels_gpu.to(dtype)
        sign = (code >> 3) & 1                                    # 0 or 1
        idx  = (code & 0x7).long()                                # 0..7
        mag = levels[idx]                                         # (G, gs)
        y = torch.where(sign.bool(), -mag, mag)
        x = y * scale.to(dtype).unsqueeze(-1)
        return x.reshape(-1)[:original_n]

    def _dequantize_gact_affine(self, packed, scale_and_mn, original_n: int, dtype):
        """Paired with _quantize_gact_affine_groups: split the (G, 2) fp16 tensor into
        scale and mn, then x = q * scale + mn.
        """
        gs = self.group_size
        q = unpack_uint4(packed).reshape(-1, gs)                  # uint8, 0..15
        scale, mn = scale_and_mn.unbind(-1)                       # each (G,)
        x = (q.to(dtype) * scale.to(dtype).unsqueeze(-1)
             + mn.to(dtype).unsqueeze(-1))
        return x.reshape(-1)[:original_n]

    def _quantize_fp8(self, tensor: torch.Tensor):
        gs = self.group_size
        x = tensor.reshape(-1)
        n = x.numel()
        n_padded = ((n + gs - 1) // gs) * gs
        if n_padded > n:
            x = F.pad(x, (0, n_padded - n))
        x = x.reshape(-1, gs)
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).float()
        scale = absmax / _FP8_MAX
        x_scaled = (x.float() / scale).to(torch.bfloat16).to(torch.float8_e4m3fn)
        return x_scaled, scale.to(torch.float16), n


# Factory helpers

def make_pack_hooks(method: str, *,
                    fp8_ratio: float = 0.20,
                    group_size: int = 128,
                    min_numel: int = 1024,
                    skip_last_dims=_SENTINEL,
                    param_ptrs=_SENTINEL,
                    mask_seed: Optional[int] = None,
                    dedupe: bool = False,
                    stochastic_rounding: bool = False,
                    sr_seed: Optional[int] = None,
                    pack_4d_mode: str = 'fp4',
                    body_encoding: str = 'e2m1') -> PackHooks:
    """Build a PackHooks for one of the method names of run_experiment.py.

    'naive_fp4' is fp8_ratio 0.0 with routing 'none', 'uniform_fp8' is 1.0 with
    'none', 'oamp' is the given fp8_ratio with 'maxabs', and 'random_mixed' the
    given fp8_ratio with 'random' (mask_seed required). For 'standard' the
    caller uses contextlib.nullcontext() instead.
    """
    if method == 'standard':
        raise ValueError(
            "make_pack_hooks('standard'): standard uses no pack context; "
            "use contextlib.nullcontext() at the call site.")
    if method == 'naive_fp4':
        r, ratio = 'none', 0.0
    elif method == 'uniform_fp8':
        r, ratio = 'none', 1.0
    elif method == 'oamp':
        r, ratio = 'maxabs', fp8_ratio
    elif method == 'random_mixed':
        r, ratio = 'random', fp8_ratio
    else:
        raise ValueError(f"make_pack_hooks: unknown method {method!r}")
    return PackHooks(
        fp8_ratio=ratio, routing=r,
        group_size=group_size, min_numel=min_numel,
        skip_last_dims=skip_last_dims, param_ptrs=param_ptrs,
        mask_seed=mask_seed, dedupe=dedupe,
        stochastic_rounding=stochastic_rounding, sr_seed=sr_seed,
        pack_4d_mode=pack_4d_mode,
        body_encoding=body_encoding,
    )


def collect_param_ptrs(model) -> set:
    """Collect untyped_storage().data_ptr() of every model parameter.

    Call this after the dtype casts: casting reallocates storage and makes
    earlier pointers stale.
    """
    ptrs = set()
    for p in model.parameters():
        try:
            ptrs.add(p.untyped_storage().data_ptr())
        except Exception:
            if p.numel() > 0:
                ptrs.add(p.data_ptr())
    return ptrs
