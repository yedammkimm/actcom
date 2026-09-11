"""ExperimentConfig — spec v1 §5.

INVARIANT-8: ``asdict(cfg)`` is the *sole* source of truth for the ``config``
block of every result JSON. If a scalar is not on this dataclass, it must not
influence a run. This is why ``pack_hooks.py`` and ``dtype_policy.py`` no
longer own a default for ``group_size`` or ``fp8_ratio`` at call time — the
runner injects everything from here.

Method → routing mapping (§2.1):

    method        fp8_ratio (default)   routing
    ------------  --------------------  -------
    standard      —                     (no pack context)
    naive_fp4     0.0                   'none'
    uniform_fp8   1.0                   'none'
    oamp          0.20                  'maxabs'
    random_mixed  0.20                  'random'  (mask_seed required)

``fp8_ratio`` may be overridden on ``oamp`` / ``random_mixed`` runs for
ablation. For ``naive_fp4`` / ``uniform_fp8`` the boundary values are
enforced by :meth:`ExperimentConfig.__post_init__` because the ``pack_hooks``
boundary handling (§2.3) hinges on them.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


# Aliases so most CLI calls stay short: --model 3B / 8B.
MODEL_ALIASES = {
    '3B':    'meta-llama/Llama-3.2-3B-Instruct',
    '8B':    'meta-llama/Llama-3.1-8B-Instruct',
    # 2026-08-18: unsloth's pre-quantized bnb-4bit ships identical quant_config
    # (nf4/bfloat16/double_quant=True, no skip_modules). Online quantization of
    # the meta-llama base blows through 100 GB anon RAM during loading on GB10;
    # the pre-quantized checkpoint loads in ~5-10 min from 35 GB shards.
    '70B':   'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit',
    '70B-online': 'meta-llama/Llama-3.1-70B-Instruct',
    'qwen3B': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen7B': 'Qwen/Qwen2.5-7B-Instruct',
    'mistral7B': 'mistralai/Mistral-7B-Instruct-v0.3',
}


_MODES = ('accuracy', 'memory')
_METHODS = ('standard', 'naive_fp4', 'uniform_fp8', 'random_mixed', 'oamp')
_WEIGHT_QUANTS = ('bf16', 'nf4')
_TASKS = ('gsm8k',)   # extend when task backends are added to oamp.data
_OPTIMIZERS = ('adamw', 'paged_adamw8bit')
_SCHEDULERS = ('cosine', 'linear', 'constant', 'none')


def routing_for(method: str) -> str:
    """Return the pack routing name paired with ``method`` (§2.1)."""
    if method == 'oamp':
        return 'maxabs'
    if method == 'random_mixed':
        return 'random'
    return 'none'


def total_train_steps(*, n_train: int, batch_size: int,
                      grad_accum_steps: int, epochs: int) -> int:
    """Optimizer step count for a run (drop_last per epoch)."""
    per_epoch = n_train // (batch_size * grad_accum_steps)
    return per_epoch * epochs


@dataclass
class ExperimentConfig:
    """Every parameter that can alter a run's numerical result lives here.

    Non-defaulted fields (top of the dataclass) must be provided by the caller
    so the intent of a run is explicit. Everything else defaults to the paper
    condition (Llama-3B, GSM8K, 2 epochs, cosine, warmup 3%).
    """

    # -------- required (no default) --------
    mode: str
    method: str
    model_id: str
    weight_quant: str
    seed: int

    # -------- compression --------
    fp8_ratio: float = 0.20
    group_size: int = 128
    min_numel: int = 1024
    mask_seed: Optional[int] = None
    # Off by default: cross-step storage_ptr recycling causes NaN (2026-08-14).
    # See oamp.pack_hooks.PackHooks for the full cache-safety analysis.
    dedupe: bool = False
    # GACT §5: unbiased stochastic rounding for FP4 quant. Off by default so
    # existing runs stay comparable; turn on to test whether γ variance drops.
    pack_stochastic_rounding: bool = False
    sr_seed: Optional[int] = None
    # Precision routing for 4-D saved tensors (attention head-views + rotary).
    # Gradient-error probe (2026-08-16) showed FP4 body compression of these
    # tensors corrupts backward gradients (cos ~0.4, norm ratio x3), while
    # 'fp8' recovers to cos>=0.99 and drops norm ratio to 1.03. 'skip' bypasses
    # compression entirely for 4-D. 'fp4' preserves legacy γ behaviour.
    pack_4d_mode: str = 'fp4'

    # 4-bit body encoding: 'int4' (uniform signed 4-bit) or 'e2m1' (non-uniform
    # FP4 grid {0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}). Grad probe
    # (2026-08-20) showed E2M1 improves Q/K cos 0.38 -> 0.71 at same 4-bit budget.
    body_encoding: str = 'e2m1'

    # Skip the optimizer step when any LoRA grad is non-finite (GradScaler
    # semantics). Off only to measure what the safeguard prevents.
    nonfinite_skip: bool = True

    # -------- dtype policy --------
    bf16_rmsnorm: bool = True
    # Gated separately so the dtype decomposition can attribute memory to the
    # prepare_model_for_kbit_training upcast and to PEFT's fp32 adapters apart.
    bf16_params: bool = True     # norms / embed / lm_head cast
    bf16_lora: bool = True       # PEFT LoRA A/B cast

    # -------- LoRA --------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # -------- task --------
    task: str = 'gsm8k'

    # -------- training (paper condition = GSM8K 7473 × 2 epochs) --------
    n_train: int = 7473
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    padding: bool = False
    # INVARIANT-8 (2026-08-18): when data_packing=True, training samples are
    # concatenated (EOS-separated) and chunked to `packed_seq_len`. The eval
    # loop is untouched (still per-sample fewshot). Runs with data_packing True
    # vs False MUST NOT be conflated in the results index.
    data_packing: bool = False
    packed_seq_len: int = 2048
    grad_clip: float = 1.0
    scheduler: str = 'cosine'
    # Legacy benchmark_multiseed (paper Standard ~58%) uses CosineAnnealingLR
    # with no warmup. Table 19's "3% warmup" is a documentation error; keep 0.0
    # here so production runs stay comparable to published numbers.
    warmup_ratio: float = 0.0
    optimizer: str = 'adamw'

    # -------- accuracy eval --------
    eval_samples: int = 500
    eval_fewshot: int = 8
    eval_max_new_tokens: int = 256

    # -------- memory sweep (single (B, L) per process for allocator isolation) --------
    mem_seq_len: int = 4096
    mem_batch_size: int = 4
    mem_steps: int = 100
    mem_warmup: int = 10
    # In-process watchdog for GB10 unified memory (2026-08-18). If reserved
    # crosses this before docker cgroup sends SIGKILL, we mark_oom and exit
    # gracefully. Keep strictly below the docker --memory ceiling.
    mem_abort_gb: float = 100.0

    # -------- runtime --------
    gc_enabled: bool = False
    checkpoint_every: int = 50
    output_dir: str = 'results'
    output_path: Optional[str] = None    # explicit JSON path; None => auto-naming

    # -------- validation --------

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"ExperimentConfig.mode must be one of {_MODES}, got {self.mode!r}")
        if self.method not in _METHODS:
            raise ValueError(f"ExperimentConfig.method must be one of {_METHODS}, got {self.method!r}")
        if self.weight_quant not in _WEIGHT_QUANTS:
            raise ValueError(f"ExperimentConfig.weight_quant must be one of {_WEIGHT_QUANTS}, "
                             f"got {self.weight_quant!r}")
        if self.task not in _TASKS:
            raise ValueError(f"ExperimentConfig.task must be one of {_TASKS}, got {self.task!r}")

        # Method → fp8_ratio invariants (spec §2.3).
        if self.method == 'naive_fp4' and self.fp8_ratio != 0.0:
            raise ValueError(
                "method='naive_fp4' requires fp8_ratio=0.0 (spec §2.3 boundary). "
                f"Got fp8_ratio={self.fp8_ratio}.")
        if self.method == 'uniform_fp8' and self.fp8_ratio != 1.0:
            raise ValueError(
                "method='uniform_fp8' requires fp8_ratio=1.0 (spec §2.3 boundary). "
                f"Got fp8_ratio={self.fp8_ratio}.")
        if self.method in ('oamp', 'random_mixed'):
            if not (0.0 < self.fp8_ratio < 1.0):
                raise ValueError(
                    f"method={self.method!r} requires 0 < fp8_ratio < 1, got {self.fp8_ratio}.")

        if self.method == 'random_mixed' and self.mask_seed is None:
            raise ValueError(
                "method='random_mixed' requires mask_seed (spec §2.4 INVARIANT-6). "
                "Pass an integer via ExperimentConfig(mask_seed=...).")
        # Fallback: SR shares nothing with mask stream; if user didn't set a
        # dedicated sr_seed, derive it from the run seed. PackHooks itself still
        # ValueError-s if it receives None — double-defense (INVARIANT-2 pattern).
        if self.pack_stochastic_rounding and self.sr_seed is None:
            self.sr_seed = self.seed
        # gact_affine's per-group affine quant is stochastic by design; give it
        # a reproducible RNG stream tied to the run seed if the user didn't set
        # one explicitly.
        if self.body_encoding == 'gact_affine' and self.sr_seed is None:
            self.sr_seed = self.seed
        if self.pack_4d_mode not in ('fp4', 'fp8', 'skip', 'chan_int4'):
            raise ValueError(
                f"pack_4d_mode must be 'fp4' | 'fp8' | 'skip' | 'chan_int4', "
                f"got {self.pack_4d_mode!r}.")
        if self.body_encoding not in ('int4', 'e2m1', 'gact_affine', 'gact_affine_det'):
            raise ValueError(
                f"body_encoding must be 'int4' | 'e2m1' | 'gact_affine' | "
                f"'gact_affine_det', got {self.body_encoding!r}.")

        if self.data_packing and self.packed_seq_len <= 0:
            raise ValueError(
                f"data_packing=True requires packed_seq_len > 0, got {self.packed_seq_len}.")

        # Mode-specific required fields.
        if self.mode == 'memory':
            if self.mem_seq_len <= 0:
                raise ValueError("mode='memory' requires mem_seq_len > 0.")
            if self.mem_batch_size <= 0:
                raise ValueError("mode='memory' requires mem_batch_size > 0.")
            if self.mem_steps <= 0:
                raise ValueError("mode='memory' requires mem_steps > 0.")
        if self.mode == 'accuracy':
            if self.eval_samples <= 0:
                raise ValueError("mode='accuracy' requires eval_samples > 0.")
            if self.eval_max_new_tokens <= 0:
                raise ValueError("mode='accuracy' requires eval_max_new_tokens > 0.")
            if self.n_train <= 0:
                raise ValueError("mode='accuracy' requires n_train > 0.")

        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"ExperimentConfig.optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}")
        if self.scheduler not in _SCHEDULERS:
            raise ValueError(f"ExperimentConfig.scheduler must be one of {_SCHEDULERS}, got {self.scheduler!r}")

        # Sanity on scalars that would otherwise silently corrupt runs.
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be in [0, 1].")

    # -------- convenience --------

    @property
    def total_train_steps(self) -> int:
        return total_train_steps(
            n_train=self.n_train, batch_size=self.batch_size,
            grad_accum_steps=self.grad_accum_steps, epochs=self.epochs)

    @property
    def routing(self) -> str:
        return routing_for(self.method)

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


# ==================================================================
# CLI
# ==================================================================

def _add_config_args(p: argparse.ArgumentParser) -> None:
    # Identification
    p.add_argument('--mode', required=True, choices=_MODES)
    p.add_argument('--method', required=True, choices=_METHODS)
    p.add_argument('--model', default='3B',
                   help='Model alias (3B/8B/70B/...) or full HF id via --model_id.')
    p.add_argument('--model_id', default=None,
                   help='Full HF model id; overrides --model alias.')
    p.add_argument('--weight_quant', default='nf4', choices=_WEIGHT_QUANTS)
    p.add_argument('--seed', type=int, default=42)

    # Compression
    p.add_argument('--fp8_ratio', type=float, default=None,
                   help='0.0=naive_fp4, 1.0=uniform_fp8. Method sets a default when omitted.')
    p.add_argument('--group_size', type=int, default=128)
    p.add_argument('--min_numel', type=int, default=1024)
    p.add_argument('--mask_seed', type=int, default=None)
    p.add_argument('--dedupe', action='store_true', default=False,
                   help='EXPERIMENTAL. Cache packed tensors by storage identity. '
                        'Cross-step pointer recycling caused NaN at step 7 on '
                        '3B/L=4096 (2026-08-14). Not safe for training runs.')
    p.add_argument('--stochastic_rounding', dest='pack_stochastic_rounding',
                   action='store_true', default=False,
                   help='Use stochastic rounding for FP4 quant (unbiased). '
                        'GACT §5 style; off by default.')
    p.add_argument('--sr_seed', type=int, default=None,
                   help='Optional. Defaults to --seed when --stochastic_rounding is set.')
    p.add_argument('--pack_4d_mode', choices=['fp4', 'fp8', 'skip', 'chan_int4'], default='fp4',
                   help="Precision routing for 4-D saved tensors (attention "
                        "head-views). 'fp4' = legacy γ; 'fp8' = gradient-recovery "
                        "config; 'skip' = no compression on 4-D.")
    p.add_argument('--body_encoding',
                   choices=['int4', 'e2m1', 'gact_affine', 'gact_affine_det'],
                   default='e2m1',
                   help="4-bit body encoding. 'int4' = uniform signed [-7,+7]; "
                        "'e2m1' = FP4 non-uniform grid (default 2026-08-20); "
                        "'gact_affine' = GACT-style per-group affine (min/max + "
                        "zp) with stochastic rounding; 'gact_affine_det' = same "
                        "affine grid without SR (deterministic).")

    # dtype
    p.add_argument('--nonfinite_skip', dest='nonfinite_skip', action='store_true', default=True)
    p.add_argument('--no_nonfinite_skip', dest='nonfinite_skip', action='store_false')
    p.add_argument('--bf16_rmsnorm', dest='bf16_rmsnorm', action='store_true', default=True)
    p.add_argument('--no_bf16_rmsnorm', dest='bf16_rmsnorm', action='store_false')
    p.add_argument('--bf16_params', dest='bf16_params', action='store_true', default=True)
    p.add_argument('--no_bf16_params', dest='bf16_params', action='store_false')
    p.add_argument('--bf16_lora', dest='bf16_lora', action='store_true', default=True)
    p.add_argument('--no_bf16_lora', dest='bf16_lora', action='store_false')

    # LoRA
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', nargs='+',
                   default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # Task
    p.add_argument('--task', default='gsm8k', choices=_TASKS)

    # Training
    p.add_argument('--n_train', type=int, default=7473)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--max_seq_len', type=int, default=512)
    p.add_argument('--padding', action='store_true', default=False)
    p.add_argument('--data_packing', action='store_true', default=False,
                   help='Concatenate GSM8K samples (EOS-separated) and chunk to '
                        '--packed_seq_len. Eval loop unchanged. INVARIANT-8.')
    p.add_argument('--packed_seq_len', type=int, default=2048,
                   help='Chunk size when --data_packing is set.')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--scheduler', default='cosine')
    # Legacy benchmark_multiseed had no warmup; keep 0.0 to match paper numbers.
    p.add_argument('--warmup_ratio', type=float, default=0.0)
    p.add_argument('--optimizer', default='adamw')

    # Accuracy eval
    p.add_argument('--eval_samples', type=int, default=500)
    p.add_argument('--eval_fewshot', type=int, default=8)
    p.add_argument('--eval_max_new_tokens', type=int, default=256)

    # Memory sweep
    p.add_argument('--mem_seq_len', type=int, default=4096)
    p.add_argument('--mem_batch_size', type=int, default=4)
    p.add_argument('--mem_steps', type=int, default=100)
    p.add_argument('--mem_warmup', type=int, default=10)
    p.add_argument('--mem_abort_gb', type=float, default=100.0,
                   help='In-process reserved-memory kill threshold (GB). Must be '
                        'below the docker --memory limit to catch the ceiling '
                        'before cgroup SIGKILL. 2026-08-18 GB10 pilot.')

    # Runtime
    p.add_argument('--gc', dest='gc_enabled', action='store_true', default=False)
    p.add_argument('--checkpoint_every', type=int, default=50)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--output', dest='output_path', default=None,
                   help='Explicit output JSON path; overrides --output_dir auto-naming.')
    # Legacy V1 reproduction preset (test_dtype_policy_3way.py Arm-4 conditions):
    #   lora_dropout=0.0, optimizer=paged_adamw8bit (NF4) / adamw (BF16),
    #   scheduler=constant, warmup_ratio=0.0.
    p.add_argument('--legacy_repro', action='store_true', default=False,
                   help='Match test_dtype_policy_3way (Arm 4) training conditions verbatim.')


def build_from_cli(argv: Optional[List[str]] = None) -> ExperimentConfig:
    """Parse CLI args and construct :class:`ExperimentConfig`.

    ``fp8_ratio`` is filled in from the method default (0.0 / 0.20 / 1.0) when
    the user does not pass ``--fp8_ratio``.
    """
    p = argparse.ArgumentParser(
        description='OAMP experiment runner — spec v1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_config_args(p)
    args = p.parse_args(argv)

    # Resolve model alias
    if args.model_id is None:
        if args.model in MODEL_ALIASES:
            model_id = MODEL_ALIASES[args.model]
        else:
            # Treat --model as a full id if it doesn't match an alias
            model_id = args.model
    else:
        model_id = args.model_id

    # Method-defaulted fp8_ratio
    if args.fp8_ratio is None:
        fp8_ratio = {
            'standard':     0.20,       # unused (no pack)
            'naive_fp4':    0.0,
            'uniform_fp8':  1.0,
            'oamp':         0.20,
            'random_mixed': 0.20,
        }[args.method]
    else:
        fp8_ratio = args.fp8_ratio

    # --legacy_repro preset (only overrides fields the user did NOT pass explicitly).
    lora_dropout = args.lora_dropout
    optimizer = args.optimizer
    scheduler = args.scheduler
    warmup_ratio = args.warmup_ratio
    if args.legacy_repro:
        # test_dtype_policy_3way: lora_dropout=0.0, PagedAdamW8bit for NF4 (bnb),
        # AdamW for BF16, no scheduler, no warmup.
        lora_dropout = 0.0
        optimizer = 'paged_adamw8bit' if args.weight_quant == 'nf4' else 'adamw'
        scheduler = 'constant'
        warmup_ratio = 0.0

    return ExperimentConfig(
        mode=args.mode,
        method=args.method,
        model_id=model_id,
        weight_quant=args.weight_quant,
        seed=args.seed,
        fp8_ratio=fp8_ratio,
        group_size=args.group_size,
        min_numel=args.min_numel,
        mask_seed=args.mask_seed,
        dedupe=args.dedupe,
        pack_stochastic_rounding=args.pack_stochastic_rounding,
        sr_seed=args.sr_seed,
        pack_4d_mode=args.pack_4d_mode,
        body_encoding=args.body_encoding,
        nonfinite_skip=args.nonfinite_skip,
        bf16_rmsnorm=args.bf16_rmsnorm,
        bf16_params=args.bf16_params, bf16_lora=args.bf16_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=lora_dropout, lora_targets=list(args.lora_targets),
        task=args.task,
        n_train=args.n_train, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        max_seq_len=args.max_seq_len, padding=args.padding,
        data_packing=args.data_packing, packed_seq_len=args.packed_seq_len,
        grad_clip=args.grad_clip, scheduler=scheduler,
        warmup_ratio=warmup_ratio, optimizer=optimizer,
        eval_samples=args.eval_samples, eval_fewshot=args.eval_fewshot,
        eval_max_new_tokens=args.eval_max_new_tokens,
        mem_seq_len=args.mem_seq_len,
        mem_batch_size=args.mem_batch_size,
        mem_steps=args.mem_steps, mem_warmup=args.mem_warmup,
        mem_abort_gb=args.mem_abort_gb,
        gc_enabled=args.gc_enabled,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        output_path=args.output_path,
    )
