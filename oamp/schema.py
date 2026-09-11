"""Result JSON schema + validators + checkpoint helpers.

Spec v1 §6. Every ``run_experiment.py`` invocation writes exactly one JSON
that conforms to :data:`RESULT_TOP_KEYS`. Fields that don't apply to the run
mode (e.g. ``results.gsm8k_accuracy`` in ``mode='memory'``) may be ``None``
but the key itself must be present so downstream tooling doesn't KeyError.

INVARIANT-9: ``validate_schema(config, env)`` is called before any GPU work,
so a mis-specified run fails in seconds instead of after a 4-hour training.

INVARIANT-10: both ``peak_allocated_gb`` and ``peak_reserved_gb`` are logged
(GB10 is unified memory; system pressure follows ``reserved``).

INVARIANT-11: mid-run progress is appended to a JSONL sidecar every
``checkpoint_every`` steps so a killed process still leaves useful data.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .env import REQUIRED_ENV_FIELDS


# ---------- top-level keys required in every result JSON ----------

RESULT_TOP_KEYS = (
    'config', 'env', 'model', 'pack', 'memory', 'sequence',
    'throughput', 'results', 'status',
)


REQUIRED_CONFIG_FIELDS = frozenset({
    'mode', 'method', 'model_id', 'weight_quant', 'seed',
    'fp8_ratio', 'group_size', 'min_numel', 'mask_seed', 'dedupe',
    'pack_stochastic_rounding', 'sr_seed',
    'pack_4d_mode', 'body_encoding',
    'bf16_rmsnorm',
    'lora_r', 'lora_alpha', 'lora_dropout', 'lora_targets',
    'task',
    'n_train', 'epochs', 'lr', 'batch_size', 'grad_accum_steps',
    'max_seq_len', 'padding', 'grad_clip', 'scheduler', 'warmup_ratio', 'optimizer',
    'eval_samples', 'eval_fewshot', 'eval_max_new_tokens',
    'mem_seq_len', 'mem_batch_size', 'mem_steps', 'mem_warmup',
    'mem_abort_gb',
    'gc_enabled', 'checkpoint_every', 'output_dir', 'output_path',
})


_STATUS_CHOICES = ('OK', 'OOM', 'NAN', 'PARITY_FAIL', 'ERROR')
_OOM_STAGES = ('pre_flight', 'model_load', 'parity', 'forward', 'backward', 'optimizer', 'memory_sweep', 'eval')


# ============================================================
# INVARIANT-9: pre-GPU validation
# ============================================================

def validate_schema(config: dict, env: dict) -> None:
    """Fail fast if config/env are missing required fields.

    Called before any GPU work so a mis-specified run doesn't burn compute.
    """
    if not isinstance(config, dict):
        raise TypeError("validate_schema: config must be a dict (asdict(dataclass))")
    if not isinstance(env, dict):
        raise TypeError("validate_schema: env must be a dict (from collect_env)")

    missing_cfg = REQUIRED_CONFIG_FIELDS - set(config.keys())
    if missing_cfg:
        raise ValueError(
            f"validate_schema: config is missing required fields: {sorted(missing_cfg)}. "
            f"See configs/base.ExperimentConfig.")

    missing_env = REQUIRED_ENV_FIELDS - set(env.keys())
    if missing_env:
        raise ValueError(
            f"validate_schema: env is missing required fields: {sorted(missing_env)}. "
            f"Call oamp.env.collect_env() to build it.")

    if config['mode'] not in ('accuracy', 'memory'):
        raise ValueError(f"validate_schema: mode must be 'accuracy' or 'memory', got {config['mode']!r}")
    if config['method'] not in ('standard', 'naive_fp4', 'uniform_fp8', 'random_mixed', 'oamp'):
        raise ValueError(f"validate_schema: unknown method {config['method']!r}")
    if config['weight_quant'] not in ('bf16', 'nf4'):
        raise ValueError(f"validate_schema: weight_quant must be 'bf16' or 'nf4'")
    if config['method'] == 'random_mixed' and config.get('mask_seed') is None:
        raise ValueError("validate_schema: method='random_mixed' requires mask_seed to be set.")

    if not env['git_commit']:
        raise ValueError(
            "validate_schema: env.git_commit is empty. Run inside a git checkout "
            "so results can be tied to a SHA.")


# ============================================================
# Result skeleton
# ============================================================

def make_result_skeleton(config: dict, env: dict) -> dict:
    """Return a result dict with every required key present (values default to None/[])."""
    return {
        'config': dict(config),
        'env':    dict(env),
        'model': {
            'n_params':      None,
            'n_trainable':   None,
            'dtype_report':  None,
        },
        'pack': {
            'pack_stats':      None,
            'coverage_bytes':  {'unique': None, 'raw': None},
            'effective_bits':  None,
        },
        # INVARIANT-10: both allocated and reserved on unified memory hardware.
        'memory': {
            'peak_allocated_gb':      None,
            'peak_reserved_gb':       None,
            'peak_measurement':       None,
            'mem_after_load_gb':      None,
            'mem_after_optimizer_init_gb': None,
            'mem_after_first_step_gb': None,
        },
        'sequence': {
            'seq_len_mean':  None,
            'seq_len_max':   None,
            'seq_len_p95':   None,
            'tokens_total':  None,
        },
        'throughput': {
            'tokens_per_second': None,
            'seconds_per_step':  None,
            'train_wall_s':      None,
            'eval_wall_s':       None,
            'steps':             None,
        },
        'results': {
            'step_losses':      [],
            'final_loss':       None,
            # accuracy mode
            'gsm8k_accuracy':   None,
            'n_correct':        None,
            'n_eval_samples':   None,
            'per_sample':       [],
            'eval_diagnostics': None,
            'adapter_path':     None,
        },
        # INVARIANT §6.2: OOM is a structured result, not a crash.
        'status':      'PENDING',
        'oom_stage':   None,
        'oom_step':    None,
        'allocated_at_failure_gb': None,
        'error':       None,
    }


# ============================================================
# Status setters
# ============================================================

def mark_ok(result: dict) -> dict:
    result['status'] = 'OK'
    return result


def mark_oom(result: dict, *, stage: str, step: Optional[int] = None,
             allocated_gb: Optional[float] = None,
             error: Optional[str] = None) -> dict:
    if stage not in _OOM_STAGES:
        raise ValueError(f"mark_oom: stage must be in {_OOM_STAGES}, got {stage!r}")
    result['status'] = 'OOM'
    result['oom_stage'] = stage
    result['oom_step'] = step
    result['allocated_at_failure_gb'] = allocated_gb
    if error is not None:
        result['error'] = error
    return result


def mark_nan(result: dict, *, step: Optional[int] = None,
             error: Optional[str] = None) -> dict:
    result['status'] = 'NAN'
    result['oom_step'] = step
    if error is not None:
        result['error'] = error
    return result


# ============================================================
# INVARIANT-11: checkpoint sidecar (JSONL, append-only)
# ============================================================

def checkpoint_path_for(output_json: str) -> str:
    """Sidecar path: ``<output>.json`` -> ``<output>.checkpoints.jsonl``."""
    base, _ = os.path.splitext(output_json)
    return f'{base}.checkpoints.jsonl'


def append_checkpoint(path: str, *, step: int, loss: float,
                      elapsed_s: float, peak_alloc_gb: float,
                      extra: Optional[dict] = None) -> None:
    """Append one line to the checkpoint JSONL. Safe to call from partial runs."""
    rec = {
        'step':          int(step),
        'loss':          float(loss),
        'elapsed_s':     float(elapsed_s),
        'peak_alloc_gb': float(peak_alloc_gb),
        'wall_time':     time.time(),
    }
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


# ============================================================
# Final write
# ============================================================

def write_result(path: str, result: dict) -> None:
    """Write the final result JSON. Fails if ``result['status']`` is still 'PENDING'."""
    if result.get('status') == 'PENDING':
        raise RuntimeError(
            "write_result: status is still 'PENDING'. Call mark_ok / mark_oom / mark_nan first.")
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(o: Any):
    # torch dtype-like objects, sets, etc.
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
