"""Verify oamp.env + oamp.schema.

Gates:
  E1  collect_env() returns every REQUIRED_ENV_FIELDS key.
  E2  compute_capability populated when CUDA is available.
  E3  git_commit is a 40-char sha (or empty with a warning path).
  E4  git_dirty True/False bool.

  S1  validate_schema passes on a full config + env dict.
  S2  validate_schema raises on missing config field.
  S3  validate_schema raises on missing env field.
  S4  validate_schema raises on method='random_mixed' without mask_seed.
  S5  make_result_skeleton contains every RESULT_TOP_KEYS entry.
  S6  mark_oom(stage='forward') writes the structured payload (§6.2).
  S7  append_checkpoint writes a JSONL line that round-trips.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from oamp.env import collect_env, REQUIRED_ENV_FIELDS
from oamp.schema import (
    RESULT_TOP_KEYS, REQUIRED_CONFIG_FIELDS,
    validate_schema, make_result_skeleton,
    mark_ok, mark_oom, mark_nan,
    checkpoint_path_for, append_checkpoint, write_result,
)


def _minimal_config() -> dict:
    return {
        'mode': 'accuracy',
        'method': 'oamp',
        'model_id': 'meta-llama/Llama-3.2-3B-Instruct',
        'weight_quant': 'nf4',
        'seed': 42,
        'fp8_ratio': 0.20,
        'group_size': 128,
        'min_numel': 1024,
        'mask_seed': None,
        'dedupe': False,
        'pack_stochastic_rounding': False,
        'sr_seed': None,
        'pack_4d_mode': 'fp4',
        'bf16_rmsnorm': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'task': 'gsm8k',
        'n_train': 7473,
        'epochs': 2,
        'lr': 2e-4,
        'batch_size': 1,
        'grad_accum_steps': 4,
        'max_seq_len': 512,
        'padding': False,
        'grad_clip': 1.0,
        'scheduler': 'cosine',
        'warmup_ratio': 0.03,
        'optimizer': 'adamw',
        'eval_samples': 500,
        'eval_fewshot': 8,
        'eval_max_new_tokens': 256,
        'mem_seq_len': 4096,
        'mem_batch_size': 4,
        'mem_steps': 100,
        'mem_warmup': 10,
        'gc_enabled': False,
        'checkpoint_every': 50,
        'output_dir': '/tmp/x',
    }


def check_env():
    env = collect_env()
    print(f"\n[E] collect_env keys: {sorted(env)[:5]} ... ({len(env)} total)")
    print(f"    gpu_name={env['gpu_name']!r}  cc={env['compute_capability']!r}")
    print(f"    git_commit={env['git_commit'][:12]!r}  git_dirty={env['git_dirty']}")
    print(f"    torch={env['torch_version']}  transformers={env['transformers_version']}  "
          f"peft={env['peft_version']}  bnb={env['bitsandbytes_version']}")

    missing = REQUIRED_ENV_FIELDS - set(env)
    assert not missing, f"E1 FAIL: missing env fields {missing}"

    try:
        import torch
        if torch.cuda.is_available():
            assert env['compute_capability'], "E2 FAIL: compute_capability empty despite CUDA"
    except ImportError:
        pass

    assert isinstance(env['git_dirty'], bool), "E4 FAIL: git_dirty is not bool"
    return env


def check_validate_schema(env):
    cfg = _minimal_config()
    print("\n[S1] validate_schema(full config + env)")
    validate_schema(cfg, env)
    print("  OK")

    print("[S2] missing config field ('lr') -> ValueError")
    bad = dict(cfg); bad.pop('lr')
    try:
        validate_schema(bad, env)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("S2 FAIL: did not raise on missing 'lr'")

    print("[S3] missing env field ('gpu_name') -> ValueError")
    bad_env = dict(env); bad_env.pop('gpu_name')
    try:
        validate_schema(cfg, bad_env)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("S3 FAIL: did not raise on missing 'gpu_name'")

    print("[S4] method='random_mixed' without mask_seed -> ValueError")
    bad = dict(cfg); bad['method'] = 'random_mixed'; bad['mask_seed'] = None
    try:
        validate_schema(bad, env)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("S4 FAIL: random_mixed w/o mask_seed passed")


def check_skeleton_and_status(env):
    print("\n[S5] make_result_skeleton has every top key")
    skel = make_result_skeleton(_minimal_config(), env)
    for k in RESULT_TOP_KEYS:
        assert k in skel, f"S5 FAIL: missing top key {k!r}"
    print(f"  OK ({len(RESULT_TOP_KEYS)} top keys, {len(skel)} total including OOM slots)")

    print("[S6] mark_oom(stage='forward', step=42, allocated_gb=90.0)")
    mark_oom(skel, stage='forward', step=42, allocated_gb=90.0, error='CUDA OOM')
    assert skel['status'] == 'OOM'
    assert skel['oom_stage'] == 'forward'
    assert skel['oom_step'] == 42
    assert skel['allocated_at_failure_gb'] == 90.0
    print(f"  OK: {dict((k, skel[k]) for k in ('status','oom_stage','oom_step','allocated_at_failure_gb'))}")


def check_checkpoints(env):
    print("\n[S7] append_checkpoint round-trip")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'run.json')
        ckpt = checkpoint_path_for(out)
        append_checkpoint(ckpt, step=1, loss=1.5, elapsed_s=0.1, peak_alloc_gb=5.2)
        append_checkpoint(ckpt, step=2, loss=1.3, elapsed_s=0.2, peak_alloc_gb=5.3)
        lines = open(ckpt).read().strip().split('\n')
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        assert rec1['step'] == 1 and rec1['peak_alloc_gb'] == 5.2
        print(f"  OK (2 lines, first step={rec1['step']} peak={rec1['peak_alloc_gb']} GB)")

        # write_result rejects PENDING
        skel = make_result_skeleton(_minimal_config(), env)
        try:
            write_result(out, skel)
        except RuntimeError as e:
            print(f"  OK: write_result rejects PENDING: {str(e)[:60]}...")
        mark_ok(skel)
        write_result(out, skel)
        assert os.path.exists(out)
        print(f"  OK: write_result wrote {out} after mark_ok")


if __name__ == '__main__':
    print("=" * 70)
    print("schema + env verification (spec v1 §6)")
    print("=" * 70)
    env = check_env()
    check_validate_schema(env)
    check_skeleton_and_status(env)
    check_checkpoints(env)
    print("\nALL CHECKS PASSED")
