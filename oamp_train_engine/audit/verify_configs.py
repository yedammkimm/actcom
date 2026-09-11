"""Verify configs.base ExperimentConfig (spec v1 §5).

Gates:
  C1  asdict(cfg) contains every REQUIRED_CONFIG_FIELDS entry.
  C2  method='random_mixed' + mask_seed=None -> ValueError.
  C3  mode='memory' + mem_seq_lens=[] -> ValueError.
  C4  Paper-default cfg has total_train_steps == 3736.
  C5  method='naive_fp4' + fp8_ratio=0.2 -> ValueError (boundary invariant).
  C6  method='uniform_fp8' auto-fp8_ratio via build_from_cli().
  C7  build_from_cli(['--model', '3B', ...]) resolves alias.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from configs import ExperimentConfig, MODEL_ALIASES, build_from_cli, total_train_steps
from oamp.schema import REQUIRED_CONFIG_FIELDS


def check_c1_c4():
    print("[C1/C4] paper default cfg → asdict covers REQUIRED_CONFIG_FIELDS, steps=3736")
    cfg = ExperimentConfig(
        mode='accuracy', method='oamp',
        model_id=MODEL_ALIASES['3B'], weight_quant='nf4', seed=42)
    d = cfg.asdict()
    missing = REQUIRED_CONFIG_FIELDS - set(d)
    assert not missing, f"C1 FAIL: missing {sorted(missing)}"
    print(f"  keys: {len(d)}  missing_required: 0")

    exp = total_train_steps(n_train=7473, batch_size=1, grad_accum_steps=4, epochs=2)
    assert cfg.total_train_steps == 3736 == exp, (
        f"C4 FAIL: total_train_steps={cfg.total_train_steps}, expected 3736")
    print(f"  total_train_steps: {cfg.total_train_steps}  routing={cfg.routing}")
    print("  C1/C4 PASS")


def check_c2():
    print("\n[C2] method='random_mixed' without mask_seed -> ValueError")
    try:
        ExperimentConfig(mode='accuracy', method='random_mixed',
                         model_id=MODEL_ALIASES['3B'], weight_quant='nf4', seed=42)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("C2 FAIL: did not raise")


def check_c3():
    print("\n[C3] mode='memory' with mem_seq_len=0 -> ValueError")
    try:
        ExperimentConfig(mode='memory', method='oamp',
                         model_id=MODEL_ALIASES['3B'], weight_quant='nf4', seed=42,
                         mem_seq_len=0)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("C3 FAIL: did not raise")


def check_c5():
    print("\n[C5] method='naive_fp4' with fp8_ratio=0.2 -> ValueError (§2.3 boundary)")
    try:
        ExperimentConfig(mode='accuracy', method='naive_fp4',
                         model_id=MODEL_ALIASES['3B'], weight_quant='nf4', seed=42,
                         fp8_ratio=0.2)
    except ValueError as e:
        print(f"  OK: {str(e)[:80]}...")
    else:
        raise AssertionError("C5 FAIL: did not raise")


def check_c6():
    print("\n[C6] build_from_cli method='uniform_fp8' auto-sets fp8_ratio=1.0")
    cfg = build_from_cli(['--mode', 'memory', '--method', 'uniform_fp8'])
    assert cfg.fp8_ratio == 1.0, f"C6 FAIL: fp8_ratio={cfg.fp8_ratio}"
    print(f"  fp8_ratio={cfg.fp8_ratio}   routing={cfg.routing}")


def check_c7():
    print("\n[C7] build_from_cli --model 3B resolves alias")
    cfg = build_from_cli(['--mode', 'accuracy', '--method', 'oamp', '--model', '3B'])
    assert cfg.model_id == MODEL_ALIASES['3B'], f"C7 FAIL: model_id={cfg.model_id}"
    print(f"  model_id={cfg.model_id}")

    print("\n[C7b] build_from_cli --model 8B, override --fp8_ratio 0.30")
    cfg = build_from_cli(['--mode', 'accuracy', '--method', 'oamp',
                          '--model', '8B', '--fp8_ratio', '0.30', '--lr', '5e-5'])
    assert cfg.model_id == MODEL_ALIASES['8B']
    assert cfg.fp8_ratio == 0.30
    assert cfg.lr == 5e-5
    print(f"  model_id={cfg.model_id}  fp8_ratio={cfg.fp8_ratio}  lr={cfg.lr}")


if __name__ == '__main__':
    print("=" * 70)
    print("configs.base verification (spec v1 §5)")
    print("=" * 70)
    check_c1_c4()
    check_c2()
    check_c3()
    check_c5()
    check_c6()
    check_c7()
    print("\nALL CHECKS PASSED")
