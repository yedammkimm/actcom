"""Verify oamp.data + oamp.evaluate (spec v1 §5 extension).

Gates:
  D1  load_task('gsm8k', 'test', n_samples=8, seed=42) returns 8 well-formed samples.
  D2  format_eval_prompt('gsm8k') has 8 fewshot Q/A blocks and no trailing content.
  D3  extract_answer('gsm8k', 'The answer is 42.') == '42'.
  D4  is_correct('gsm8k', '42', '42.0') is True; ('42', '43') is False.
  D5  Unknown task -> ValueError.
  D6  seq_len_stats([]) returns zero-valued dict with all 4 keys.
  D7  seq_len_stats on a fixed list matches the hand-computed mean/max/p95/tot.
  D8  Two consecutive load_task(seed=42) calls return the same order (deterministic).
  D9  Config field 'task' present in REQUIRED_CONFIG_FIELDS.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from oamp.data import (
    load_task, format_eval_prompt, format_train_prompt,
    extract_answer, is_correct, default_fewshot, seq_len_stats,
)
from oamp.schema import REQUIRED_CONFIG_FIELDS
from configs import ExperimentConfig, MODEL_ALIASES

CACHE_DIR = os.environ.get('HF_HOME', '/app/hf_cache')


def d1_load_test():
    print("[D1] load_task('gsm8k', 'test', n=8, seed=42)")
    data = load_task('gsm8k', 'test', n_samples=8, seed=42, cache_dir=CACHE_DIR)
    assert len(data) == 8, f"D1 FAIL: expected 8, got {len(data)}"
    for s in data:
        assert set(s) == {'question', 'answer_text', 'answer_number'}, s.keys()
        assert isinstance(s['question'], str) and s['question']
        assert isinstance(s['answer_number'], str) and s['answer_number']
    print(f"  OK  first_q={data[0]['question'][:60]!r}  gold={data[0]['answer_number']!r}")
    return data


def d2_format_eval(sample):
    print("\n[D2] format_eval_prompt('gsm8k', sample)")
    prompt = format_eval_prompt('gsm8k', sample)
    n_shots = prompt.count("\nA:") - 1   # last one is the target
    assert n_shots == 8, f"D2 FAIL: {n_shots} fewshot blocks, expected 8"
    assert prompt.rstrip().endswith("A:"), "D2 FAIL: prompt does not end with 'A:'"
    print(f"  OK  n_fewshot={n_shots}  ends_with='A:'  len={len(prompt)}")


def d3_extract():
    print("\n[D3] extract_answer('gsm8k', ...)")
    assert extract_answer('gsm8k', 'The answer is 42.') == '42'
    assert extract_answer('gsm8k', 'the answer is 3.14') == '3.14'
    assert extract_answer('gsm8k', '#### 100') == '100'
    assert extract_answer('gsm8k', 'blah 7') == '7'
    assert extract_answer('gsm8k', '') == ''
    print("  OK  ('The answer is 42.', '#### 100', 'blah 7', '') covered")


def d4_score():
    print("\n[D4] is_correct('gsm8k', ...)")
    assert is_correct('gsm8k', '42', '42.0') is True
    assert is_correct('gsm8k', '42', '43') is False
    assert is_correct('gsm8k', '3.14', '3.140000') is True
    assert is_correct('gsm8k', '', '42') is False
    print("  OK  numeric tolerance + empty pred handled")


def d5_unknown_task():
    print("\n[D5] load_task('math', ...) -> ValueError")
    try:
        load_task('math', 'test', n_samples=1, seed=0, cache_dir=CACHE_DIR)
    except ValueError as e:
        print(f"  OK  {str(e)[:80]}...")
    else:
        raise AssertionError("D5 FAIL: no ValueError")


def d6_d7_seq_len_stats():
    print("\n[D6] seq_len_stats([]) -> zero dict")
    z = seq_len_stats([])
    assert set(z) == {'seq_len_mean', 'seq_len_max', 'seq_len_p95', 'tokens_total'}
    assert z == {'seq_len_mean': 0.0, 'seq_len_max': 0, 'seq_len_p95': 0, 'tokens_total': 0}
    print(f"  OK  {z}")

    print("\n[D7] seq_len_stats([1..20]) hand-checked")
    stats = seq_len_stats(list(range(1, 21)))
    # mean = 10.5, max = 20, p95 → index round(0.95*19)=18 → value 19, total = 210
    assert stats['seq_len_mean'] == 10.5, stats
    assert stats['seq_len_max'] == 20, stats
    assert stats['seq_len_p95'] == 19, stats
    assert stats['tokens_total'] == 210, stats
    print(f"  OK  {stats}")


def d8_deterministic():
    print("\n[D8] two load_task(seed=42) calls return the same order")
    a = load_task('gsm8k', 'test', n_samples=16, seed=42, cache_dir=CACHE_DIR)
    b = load_task('gsm8k', 'test', n_samples=16, seed=42, cache_dir=CACHE_DIR)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x['question'] == y['question'], f"D8 FAIL at {i}: shuffle mismatch"
    print(f"  OK  first 16 identical, first_q_hash={hash(a[0]['question']) & 0xffff}")

    # Different seed -> different order
    c = load_task('gsm8k', 'test', n_samples=16, seed=123, cache_dir=CACHE_DIR)
    diffs = sum(x['question'] != z['question'] for x, z in zip(a, c))
    assert diffs > 0, "D8 FAIL: seed=42 and seed=123 returned identical order"
    print(f"  OK  seed=42 vs seed=123 differs at {diffs}/16 positions")


def d9_config_field():
    print("\n[D9] 'task' is in REQUIRED_CONFIG_FIELDS + present on default cfg")
    assert 'task' in REQUIRED_CONFIG_FIELDS, "D9 FAIL: 'task' missing from schema"
    cfg = ExperimentConfig(
        mode='accuracy', method='oamp',
        model_id=MODEL_ALIASES['3B'], weight_quant='nf4', seed=42)
    assert cfg.task == 'gsm8k'
    assert 'task' in cfg.asdict()
    print(f"  OK  cfg.task={cfg.task!r}")


if __name__ == '__main__':
    print("=" * 70)
    print("data + evaluate verification (spec v1 §5 extension)")
    print("=" * 70)
    sample_data = d1_load_test()
    d2_format_eval(sample_data[0])
    d3_extract()
    d4_score()
    d5_unknown_task()
    d6_d7_seq_len_stats()
    d8_deterministic()
    d9_config_field()
    print("\nALL CHECKS PASSED")
