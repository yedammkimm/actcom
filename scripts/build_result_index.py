"""Scan results/**/*.json and emit `_INDEX.md` + `_INDEX.csv`.

Groups accuracy runs by condition tuple:
    (method, pack_4d_mode, bf16_rmsnorm, warmup_ratio, weight_quant,
     stochastic_rounding)
so that seed variance can be reported per condition rather than per run.

Handles pre-batching JSONs gracefully:
  - `pack_4d_mode` absent  -> defaults to 'fp4' (legacy γ behaviour)
  - `adapter_path` absent  -> flagged as "no adapter" (cannot re-evaluate)
  - `pack_stochastic_rounding` absent -> False

Emits two tables in `_INDEX.md`:
  1. Condition summary (per-group mean / std / seeds / adapter fraction)
  2. Full run list (path, method, seed, pack_4d, rms_bf16, wu, acc, peak, git)

Also writes `_INDEX.csv` with every field for downstream spreadsheet use.

Usage: python scripts/build_result_index.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime

RESULTS_ROOT = '/home/yedam/HMA/HMA_Project/results'


def _get(d, path, default=None):
    """Nested dict lookup by dotted path."""
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _mean_last50(losses):
    if not losses:
        return None
    return statistics.mean(losses[-50:] if len(losses) >= 50 else losses)


def _mean_last_n_steps(path, n_steps=500):
    """Mean loss over the final ``n_steps`` optimizer steps, from the sidecar.

    A step-based window rather than a fixed number of checkpoint records: the
    checkpoint interval differs between cohorts (50 steps for the earliest
    3B seeds, 10 for the rest), so "the last 50 records" spans a different
    interval per seed and the values are not comparable across runs.
    """
    # Prefer the complete per-step losses stored in the run JSON; every one of
    # the final n_steps is then averaged. Fall back to the checkpoint sidecar,
    # which only samples every 10 or 50 steps, when the JSON has no trace.
    try:
        with open(path) as f:
            sl = json.load(f).get('results', {}).get('step_losses')
        if sl and len(sl) >= n_steps:
            return sum(sl[-n_steps:]) / n_steps
    except Exception:
        pass
    side = path[:-len('.json')] + '.checkpoints.jsonl'
    if not os.path.exists(side):
        return None
    rows = []
    with open(side) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('loss') is not None and r.get('step') is not None:
                rows.append(r)
    if not rows:
        return None
    last = max(r['step'] for r in rows)
    win = [r['loss'] for r in rows if r['step'] > last - n_steps]
    return statistics.mean(win) if win else None


def _accuracy(d):
    """Percent correct from counts, never from the stored accuracy field.

    ``results.gsm8k_accuracy`` is null in the re-evaluation JSONs and, in the
    oldest runs, is stored already as a percent while newer ones store a
    fraction. The counts are unambiguous in every file that has them.
    """
    nc = _get(d, 'results.n_correct')
    ne = _get(d, 'results.n_eval_samples')
    if ne in (None, 0):
        ps = _get(d, 'results.per_sample')
        ne = len(ps) if ps else None
    if nc is None or not ne:
        return None
    return 100.0 * nc / ne


def _protocol(cfg, path):
    """Which of the three accuracy protocols a run's number came from."""
    if 'reeval' in os.path.basename(path) or 'reeval' in path:
        return 'c reeval n200 mnt512 stop'
    n = cfg.get('eval_samples')
    mnt = cfg.get('eval_max_new_tokens')
    if n is None:
        return '-'
    return f"{'a' if n == 500 else 'b'} in-train n{n} mnt{mnt}"


def _mean_last_all(losses):
    return statistics.mean(losses) if losses else None


def _fmt(x, spec=''):
    if x is None:
        return '-'
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def collect_rows():
    rows = []
    pattern = os.path.join(RESULTS_ROOT, '**', '*.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        base = os.path.basename(path)
        if 'checkpoint' in base or base.startswith('_INDEX'):
            continue
        d = _load(path)
        if not isinstance(d, dict) or 'config' not in d or 'status' not in d:
            continue                    # not a spec v1 run JSON
        cfg = d.get('config', {})
        if not cfg.get('mode'):
            continue

        losses = _get(d, 'results.step_losses', []) or []
        rows.append({
            'path':               os.path.relpath(path, RESULTS_ROOT),
            'timestamp':          _get(d, 'env.timestamp'),
            'mode':               cfg.get('mode'),
            'method':             cfg.get('method'),
            'seed':               cfg.get('seed'),
            'weight_quant':       cfg.get('weight_quant'),
            'pack_4d_mode':       cfg.get('pack_4d_mode', 'fp4'),
            'body_encoding':      cfg.get('body_encoding'),
            'nonfinite_skip':     cfg.get('nonfinite_skip'),
            'n_nonfinite':        _get(d, 'results.n_nonfinite_grad_steps'),
            'bf16_rmsnorm':       cfg.get('bf16_rmsnorm'),
            'warmup_ratio':       cfg.get('warmup_ratio'),
            'lr':                 cfg.get('lr'),
            'optimizer':          cfg.get('optimizer'),
            'scheduler':          cfg.get('scheduler'),
            'n_train':            cfg.get('n_train'),
            'epochs':             cfg.get('epochs'),
            'mask_seed':          cfg.get('mask_seed'),
            'sr':                 cfg.get('pack_stochastic_rounding', False),
            'sr_seed':            cfg.get('sr_seed'),
            'accuracy':           _accuracy(d),
            'accuracy_field':     _get(d, 'results.gsm8k_accuracy'),
            'protocol':           _protocol(cfg, path),
            'n_correct':          _get(d, 'results.n_correct'),
            'n_eval':             _get(d, 'results.n_eval_samples'),
            'final_loss':         _get(d, 'results.final_loss'),
            'mean_last50':        _mean_last50(losses),
            'mean_last500steps':  _mean_last_n_steps(path, 500),
            'mean_all':           _mean_last_all(losses),
            'peak_alloc_gb':      _get(d, 'memory.peak_allocated_gb'),
            'peak_reserved_gb':   _get(d, 'memory.peak_reserved_gb'),
            'mem_batch':          cfg.get('mem_batch_size'),
            'mem_seq_len':        cfg.get('mem_seq_len'),
            'mem_steps':          cfg.get('mem_steps'),
            'mem_warmup':         cfg.get('mem_warmup'),
            'gc_enabled':         cfg.get('gc_enabled'),
            'seconds_per_step':   _get(d, 'throughput.seconds_per_step'),
            'adapter':            _get(d, 'results.adapter_path') is not None,
            'per_sample':         bool(_get(d, 'results.per_sample')),
            'has_diagnostics':    _get(d, 'results.eval_diagnostics') is not None,
            'n_truncated':        _get(d, 'results.eval_diagnostics.n_truncated'),
            'status':             d.get('status'),
            'git_commit':         (_get(d, 'env.git_commit') or '')[:12],
            'git_dirty':          _get(d, 'env.git_dirty'),
        })
    return rows


def collect_ppl(rows):
    """Perplexity measurements, with the arm resolved from the training run.

    A perplexity JSON carries a `training_config` copied from the run that made
    the adapter, and the arm is (method, body_encoding, pack_4d_mode). Reading
    only the last two mixes arms: an uncompressed run records a body encoding
    and a rank-4 mode as well, both inert for it, so a `standard` run sitting in
    the same directory as a compressed one is indistinguishable without
    `method`. That is not hypothetical -- results/ppl_eval holds arm B and
    C_STD at seed 42, and dropping `method` puts the uncompressed 16.0717 into
    the compressed arm and moves its variance ratio from 33.66 to 37.03.

    `adapter_dir` is emitted alongside so a measurement can be traced back to
    the training run that produced it.
    """
    LAB = {('naive_fp4', 'e2m1', 'fp4'): 'A  e2m1 + blockwise',
           ('naive_fp4', 'e2m1', 'fp8'): 'B  e2m1 + FP8',
           ('standard', None, None):     'C  uncompressed',
           ('naive_fp4', 'int4', 'fp4'): 'D  INT4 + blockwise',
           ('naive_fp4', 'int4', 'fp8'): 'E  INT4 + FP8',
           ('naive_fp4', 'e2m1', 'chan_int4'): 'F  e2m1 + per-channel'}
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, 'ppl_*', '*.json'))):
        d = _load(path)
        if not isinstance(d, dict) or 'wikitext2_ppl' not in d:
            continue
        tc = d.get('training_config') or {}
        key = (tc.get('method'), tc.get('body_encoding'), tc.get('pack_4d_mode'))
        if key not in LAB and tc.get('method') == 'standard':
            key = ('standard', None, None)
        ap = d.get('adapter_path', '')
        out.append({
            'arm':      LAB.get(key, f"? {key}"),
            'seed':     d.get('seed'),
            'wikitext': d.get('wikitext2_ppl'),
            'gsm8k':    d.get('gsm8k_test_ppl'),
            'narr':     d.get('narrativeqa_ppl'),
            'gov':      d.get('govreport_ppl'),
            'n_train':  tc.get('n_train'),
            'epochs':   tc.get('epochs'),
            'adapter_dir': ap.split('/')[1] if '/' in ap else '',
            'dir':      os.path.basename(os.path.dirname(path)),
            'path':     os.path.relpath(path, RESULTS_ROOT),
        })
    return out


def collect_reeval():
    """The unified-protocol re-evaluations, which carry a different schema.

    These files have no ``config`` block, so the main scan skips them, yet they
    are the only accuracy numbers comparable across seeds: every arm is scored
    on the same 200 problems with stop sequences and a 512-token budget
    (protocol (c)). Any group mean or test over seeds must come from here.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, '**', 'reeval__*.json'),
                                 recursive=True)):
        d = _load(path)
        if not isinstance(d, dict) or 'results' not in d:
            continue
        r = d['results']
        n = r.get('n_samples') or d.get('n_samples')
        nc = r.get('n_correct')
        ps = r.get('per_sample') or []
        parts = os.path.basename(path).split('__')
        out.append({
            'arm':      parts[1] if len(parts) > 1 else '?',
            'seed':     d.get('seed'),
            'n':        n,
            'n_correct': nc,
            'acc':      (100.0 * nc / n) if (nc is not None and n) else None,
            'acc_field': r.get('accuracy_pct'),
            'mnt':      d.get('max_new_tokens'),
            'stop':     bool(d.get('stop_strings')),
            'capped':   sum(1 for x in ps if x.get('n_gen_tokens', 0) >= (d.get('max_new_tokens') or 10**9)),
            'path':     os.path.relpath(path, RESULTS_ROOT),
        })
    return out


def write_csv(rows, out_path):
    if not rows:
        return
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def group_key(r):
    """Condition tuple for accuracy-run grouping.

    ``pack_4d_mode`` is a no-op for method='standard' (nullcontext, no pack
    hooks fire), so we collapse it to ``'n/a'`` in that case — otherwise the
    Standard seeds get split across fp4/fp8 groups even though they are
    bit-exact identical.
    """
    p4d = 'n/a' if r['method'] == 'standard' else r['pack_4d_mode']
    body = 'n/a' if r['method'] == 'standard' else (r['body_encoding'] or 'int4(pre-field)')
    return (r['method'], body, p4d, r['bf16_rmsnorm'],
            r['warmup_ratio'], r['weight_quant'], r['sr'], r['lr'],
            r['n_train'], r['epochs'], r['protocol'])


def write_markdown(rows, out_path):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '# Results Index',
        '',
        f'Auto-generated by `scripts/build_result_index.py` on {now}.',
        f'Sources: `{RESULTS_ROOT}/**/*.json` (spec v1 runs only).',
        '',
        '**Conditions that matter**: `pack_4d_mode`, `bf16_rmsnorm`, '
        '`warmup_ratio`, `weight_quant`, `stochastic_rounding`, `lr`, '
        '`n_train`, `epochs`. Runs are grouped by the full tuple so pilot '
        '(`pack_4d_mode=fp4`) and post-fix (`pack_4d_mode=fp8`) results are '
        'never conflated.',
        '',
        '## Accuracy runs — grouped by condition',
        '',
        '| method | body | pack4d | rms | wu | wq | SR | lr | n_train | ep | '
        'protocol | n | mean%  | std%  | range% | seeds | adapters | nnf |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    acc_rows = [r for r in rows if r['mode'] == 'accuracy'
                and r['accuracy'] is not None]
    groups = defaultdict(list)
    for r in acc_rows:
        groups[group_key(r)].append(r)

    for key, grp in sorted(groups.items()):
        method, body, p4d, rms, wu, wq, sr, lr, n_train, ep, proto = key
        accs = [r['accuracy'] for r in grp]
        n = len(accs)
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if n > 1 else 0.0
        rng = (max(accs) - min(accs)) if n > 1 else 0.0
        seeds = sorted({r['seed'] for r in grp})
        seeds_str = ','.join(str(x) for x in seeds)
        n_ad = sum(1 for r in grp if r['adapter'])
        ad_str = f'{n_ad}/{n}'
        nnfs = sorted({r['n_nonfinite'] for r in grp if r['n_nonfinite'] is not None})
        nnf_str = ','.join(str(x) for x in nnfs) if nnfs else '-'
        lines.append(
            f'| {method} | {body} | {p4d} | {rms} | {_fmt(wu)} | {wq} | {sr} | '
            f'{_fmt(lr)} | {n_train} | {ep} | {proto} | '
            f'{n} | {m:.2f} | {s:.2f} | {rng:.2f} | {seeds_str} | {ad_str} | {nnf_str} |'
        )

    # ------------------------------------------------------------
    ppl = collect_ppl(rows)
    if ppl:
        lines += [
            '',
            '## Perplexity measurements, by arm',
            '',
            'Arm is resolved from `training_config` as (method, body_encoding, '
            'pack_4d_mode). All three are needed: an uncompressed run records a '
            'body encoding and a rank-4 mode too, both inert for it, so omitting '
            '`method` merges C_STD into arm B wherever they share a directory '
            '(`ppl_eval` at seed 42).',
            '',
            '| arm | seed | n_train | ep | wikitext | gsm8k | narr | gov | adapter dir | path |',
            '|---|---|---|---|---|---|---|---|---|---|',
        ]
        for r in sorted(ppl, key=lambda x: (x['arm'], x['n_train'] or 0,
                                            x['seed'] if x['seed'] is not None else -1)):
            lines.append(
                f"| {r['arm']} | {r['seed']} | {r['n_train']} | {r['epochs']} | "
                f"{_fmt(r['wikitext'], '.4f')} | {_fmt(r['gsm8k'], '.4f')} | "
                f"{_fmt(r['narr'], '.4f')} | {_fmt(r['gov'], '.4f')} | "
                f"{r['adapter_dir']} | `{r['path']}` |")
        grp = defaultdict(list)
        for r in ppl:
            if r['wikitext'] is not None and r['n_train'] == 7473 and r['epochs'] == 2:
                grp[r['arm']].append((r['seed'], r['wikitext'], r['dir']))
        lines += ['', 'Full-length runs (7473 x 2 epochs), WikiText-2:', '',
                  '| arm | runs | distinct seeds | mean | sd | note |',
                  '|---|---|---|---|---|---|']
        for arm in sorted(grp):
            v = grp[arm]
            w = [x[1] for x in v]
            seeds = sorted({x[0] for x in v})
            dup = '' if len(seeds) == len(v) else \
                  f"seed {[s for s in seeds if sum(1 for x in v if x[0] == s) > 1]} measured twice"
            sd = statistics.stdev(w) if len(w) > 1 else 0.0
            lines.append(f'| {arm} | {len(v)} | {len(seeds)} | '
                         f'{statistics.mean(w):.4f} | {sd:.4f} | {dup} |')

    rev = collect_reeval()
    if rev:
        lines += [
            '',
            '## Unified-protocol re-evaluations (protocol (c))',
            '',
            'Every accuracy statistic taken over seeds must come from this table. '
            'The in-training numbers above were produced under two different '
            'protocols and are not comparable across cohorts.',
            '',
            '| arm | seed | n | correct | acc% | mnt | stop | capped | path |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for r in sorted(rev, key=lambda x: (x['arm'], x['seed'] if x['seed'] is not None else -1)):
            lines.append(
                f"| {r['arm']} | {r['seed']} | {r['n']} | {r['n_correct']} | "
                f"{_fmt(r['acc'], '.1f')} | {r['mnt']} | {r['stop']} | "
                f"{r['capped']} | `{r['path']}` |")
        # Group by (arm, n_samples): a re-evaluation at 500 problems is a
        # different protocol from one at 200 and must not enter the same mean.
        by_arm = defaultdict(list)
        for r in rev:
            if r['acc'] is not None:
                by_arm[(r['arm'], r['n'])].append(r['acc'])
        lines += ['', '| arm | eval n | runs | mean% | sd% | range% |',
                  '|---|---|---|---|---|---|']
        for (arm, nn) in sorted(by_arm):
            a = by_arm[(arm, nn)]
            sd = statistics.stdev(a) if len(a) > 1 else 0.0
            lines.append(f'| {arm} | {nn} | {len(a)} | {statistics.mean(a):.2f} | '
                         f'{sd:.2f} | {(max(a)-min(a)) if len(a) > 1 else 0.0:.2f} |')

    # ------------------------------------------------------------
    lines += [
        '',
        '## Memory runs',
        '',
        '| method | body | pack4d | gc | wq | B×L | steps | peak_alloc | '
        'peak_reserved | s/step | git | path |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    mem_rows = [r for r in rows if r['mode'] == 'memory']
    for r in sorted(mem_rows, key=lambda r: (r['method'], r['pack_4d_mode'])):
        bxl = f"{r['mem_batch']}×{r['mem_seq_len']}"
        lines.append(
            f"| {r['method']} | {r['body_encoding']} | {r['pack_4d_mode']} | "
            f"{r['gc_enabled']} | {r['weight_quant']} | {bxl} | "
            f"{r['mem_steps']}+{r['mem_warmup']} | "
            f"{_fmt(r['peak_alloc_gb'], '.2f')} | "
            f"{_fmt(r['peak_reserved_gb'], '.2f')} | "
            f"{_fmt(r['seconds_per_step'], '.3f')} | {r['git_commit']} | "
            f"`{r['path']}` |"
        )

    # ------------------------------------------------------------
    lines += [
        '',
        '## Individual accuracy runs (chronological)',
        '',
        '| method | seed | p4d | rms | wu | SR | acc% | trunc | adapter | git | ts | path |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in sorted(acc_rows, key=lambda r: r['timestamp'] or ''):
        ad = '✓' if r['adapter'] else '✗'
        tr = r['n_truncated'] if r['n_truncated'] is not None else '-'
        lines.append(
            f"| {r['method']} | {r['seed']} | {r['pack_4d_mode']} | "
            f"{r['bf16_rmsnorm']} | {_fmt(r['warmup_ratio'])} | {r['sr']} | "
            f"{_fmt(r['accuracy'], '.2f')} | {tr} | {ad} | "
            f"{r['git_commit']} | {r['timestamp'] or '-'} | `{r['path']}` |"
        )

    # ------------------------------------------------------------
    n_runs = len(rows)
    n_acc = len([r for r in rows if r['mode'] == 'accuracy'])
    n_mem = len([r for r in rows if r['mode'] == 'memory'])
    n_with_adapter = sum(1 for r in rows
                         if r['mode'] == 'accuracy' and r['adapter'])
    n_with_diag = sum(1 for r in rows
                      if r['mode'] == 'accuracy' and r['has_diagnostics'])
    lines += [
        '',
        '## Summary counts',
        '',
        f'- Total spec v1 JSONs: **{n_runs}** (accuracy: {n_acc}, memory: {n_mem})',
        f'- Accuracy runs **with adapter saved** (re-evaluatable): '
        f'{n_with_adapter}/{n_acc}',
        f'- Accuracy runs **with eval_diagnostics** (trunc counts, extraction sources): '
        f'{n_with_diag}/{n_acc}',
        '',
        '## Legend',
        '',
        '- `pack_4d_mode=fp4`: attention head-view tensors compressed at FP4. '
        'Pre-2026-08-16 default. Corrupts backward gradients (cos ≈ 0.40, '
        'norm_ratio ≈ 3.0). Use `fp8` for gradient-clean runs.',
        '- `adapter ✗`: run pre-dates adapter batching (added 2026-08-16). '
        'Cannot be re-evaluated under a different eval protocol without '
        'retraining.',
        '- `git_dirty=True` in individual runs: reproducibility caveat — '
        'source tree had uncommitted changes when the run started.',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    rows = collect_rows()
    print(f'scanned: {len(rows)} spec v1 JSONs')
    write_csv(rows, os.path.join(RESULTS_ROOT, '_INDEX.csv'))
    write_markdown(rows, os.path.join(RESULTS_ROOT, '_INDEX.md'))
    print(f'wrote: {RESULTS_ROOT}/_INDEX.csv')
    print(f'wrote: {RESULTS_ROOT}/_INDEX.md')


if __name__ == '__main__':
    main()
