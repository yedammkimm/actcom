"""Compare the same-seed pairs of the determinism probe and write the summary.

Inputs are the run records that scripts/run_determinism_probe.sh writes to
results/determinism/. Runs are grouped by (arm, seed, optimizer steps, kernel
mode, SDPA backend), read from each record's config block, never from the
file name; arm membership is (method, body_encoding, pack_4d_mode) jointly,
as everywhere else in the repository. For every pair in a group the script
reports the first point at which the two runs differ on four ladders, from
the forward pass inwards:

    micro-step loss     results.micro_losses      (repr of every micro-step loss)
    optimizer-step loss results.step_losses
    gradient hash       results.grad_hash_trace   (sha256 of every LoRA gradient before clipping)
    parameter hash      results.param_hash_trace  (sha256 of every LoRA parameter after the update)

A pair whose parameter hashes agree at every step is bit-for-bit the same
training run. The step time is the mean of results.step_wall_s without the
first step (which includes kernel selection and allocator warm-up); hashing
is outside the timed region. The first two optimizer-step losses of every run
are also compared with the runs of the same arm and seed stored elsewhere in
results/, and, for seed 123, with the two values printed in Table 10 of the
paper.

Usage
    python scripts/audit/determinism_compare.py                       # results/determinism/, print
    python scripts/audit/determinism_compare.py --write results/determinism/summary.md
    python scripts/audit/determinism_compare.py --gate A.json B.json  # one line, for the chain
"""
import argparse
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Table 10 (tab:rerun) of the paper, seed 123, original run
# results/naive4bit_e2m1/accuracy__naive_fp4__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed123__20260826_073002.json
TABLE10_SEED123 = {1: 1.2956404238939285, 2: 1.6416974365711212}

# The training condition of the paper's 3B runs; stored runs are matched on these fields.
PAPER_3B = dict(n_train=7473, epochs=2, lr=2e-4, batch_size=1, grad_accum_steps=4, max_seq_len=512,
                optimizer='paged_adamw8bit', lora_dropout=0.05, weight_quant='nf4',
                model_id='meta-llama/Llama-3.2-3B-Instruct')


def arm_of(cfg):
    if cfg.get('method') == 'standard':
        return 'C'
    key = (cfg.get('method'), cfg.get('body_encoding'), cfg.get('pack_4d_mode'))
    return {('naive_fp4', 'e2m1', 'fp4'): 'A', ('naive_fp4', 'e2m1', 'fp8'): 'B'}.get(key, 'other')


def load_run(path):
    d = json.load(open(path))
    cfg, env, res = d['config'], d['env'], d.get('results', {})
    return {
        'path': path, 'name': os.path.basename(path)[:-5], 'cfg': cfg, 'env': env, 'res': res,
        'status': d.get('status'), 'error': d.get('error'),
        'arm': arm_of(cfg), 'seed': cfg.get('seed'),
        'mode': 'deterministic' if cfg.get('deterministic') else 'default',
        'backend': cfg.get('sdpa_backend', 'default'),
        'steps': cfg.get('max_opt_steps') or len(res.get('step_losses') or []),
        'throughput': d.get('throughput', {}),
    }


NR = 'n/r'   # ladder not recorded in one of the two records


def is_diff(v):
    return v not in (None, NR)


def first_diff_seq(a, b):
    """1-based index of the first differing element over the common length; None if
    equal on that length; NR if either side recorded nothing."""
    if not a or not b:
        return NR
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i + 1
    return None


def first_diff_trace(ta, tb):
    if not ta or not tb:
        return NR
    da = {int(s): h for s, h in ta}
    db = {int(s): h for s, h in tb}
    for s in sorted(set(da) & set(db)):
        if da[s] != db[s]:
            return s
    return None


def compare_pair(r1, r2):
    a, b = r1['res'], r2['res']
    return {
        'micro': first_diff_seq(a.get('micro_losses') or [], b.get('micro_losses') or []),
        'loss': first_diff_seq(a.get('step_losses') or [], b.get('step_losses') or []),
        'grad': first_diff_trace(a.get('grad_hash_trace'), b.get('grad_hash_trace')),
        'param': first_diff_trace(a.get('param_hash_trace'), b.get('param_hash_trace')),
        'n_steps': min(len(a.get('step_losses') or []), len(b.get('step_losses') or [])),
        'n_micro': min(len(a.get('micro_losses') or []), len(b.get('micro_losses') or [])),
    }


def mean_step_time(run):
    sw = run['res'].get('step_wall_s') or []
    if len(sw) > 1:
        return statistics.mean(sw[1:])
    return run['throughput'].get('seconds_per_step')


def fmt(x):
    return 'none' if x is None else str(x)


def stored_runs(arm, seed):
    """The paper's 3B runs of this arm and seed, outside results/determinism/."""
    out = []
    pat = os.path.join(ROOT, 'results', '**', f'accuracy__*__Llama-3.2-3B-Instruct__nf4__rmsbf16__seed{seed}__*.json')
    for f in sorted(glob.glob(pat, recursive=True)):
        rel = os.path.relpath(f, ROOT)
        if rel.startswith('results/determinism/'):
            continue
        try:
            d = json.load(open(f))
        except Exception:                      # noqa: BLE001
            continue
        cfg = d.get('config', {})
        if arm_of(cfg) != arm or any(cfg.get(k) != v for k, v in PAPER_3B.items()):
            continue
        sl = d.get('results', {}).get('step_losses') or []
        if len(sl) < 2:
            continue
        env = d.get('env', {})
        out.append({'path': rel, 'commit': (env.get('git_commit') or '')[:7], 'driver': env.get('driver_version'),
                    'torch': env.get('torch_version'), 'losses': sl[:2]})
    return out


def build_summary(runs, out_dir):
    L = []
    P = L.append
    P('# Determinism probe: summary')
    P('')
    P(f'Generated by `scripts/audit/determinism_compare.py` from {len(runs)} run records in `{os.path.relpath(out_dir, ROOT)}/`. '
      'Grouping and arm membership are read from each record\'s config block. Rules of reading are in the header of '
      '`scripts/run_determinism_probe.sh`, which was committed before the first cell ran.')
    P('')
    runs = sorted(runs, key=lambda r: (r['arm'], r['seed'], r['steps'], r['mode'], r['backend'], r['name']))
    # 1. environment
    P('## 1. Environment')
    P('')
    if runs:
        e = runs[0]['env']
        P('| field | value |')
        P('|---|---|')
        for k in ('hostname', 'gpu_name', 'compute_capability', 'driver_version', 'cuda_version', 'torch_version',
                  'transformers_version', 'peft_version', 'bitsandbytes_version', 'torchao_version', 'python_version'):
            P(f'| {k} | `{e.get(k)}` |')
        commits = sorted({r['env'].get('git_commit') for r in runs})
        dirty = sorted({str(r['env'].get('git_dirty')) for r in runs})
        P(f"| git_commit (all records) | {', '.join(f'`{c}`' for c in commits)} |")
        P(f"| git_dirty (all records) | {', '.join(dirty)} |")
        P(f"| timestamp_start, first and last | `{min(r['env'].get('timestamp_start','') for r in runs)}` to "
          f"`{max(r['env'].get('timestamp_start','') for r in runs)}` |")
    P('')
    # 2. per-run kernel condition
    P('## 2. Runs and their kernel condition')
    P('')
    P('| run | arm | seed | steps | mode | backend | det. algorithms | CUBLAS_WORKSPACE_CONFIG | cudnn det. | default SDPA dispatch | backend probe | status | git_dirty | steps done | mean step s (excl. 1st) |')
    P('|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|')
    for r in runs:
        e = r['env']
        mst = mean_step_time(r)
        P(f"| `{r['name']}` | {r['arm']} | {r['seed']} | {r['steps']} | {r['mode']} | {r['backend']} | "
          f"{e.get('deterministic_algorithms')} | `{e.get('cublas_workspace_config')}` | {e.get('cudnn_deterministic')} | "
          f"{e.get('sdpa_default_dispatch')} | {e.get('sdpa_backend_probe')} | {r['status']} | {e.get('git_dirty')} | "
          f"{r['throughput'].get('steps_completed')} | {mst:.3f} |" if mst is not None else
          f"| `{r['name']}` | {r['arm']} | {r['seed']} | {r['steps']} | {r['mode']} | {r['backend']} | "
          f"{e.get('deterministic_algorithms')} | `{e.get('cublas_workspace_config')}` | {e.get('cudnn_deterministic')} | "
          f"{e.get('sdpa_default_dispatch')} | {e.get('sdpa_backend_probe')} | {r['status']} | {e.get('git_dirty')} | "
          f"{r['throughput'].get('steps_completed')} | n/a |")
    P('')
    errs = [r for r in runs if r['status'] != 'OK']
    if errs:
        P('Records whose status is not OK:')
        P('')
        for r in errs:
            P(f"- `{r['name']}`: status {r['status']}, error: `{(r['error'] or '')[:600]}`")
        P('')
    # 3. pairs
    P('## 3. Same-seed pairs: first point of difference')
    P('')
    P('"none" means no difference over the steps both runs completed. The four columns are the ladders of the docstring: '
      'a difference that first appears in the micro-step loss is in the forward pass, one that first appears in the '
      'gradient hash is in the backward pass, one that first appears in the parameter hash is in the optimizer step.')
    P('')
    P('| arm | seed | steps | mode | backend | pair | micro-step loss | optimizer-step loss | gradient hash | parameter hash | steps compared | reading |')
    P('|---|---:|---:|---|---|---|---|---|---|---|---:|---|')
    groups = {}
    for r in runs:
        groups.setdefault((r['arm'], r['seed'], r['steps'], r['mode'], r['backend']), []).append(r)
    pair_results = {}
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r['name'])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                c = compare_pair(rs[i], rs[j])
                pair_results.setdefault(key, []).append(c)
                if not any(is_diff(c[k]) for k in ('micro', 'loss', 'grad', 'param')):
                    reading = f"identical through {c['n_steps']} optimizer steps"
                else:
                    firsts = {k: c[k] for k in ('micro', 'loss', 'grad', 'param') if is_diff(c[k])}
                    reading = 'separates: ' + ', '.join(f'{k} at {v}' for k, v in firsts.items())
                P(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | `{rs[i]['name']}` vs `{rs[j]['name']}` | "
                  f"{fmt(c['micro'])} | {fmt(c['loss'])} | {fmt(c['grad'])} | {fmt(c['param'])} | {c['n_steps']} | {reading} |")
    P('')
    # 4. overhead
    P('## 4. Step-time overhead of deterministic mode')
    P('')
    P('Mean of `results.step_wall_s` without the first step, averaged over the runs of a cell; the hashing is outside the timed region. '
      'Overhead is deterministic over default, minus one.')
    P('')
    P('| arm | seed | steps | default s/step | deterministic s/step | overhead |')
    P('|---|---:|---:|---:|---:|---:|')
    cells = {}
    for r in runs:
        mst = mean_step_time(r)
        if mst is not None and r['status'] == 'OK':
            cells.setdefault((r['arm'], r['seed'], r['steps']), {}).setdefault(r['mode'], []).append(mst)
    for key, modes in sorted(cells.items()):
        d = statistics.mean(modes['default']) if modes.get('default') else None
        t = statistics.mean(modes['deterministic']) if modes.get('deterministic') else None
        ov = f'{100 * (t / d - 1):+.1f}%' if d and t else 'n/a'
        P(f"| {key[0]} | {key[1]} | {key[2]} | {d:.3f} | {t:.3f} | {ov} |" if d and t else
          f"| {key[0]} | {key[1]} | {key[2]} | {'none' if d is None else f'{d:.3f}'} | "
          f"{'none' if t is None else f'{t:.3f}'} | {ov} |")
    P('')
    # 5. agreement with stored runs / Table 10
    P('## 5. First two optimizer-step losses against the stored runs and Table 10')
    P('')
    P('Equality is exact (the JSON float round-trips the fp32 loss). A stored run under a different torch build can '
      'legitimately differ; the torch version of each stored run is listed so that case is visible. '
      'Table 10 of the paper prints seed 123, steps 1 and 2, as 1.295640 and 1.641697; the stored values are '
      f'{TABLE10_SEED123[1]!r} and {TABLE10_SEED123[2]!r}.')
    P('')
    P('| probe run | step 1 | step 2 | reference | ref commit | ref driver | ref torch | step 1 equal | step 2 equal |')
    P('|---|---|---|---|---|---|---|---|---|')
    for r in runs:
        sl = r['res'].get('step_losses') or []
        if len(sl) < 2:
            continue
        refs = stored_runs(r['arm'], r['seed'])
        if r['seed'] == 123:
            refs = [{'path': 'Table 10 (tab:rerun), seed 123', 'commit': '', 'driver': '', 'torch': '',
                     'losses': [TABLE10_SEED123[1], TABLE10_SEED123[2]]}] + refs
        for ref in refs:
            P(f"| `{r['name']}` | {sl[0]!r} | {sl[1]!r} | `{ref['path']}` | {ref['commit']} | {ref['driver']} | {ref['torch']} | "
              f"{sl[0] == ref['losses'][0]} | {sl[1] == ref['losses'][1]} |")
        if not refs:
            P(f"| `{r['name']}` | {sl[0]!r} | {sl[1]!r} | (no stored run of arm {r['arm']} at seed {r['seed']}) | | | | | |")
    P('')
    # 6. op-level probe
    ops_files = sorted(glob.glob(os.path.join(out_dir, 'ops_probe__*.json')))
    if ops_files:
        P('## 6. Op-level probe: which kernels repeat bit-for-bit')
        P('')
        ops = {os.path.basename(f)[len('ops_probe__'):-5]: json.load(open(f)) for f in ops_files}
        modes = sorted(ops)
        rep = {m: ops[m]['repeats'] for m in modes}
        P('Each operation is run on identical inputs and every output compared bitwise with the first call '
          f"({', '.join(f'{m}: {rep[m]} repeats' for m in modes)}). "
          "An entry gives the number of repeats that differed; 'raised' means deterministic mode refused the operation, "
          'with the message below.')
        P('')
        P('| operation | ' + ' | '.join(modes) + ' |')
        P('|---|' + '---|' * len(modes))
        names = []
        for m in modes:
            for o in ops[m]['ops']:
                if o['name'] not in names:
                    names.append(o['name'])
        raised = []
        for n in names:
            row = []
            for m in modes:
                o = next((x for x in ops[m]['ops'] if x['name'] == n), None)
                if o is None:
                    row.append('n/a')
                elif o['status'] != 'ok':
                    row.append('raised'); raised.append((m, n, o['error']))
                else:
                    row.append(f"{o['mismatching_repeats']}/{rep[m]}" +
                               (f" (max |diff| {o['max_abs_diff']:.2e})" if o['mismatching_repeats'] else ''))
            P(f'| {n} | ' + ' | '.join(row) + ' |')
        P('')
        if raised:
            P('Operations refused or failed:')
            P('')
            for m, n, err in raised:
                P(f'- {m}, {n}: `{err}`')
            P('')
    # 7. nondeterministic ops file
    nd = os.path.join(out_dir, 'nondeterministic_ops.txt')
    if os.path.exists(nd):
        P('## 7. Deterministic-mode errors and warnings collected from the logs')
        P('')
        P('```')
        L.extend(open(nd).read().rstrip('\n').split('\n'))
        P('```')
        P('')
    # 8. outcome
    P('## 8. Outcome against the pre-registered reading rules')
    P('')
    det_pairs = [(k, c) for k, cs in pair_results.items() for c in cs if k[3] == 'deterministic']
    def_pairs = [(k, c) for k, cs in pair_results.items() for c in cs if k[3] == 'default']
    det_err = [r for r in runs if r['mode'] == 'deterministic' and r['status'] != 'OK']
    det_agree = all(not any(is_diff(c[x]) for x in ('micro', 'loss', 'grad', 'param')) for _, c in det_pairs)
    def_sep = [(k, c) for k, c in def_pairs if any(is_diff(c[x]) for x in ('micro', 'loss', 'grad', 'param'))]
    P(f'- deterministic-mode pairs compared: {len(det_pairs)}; all identical on every ladder: {det_agree if det_pairs else "n/a"}')
    P(f'- default-mode pairs compared: {len(def_pairs)}; pairs that separate: {len(def_sep)}')
    if def_sep:
        P('  - ' + '; '.join(f"arm {k[0]} seed {k[1]} ({k[2]} steps): " +
                              ', '.join(f'{x} at {c[x]}' for x in ('micro', 'loss', 'grad', 'param') if is_diff(c[x]))
                              for k, c in def_sep))
    P(f'- deterministic-mode records that are not OK: {len(det_err)}')
    if det_err:
        P('  - outcome 3 territory: ' + '; '.join(f"`{r['name']}`: {(r['error'] or '')[:200]}" for r in det_err))
    elif det_pairs and det_agree and def_sep:
        P('- reading: outcome 1. Under deterministic kernels the pairs are bit-for-bit identical over the probed steps, '
          'and under default kernels the same configuration separates; over these steps the kernels are the only source '
          'of the divergence.')
    elif det_pairs and not det_agree:
        P('- reading: outcome 2. A pair separates under deterministic kernels; a source that deterministic mode does not '
          'cover is present.')
    elif det_pairs and det_agree and def_pairs and not def_sep:
        P('- reading: the positive control did not reproduce at this length. The default-mode pairs did not separate '
          'over the probed steps, so the agreement under deterministic mode carries no information on its own.')
    else:
        P('- reading: incomplete; not every cell is present.')
    P('')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--dir', default=os.path.join(ROOT, 'results', 'determinism'))
    ap.add_argument('--write', default=None, help='write the markdown summary here')
    ap.add_argument('--gate', action='store_true',
                    help='exactly two records: print one line with the first differing point of each ladder')
    a = ap.parse_args()
    if a.gate:
        if len(a.paths) != 2:
            sys.exit('--gate needs exactly two record paths')
        r1, r2 = load_run(a.paths[0]), load_run(a.paths[1])
        c = compare_pair(r1, r2)
        print(f"first_diff_micro={fmt(c['micro'])} first_diff_loss={fmt(c['loss'])} "
              f"first_diff_grad={fmt(c['grad'])} first_diff_param={fmt(c['param'])} steps={c['n_steps']} "
              f"status={r1['status']},{r2['status']} git_dirty={r1['env'].get('git_dirty')},{r2['env'].get('git_dirty')}")
        return
    paths = a.paths or [f for f in sorted(glob.glob(os.path.join(a.dir, '*.json')))
                        if not os.path.basename(f).startswith('ops_probe__')]
    runs = []
    for p in paths:
        try:
            runs.append(load_run(p))
        except Exception as e:                  # noqa: BLE001
            print(f'skipping {p}: {type(e).__name__}: {e}', file=sys.stderr)
    out_dir = a.dir if not a.paths else os.path.dirname(os.path.abspath(a.paths[0]))
    text = build_summary(runs, out_dir)
    if a.write:
        with open(a.write, 'w') as f:
            f.write(text)
        print(f'wrote {a.write}')
    else:
        print(text)


if __name__ == '__main__':
    main()
