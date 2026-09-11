#!/usr/bin/env python3
"""Package every LoRA adapter under results/ for deposit outside git.

The adapters (adapter_model.safetensors, bfloat16, about 2.3 GB in total) are
the one input of scripts/weight_distance.py, scripts/pa_all_arms.py and
scripts/audit/_check_levels.py that the repository does not carry. This writes,
into --out:

  oamp_adapters.tar          the adapter directories, paths as in results/
  SHA256SUMS                 sha256 of every file in the tar, and of the tar
  README.md                  adapter -> arm, method, encodings, model, seed,
                             training run JSON and perplexity JSONs

Arm letters use the same (method, body_encoding, pack_4d_mode) table as
_check_levels.py, restricted to the Llama-3.2-3B 7,473 x 2-epoch cohort of
the paper; every other adapter (pilots, Qwen, 70B, short training) is listed
with its configuration and no letter. The one adapter whose training JSON was
overwritten is attributed by directory, as Appendix D marks it.

Usage: python3 scripts/audit/make_adapter_archive.py --out /path/outside/repo
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _check_levels as L  # noqa: E402  (ARMS, BY_DIRECTORY, norm)

ROOT = L.ROOT


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dirs = sorted(os.path.relpath(os.path.dirname(f), ROOT)
                  for f in glob.glob(os.path.join(ROOT, 'results', '**', 'adapter_model.safetensors'), recursive=True))
    ppl_refs = {}
    for p in glob.glob(os.path.join(ROOT, 'results', 'ppl_*', '*.json')):
        d = json.load(open(p))
        ap_ = L.norm(d.get('adapter_path'))
        if ap_:
            ppl_refs.setdefault(ap_, []).append(os.path.relpath(p, ROOT))

    rows = []
    for d in dirs:
        run = d[:-len('_adapter')] + '.json' if d.endswith('_adapter') else None
        cfg = {}
        if run and os.path.exists(os.path.join(ROOT, run)):
            cfg = (json.load(open(os.path.join(ROOT, run))).get('config') or {})
        else:
            run = None
        m = cfg.get('method')
        key = ('standard', None, None) if m == 'standard' else (m, cfg.get('body_encoding'), cfg.get('pack_4d_mode'))
        arm = ''
        if cfg and 'Llama-3.2-3B' in (cfg.get('model_id') or '') and cfg.get('n_train') == 7473 and cfg.get('epochs') == 2:
            arm = L.ARMS.get(key, '')
        if not cfg:
            arm = L.BY_DIRECTORY.get(os.path.basename(os.path.dirname(d)), '')
            if arm:
                arm += ' (by directory)'
        rows.append({
            'adapter': d, 'arm': arm, 'method': m or '', 'body_encoding': cfg.get('body_encoding') or '',
            'pack_4d_mode': cfg.get('pack_4d_mode') or '', 'model': (cfg.get('model_id') or '').split('/')[-1],
            'seed': cfg.get('seed', ''), 'n_train': cfg.get('n_train', ''), 'epochs': cfg.get('epochs', ''),
            'run_json': run or '', 'ppl_json': sorted(ppl_refs.get(d, [])),
            'sha256_safetensors': sha256(os.path.join(ROOT, d, 'adapter_model.safetensors')),
        })

    tar_path = os.path.join(args.out, 'oamp_adapters.tar')
    sums = []
    with tarfile.open(tar_path, 'w') as tar:
        for d in dirs:
            for f in sorted(os.listdir(os.path.join(ROOT, d))):
                full = os.path.join(ROOT, d, f)
                tar.add(full, arcname=os.path.join(d, f))
                sums.append((sha256(full), os.path.join(d, f)))
    tar_sum = sha256(tar_path)
    with open(os.path.join(args.out, 'SHA256SUMS'), 'w') as f:
        for s, p in sums:
            f.write(f'{s}  {p}\n')
        f.write(f'{tar_sum}  oamp_adapters.tar\n')

    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ['# OAMP LoRA adapters', '',
             f'Written by `scripts/audit/make_adapter_archive.py` at {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} '
             f'from the tree at commit `{head}`. {len(dirs)} adapter directories, {os.path.getsize(tar_path):,} bytes in '
             f'`oamp_adapters.tar` (sha256 `{tar_sum}`). Unpack at the repository root; paths are the ones the training '
             'runs wrote, and the ones the perplexity JSONs name in `adapter_path`.', '',
             'Arm letters follow the paper (A: naive_fp4/e2m1/fp4, B: naive_fp4/e2m1/fp8, C: standard, D: naive_fp4/int4/fp4, '
             'E: naive_fp4/int4/fp8, F: naive_fp4/e2m1/chan_int4), for the Llama-3.2-3B, 7,473-example, two-epoch cohort only. '
             'Re-trainings of a seed (`_rerun_`, `_dup_` in the perplexity file name) are included and carry the same arm letter; '
             'the paper excludes them from arm aggregates.', '',
             '## License', '',
             '**Built with Llama.** Every adapter whose model column names Llama-3.2-3B-Instruct is a LoRA fine-tune of, '
             'and therefore a derivative of, Llama 3.2 materials; it is distributed under the Llama 3.2 Community License '
             '(Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright \u00a9 Meta Platforms, Inc. All Rights Reserved). '
             'The adapters trained on Meta-Llama-3.1-70B-Instruct are derivatives of Llama 3.1 materials under the Llama 3.1 '
             'Community License. The adapters trained on Qwen2.5-3B-Instruct are derivatives of Qwen2.5 and are subject to its '
             'license. The adapter weights carry no other restriction from the authors.', '',
             '| adapter | arm | method | body | 4-D | model | seed | train | training run | perplexity files | sha256 (safetensors) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        ppl = '<br>'.join(f'`{p}`' for p in r['ppl_json']) or '-'
        lines.append(f"| `{r['adapter']}` | {r['arm']} | {r['method']} | {r['body_encoding']} | {r['pack_4d_mode']} | "
                     f"{r['model']} | {r['seed']} | {r['n_train']}x{r['epochs']} | "
                     f"{'`' + r['run_json'] + '`' if r['run_json'] else '(training JSON absent)'} | {ppl} | `{r['sha256_safetensors'][:16]}` |")
    open(os.path.join(args.out, 'README.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump(rows, open(os.path.join(args.out, 'adapters.json'), 'w'), indent=1)
    by_arm = {}
    for r in rows:
        by_arm[r['arm'] or '(none)'] = by_arm.get(r['arm'] or '(none)', 0) + 1
    print(f"{len(dirs)} adapters -> {tar_path} ({os.path.getsize(tar_path)/1e9:.2f} GB), sha256 {tar_sum}")
    print('by arm:', ', '.join(f'{k}={v}' for k, v in sorted(by_arm.items())))


if __name__ == '__main__':
    main()
