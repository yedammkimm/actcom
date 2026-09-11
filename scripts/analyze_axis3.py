#!/usr/bin/env python3
"""Unified §5 aggregator across all four arms (2026-09-02).

Supersedes analyze_axis2.py, which had two defects this script fixes:

  1. It applied a damage threshold to GSM8K PPL. GSM8K carries a uniform
     arm-level offset (A sits ~+0.007 above C on every seed), so a threshold
     marks 8/8 "damaged" and the number is meaningless. GSM8K is reported here
     as an in-distribution stability metric only: per-arm spread alongside the
     arm-level offset vs C_STD. Small does not mean absent — +0.2% that is
     identical across every seed is a clean systematic observation, and this
     script reports it rather than calling it "no difference".

  2. It framed the A-vs-B comparison as a binomial test against a fixed 50%
     null. That 50% was estimated from arm A's own eight seeds, so using it as
     a known null uses the data twice and inflates significance by roughly an
     order of magnitude. The correct test for two arms is Fisher's exact test
     on the 2x2 table, computed here exactly from the hypergeometric
     distribution (scipy is not installed on the analysis host, and the
     codebase already prefers exact pure-Python tests over scipy).

Pre-registered damage rule (locked before the axis-2 launch, reused unchanged):
    tau = C_STD mean + 3 * B_4DFP8 sd = 16.08 + 3*0.19 = 16.65
    A seed with WikiText-2 PPL > tau counts as damaged.
WikiText is the only corpus the damage count is taken from. NarrativeQA and
GovReport are reported as supporting evidence for already-damaged seeds, never
counted, so the damage rate is not inflated by multiple comparisons.

Usage:
    python3 scripts/analyze_axis3.py
"""
import glob
import json
import os
import statistics as st
from math import comb

ROOT = '/home/yedam/HMA/HMA_Project'
THRESHOLD_WT = 16.65
C_WIKI_MEAN = 16.08     # C_STD WikiText mean used to set the threshold
B_WIKI_SD = 0.19        # B_4DFP8 3-seed WikiText sd used to set the threshold
GRAD_CLIP = 1.0         # cfg.grad_clip for every run in §5

ARMS = ('A_4DFP4', 'B_4DFP8', 'C_STD', 'D_INT4')

# Training runs per arm, in the order they should win on a seed collision:
# later globs override earlier ones for the same seed.
TRAIN_GLOBS = {
    'A_4DFP4': ['results/naive4bit_e2m1/accuracy__naive_fp4*.json',
                'results/axis2_seeds/accuracy__naive_fp4*.json'],
    # B seed 789 exists twice: an August pilot run and the axis-3 re-run. The
    # axis-3 run is listed last so it wins, and the duplicate is reported.
    'B_4DFP8': ['results/pilot_e2m1/accuracy__naive_fp4*.json',
                'results/pilot_e2m1_seeds/accuracy__naive_fp4*.json',
                'results/axis3_b4dfp8/accuracy__naive_fp4*.json'],
    'C_STD':   ['results/pilot_e2m1/accuracy__standard*.json',
                'results/pilot_pack4d_fp8_matrix/accuracy__standard*.json'],
    'D_INT4':  ['results/naive4bit_int4/accuracy__naive_fp4*.json'],
}

# Perplexity: the original three seeds split gsm/wiki from narr/gov across two
# directories; the axis-2 and axis-3 seeds carry all four corpora in one file.
PPL_GLOBS = {
    'A_4DFP4': [('results/ppl_eval/ppl__A_4DFP4__seed*.json', 'ppl__A_4DFP4__seed'),
                ('results/ppl_axis2/ppl_axis2__A_4DFP4__seed*.json', 'ppl_axis2__A_4DFP4__seed')],
    'B_4DFP8': [('results/ppl_eval/ppl__B_4DFP8__seed*.json', 'ppl__B_4DFP8__seed'),
                ('results/ppl_axis3/ppl_axis3__B_4DFP8__seed*.json', 'ppl_axis3__B_4DFP8__seed')],
    'C_STD':   [('results/ppl_eval/ppl__C_STD__seed*.json', 'ppl__C_STD__seed')],
    'D_INT4':  [('results/ppl_eval/ppl__D_INT4__seed*.json', 'ppl__D_INT4__seed')],
}
OOD_GLOBS = {a: (f'results/ppl_ood_extra/ppl_ood__{a}__seed*.json',
                 f'ppl_ood__{a}__seed') for a in ARMS}

CORPORA = [('gsm', 'gsm8k_test_ppl', 'GSM8K'),
           ('wiki', 'wikitext2_ppl', 'WikiText'),
           ('narr', 'narrativeqa_ppl', 'NarrQA'),
           ('gov', 'govreport_ppl', 'GovRep')]


def _seed_of(path, prefix):
    return int(os.path.basename(path).replace(prefix, '').replace('.json', ''))


def load_ppl():
    """dict[arm][seed] = {'gsm','wiki','narr','gov'}."""
    out = {a: {} for a in ARMS}
    for arm, specs in PPL_GLOBS.items():
        for pat, prefix in specs:
            for f in sorted(glob.glob(os.path.join(ROOT, pat))):
                d = json.load(open(f))
                r = out[arm].setdefault(_seed_of(f, prefix), {})
                for key, jkey, _ in CORPORA:
                    if d.get(jkey) is not None:
                        r[key] = d[jkey]
        pat, prefix = OOD_GLOBS[arm]
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            d = json.load(open(f))
            r = out[arm].setdefault(_seed_of(f, prefix), {})
            for key, jkey, _ in CORPORA:
                if r.get(key) is None and d.get(jkey) is not None:
                    r[key] = d[jkey]
    return out


def load_training():
    """dict[arm][seed] = {'grad': [...], 'acc_intrain': float, 'mnt': int, 'src': str}.

    Also returns the list of (arm, seed, [sources]) collisions so a seed that
    was trained twice is never silently double-counted.
    """
    out = {a: {} for a in ARMS}
    seen = {a: {} for a in ARMS}
    for arm, pats in TRAIN_GLOBS.items():
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(ROOT, pat))):
                if 'checkpoints' in f or '_adapter' in f:
                    continue
                d = json.load(open(f))
                c = d.get('config')
                if not c or c.get('seed') is None:
                    continue
                s = c['seed']
                r = d.get('results', {})
                trace = r.get('grad_norm_trace') or []
                out[arm][s] = {
                    'grad': [x[1] for x in trace],
                    'finite': all(x[2] for x in trace) if trace else None,
                    'acc_intrain': r.get('gsm8k_accuracy'),
                    'mnt': c.get('eval_max_new_tokens'),
                    'src': os.path.relpath(f, ROOT),
                }
                seen[arm].setdefault(s, []).append(os.path.dirname(os.path.relpath(f, ROOT)))
    # Recover runs whose result JSON was never written (process died during eval)
    # but whose per-checkpoint grad norms survive on disk.
    for arm, pats in TRAIN_GLOBS.items():
        for pat in pats:
            for ck in sorted(glob.glob(os.path.join(ROOT, pat.replace('.json', '.checkpoints.jsonl')))):
                if os.path.exists(ck.replace('.checkpoints.jsonl', '.json')):
                    continue
                base = os.path.basename(ck)
                if '__seed' not in base:
                    continue
                s = int(base.split('__seed')[1].split('__')[0])
                if s in out[arm]:
                    continue
                norms = []
                for line in open(ck):
                    try:
                        j = json.loads(line)
                    except ValueError:
                        continue
                    if j.get('grad_norm') is not None:
                        norms.append(j['grad_norm'])
                if norms:
                    out[arm][s] = {'grad': norms, 'finite': None, 'acc_intrain': None,
                                   'mnt': None,
                                   'src': os.path.relpath(ck, ROOT) + '  [recovered]'}
    collisions = [(a, s, v) for a in ARMS for s, v in seen[a].items() if len(v) > 1]
    return out, collisions


def load_accuracy():
    """§5 accuracy row — ONLY the unified protocol (stop_strings + mnt=512, n=200)."""
    out = {a: {} for a in ARMS}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results/reeval_stop_mnt512_n200/*.json'))):
        d = json.load(open(f))
        nm = os.path.basename(f).replace('reeval__', '').replace('__stop_mnt512_n200.json', '')
        arm, seed = nm.split('__seed')
        if arm in out:
            out[arm][int(seed)] = {'acc': d['results']['accuracy_pct'],
                                   'n': d.get('n_samples'),
                                   'mnt': d.get('max_new_tokens'),
                                   'stop': bool(d.get('stop_strings'))}
    return out


def fisher_exact_2x2(a, b, c, d):
    """Exact Fisher test on [[a, b], [c, d]] (rows = arms, col 0 = damaged).

    Returns (p_one_tailed_less, p_two_tailed). ``less`` is the probability of
    seeing this few damaged in row 2 or fewer, i.e. the directional test that
    arm 2 is safer than arm 1.
    """
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return float('nan'), float('nan')

    def prob(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    p_obs = prob(a)
    one = sum(prob(x) for x in range(a, hi + 1))          # row-2 damaged <= observed
    two = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12)
    return one, two


def fmt(v, w=8, p=4):
    return f'{v:>{w}.{p}f}' if v is not None else ' ' * (w - 1) + '-'


def main():
    ppl = load_ppl()
    train, collisions = load_training()
    acc = load_accuracy()

    print('=' * 92)
    print('§5 unified aggregate — 4 arms (A_4DFP4 / B_4DFP8 / C_STD / D_INT4)')
    print('=' * 92)

    if collisions:
        print()
        print('  NOTE — seeds trained more than once (last listed source wins, counted once):')
        for arm, s, dirs in collisions:
            print(f'    {arm} seed {s}: {" , ".join(dirs)}  ->  using {dirs[-1]}')

    # ---------------- Per-arm perplexity ----------------
    print()
    print('=' * 92)
    print(f'Perplexity by arm — damage counted on WikiText ONLY (pre-registered tau = {THRESHOLD_WT})')
    print('=' * 92)
    for arm in ARMS:
        seeds = sorted(ppl[arm])
        if not seeds:
            continue
        print()
        print(f'  {arm}   (n = {len(seeds)})')
        print(f"    {'seed':>5}  {'GSM8K':>8} {'WikiText':>9} {'NarrQA':>8} {'GovRep':>8}   WT verdict")
        for s in seeds:
            r = ppl[arm][s]
            wt = r.get('wiki')
            verdict = '-' if wt is None else ('DAMAGED' if wt > THRESHOLD_WT else 'safe')
            print(f"    {s:>5}  {fmt(r.get('gsm'))} {fmt(r.get('wiki'), 9)} "
                  f"{fmt(r.get('narr'))} {fmt(r.get('gov'))}   {verdict}")
        wt = [ppl[arm][s]['wiki'] for s in seeds if ppl[arm][s].get('wiki') is not None]
        if wt:
            dmg = [s for s in seeds if (ppl[arm][s].get('wiki') or 0) > THRESHOLD_WT]
            safe_v = [v for v in wt if v <= THRESHOLD_WT]
            dmg_v = [v for v in wt if v > THRESHOLD_WT]
            print(f'    WikiText damaged: {len(dmg)}/{len(wt)}   seeds {dmg}')
            if safe_v and dmg_v:
                gap = min(dmg_v) - max(safe_v)
                print(f'    bimodality: safe max {max(safe_v):.4f} -> damaged min {min(dmg_v):.4f}'
                      f'   gap {gap:.4f} = {gap / B_WIKI_SD:.2f} sd')

    # ---------------- GSM8K: offset, not threshold ----------------
    print()
    print('=' * 92)
    print('In-distribution stability (GSM8K PPL) — NOT threshold-tested')
    print('=' * 92)
    print('  A uniform arm-level offset is not seed-dependent damage; a threshold would')
    print('  mark every seed and mean nothing. Reported as spread vs offset instead.')
    print('  A small offset that is identical on every seed is a systematic effect,')
    print('  not an absence of one.')
    print()
    c_gsm = [ppl['C_STD'][s]['gsm'] for s in ppl['C_STD'] if ppl['C_STD'][s].get('gsm')]
    base = st.mean(c_gsm) if c_gsm else None
    print(f"    {'arm':<9} {'n':>2}  {'mean':>8} {'min':>8} {'max':>8} "
          f"{'spread':>8}  {'offset vs C_STD':>16}")
    for arm in ARMS:
        vals = [ppl[arm][s]['gsm'] for s in sorted(ppl[arm]) if ppl[arm][s].get('gsm')]
        if not vals:
            continue
        spread = max(vals) - min(vals)
        off = (st.mean(vals) - base) if base else None
        offs = (f'{off:+.4f} ({off / base * 100:+.2f}%)' if off is not None else '-')
        print(f'    {arm:<9} {len(vals):>2}  {st.mean(vals):>8.4f} {min(vals):>8.4f} '
              f'{max(vals):>8.4f} {spread:>8.4f}  {offs:>16}')
    if base:
        a = [ppl['A_4DFP4'][s]['gsm'] for s in ppl['A_4DFP4'] if ppl['A_4DFP4'][s].get('gsm')]
        if a:
            print()
            print(f'    A_4DFP4 reads as: spread {max(a) - min(a):.4f} vs offset '
                  f'{st.mean(a) - base:+.4f} — the offset is ~{(st.mean(a) - base) / (max(a) - min(a)):.1f}x')
            print('    the seed-to-seed spread, so it is systematic, not seed-dependent.')

    # ---------------- WikiText sensitivity sweep ----------------
    print()
    print('=' * 92)
    print('WikiText threshold sensitivity (2 / 3 / 4 sd) — is the damage count robust?')
    print('=' * 92)
    print(f'  Base: C_STD mean {C_WIKI_MEAN}, B_4DFP8 sd {B_WIKI_SD}')
    for arm in ARMS:
        vals = [(s, ppl[arm][s]['wiki']) for s in sorted(ppl[arm]) if ppl[arm][s].get('wiki')]
        if not vals:
            continue
        print()
        print(f'  {arm}')
        for k in (2, 3, 4):
            th = C_WIKI_MEAN + k * B_WIKI_SD
            hit = [(s, v) for s, v in vals if v > th]
            tag = '  <- pre-registered' if k == 3 else ''
            names = ', '.join(f'{s}({v:.2f})' for s, v in hit) or '-'
            print(f'    {k}sd  tau={th:>7.3f}   {len(hit):>2}/{len(vals):<2}   {names}{tag}')

    # ---------------- Fisher exact, A vs B ----------------
    print()
    print('=' * 92)
    print('Damage frequency: A_4DFP4 vs B_4DFP8 — Fisher exact test')
    print('=' * 92)
    counts = {}
    for arm in ('A_4DFP4', 'B_4DFP8'):
        vals = [ppl[arm][s]['wiki'] for s in ppl[arm] if ppl[arm][s].get('wiki')]
        counts[arm] = (sum(1 for v in vals if v > THRESHOLD_WT), len(vals))
    (da, na), (db, nb) = counts['A_4DFP4'], counts['B_4DFP8']
    print()
    print(f'                damaged   safe   n')
    print(f'    A_4DFP4       {da:>5}  {na - da:>5}  {na:>3}')
    print(f'    B_4DFP8       {db:>5}  {nb - db:>5}  {nb:>3}')
    one, two = fisher_exact_2x2(da, na - da, db, nb - db)
    print()
    print(f'    one-tailed (B safer than A)   p = {one:.4f}')
    print(f'    two-tailed                    p = {two:.4f}')
    print()
    print('    The one-tailed test is admissible because the direction was fixed in')
    print('    advance: the damage threshold is defined as C_STD mean + 3 * B_4DFP8 sd,')
    print('    which already presupposes B is the stable arm. State this in the paper.')
    print()
    print('    Do NOT report a binomial p against a 50% null. That 50% is estimated from')
    print("    arm A's own seeds, so using it as a known null uses the data twice.")
    if nb < 8:
        print()
        print(f'    B_4DFP8 is still at n={nb}. Projected outcomes at n=8:')
        for hyp in (0, 1, 2):
            o, t = fisher_exact_2x2(da, na - da, hyp, 8 - hyp)
            print(f'      B = {hyp}/8 damaged  ->  one-tailed p = {o:.4f}   two-tailed p = {t:.4f}')

    # ---------------- Detection depth ----------------
    print()
    print('=' * 92)
    print('Detection depth — which layer does each failure mode become visible at?')
    print('=' * 92)
    print(f'  Gradient norms are pre-clip; clip% is the share of logged steps above')
    print(f'  grad_clip={GRAD_CLIP}. (grad_norm_trace[2] is a FINITE flag, not a clip flag.)')
    print()
    print(f"    {'arm':<9} {'n':>2}  {'grad mean':>10} {'grad med':>9} {'grad max':>10} "
          f"{'clip%':>7}  {'GSM8K acc':>16}  {'in-dist ppl':>13}  {'OOD damaged':>11}")
    for arm in ARMS:
        gs = [train[arm][s]['grad'] for s in sorted(train[arm]) if train[arm][s]['grad']]
        flat = [x for g in gs for x in g]
        accs = [acc[arm][s]['acc'] for s in sorted(acc.get(arm, {}))]
        gsm = [ppl[arm][s]['gsm'] for s in ppl[arm] if ppl[arm][s].get('gsm')]
        wt = [ppl[arm][s]['wiki'] for s in ppl[arm] if ppl[arm][s].get('wiki')]
        if not flat:
            continue
        clip = sum(1 for x in flat if x > GRAD_CLIP) / len(flat) * 100
        accs_s = f'{min(accs):.1f}-{max(accs):.1f}' if accs else '-'
        gsm_s = f'{min(gsm):.3f}-{max(gsm):.3f}' if gsm else '-'
        ood_s = f'{sum(1 for v in wt if v > THRESHOLD_WT)}/{len(wt)}' if wt else '-'
        print(f'    {arm:<9} {len(gs):>2}  {st.mean(flat):>10.2f} {st.median(flat):>9.2f} '
              f'{max(flat):>10.1f} {clip:>7.1f}  {accs_s:>16}  {gsm_s:>13}  {ood_s:>11}')
    print()
    print('    Read down the columns, not across the arms: each failure mode surfaces at')
    print('    a different depth, and accuracy is the one column that separates nothing.')

    # ---------------- Per-seed grad detail ----------------
    print()
    print('=' * 92)
    print('Gradient norms per seed — does the gradient layer separate ARMS or SEEDS?')
    print('=' * 92)
    for arm in ARMS:
        seeds = [s for s in sorted(train[arm]) if train[arm][s]['grad']]
        if not seeds:
            continue
        print()
        print(f'  {arm}')
        for s in seeds:
            g = train[arm][s]['grad']
            clip = sum(1 for x in g if x > GRAD_CLIP) / len(g) * 100
            wt = ppl[arm].get(s, {}).get('wiki')
            verdict = '' if wt is None else ('  [WT DAMAGED]' if wt > THRESHOLD_WT else '  [WT safe]')
            note = '  <- recovered from checkpoints' if 'recovered' in train[arm][s]['src'] else ''
            print(f'    seed {s:<5} mean {st.mean(g):>8.2f}  med {st.median(g):>7.2f}  '
                  f'max {max(g):>9.1f}  clip {clip:>5.1f}%{verdict}{note}')

    # ---------------- Accuracy, protocol-isolated ----------------
    print()
    print('=' * 92)
    print('§5 accuracy row — unified protocol ONLY (stop_strings + max_new_tokens=512, n=200)')
    print('=' * 92)
    print('  In-training eval is a DIFFERENT protocol: run_experiment.py calls evaluate()')
    print('  without stop_strings and at eval_max_new_tokens=256. Those numbers are listed')
    print('  separately below and must never be pooled into the row above.')
    print()
    all_seeds = sorted({s for a in ARMS for s in acc.get(a, {})})
    print(f"    {'arm':<9} " + ' '.join(f'{s:>7}' for s in all_seeds) + f"  {'mean':>7}")
    for arm in ARMS:
        row = acc.get(arm, {})
        cells = ' '.join(f'{row[s]["acc"]:>7.1f}' if s in row else f'{"-":>7}' for s in all_seeds)
        vals = [row[s]['acc'] for s in row]
        m = f'{st.mean(vals):>7.1f}' if vals else f'{"-":>7}'
        print(f'    {arm:<9} {cells}  {m}')
    print()
    print('    In-training eval (mnt=256, no stop_strings) — NOT comparable to the row above:')
    for arm in ARMS:
        rows = [(s, train[arm][s]) for s in sorted(train[arm])
                if train[arm][s].get('acc_intrain') is not None]
        if not rows:
            continue
        cells = ', '.join(f'{s}:{v["acc_intrain"]:.1f}' for s, v in rows)
        mnts = {v['mnt'] for _, v in rows}
        print(f'      {arm:<9} mnt={sorted(mnts)}  {cells}')

    # ---------------- Supporting OOD evidence ----------------
    print()
    print('=' * 92)
    print('Supporting evidence for already-damaged seeds (NOT counted)')
    print('=' * 92)
    print('  NarrativeQA and GovReport are descriptive only. Counting damage across four')
    print('  corpora would inflate the rate through multiple comparisons, so the damage')
    print('  rate above comes from WikiText alone.')
    print()
    for arm in ARMS:
        dmg = [s for s in sorted(ppl[arm])
               if (ppl[arm][s].get('wiki') or 0) > THRESHOLD_WT]
        if not dmg:
            continue
        print(f'  {arm}')
        for s in dmg:
            r = ppl[arm][s]
            print(f"    seed {s:<5} WT {r['wiki']:.4f}   NarrQA {fmt(r.get('narr'))}   "
                  f"GovRep {fmt(r.get('gov'))}")


if __name__ == '__main__':
    main()
