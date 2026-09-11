"""Leave-one-arm-out check behind Appendix E: can the four compressed arms
predict a held-out arm's subspace dispersion from its gradient cosine?

x  = arm gradient cosine, read from the dose-response probe
     results/audit/dose_response_prod_20260904_192407.json
     (qk_int4_block -> D, qk_e2m1_block -> A, qk_chan_int4_prod -> F,
      qk_fp8_block -> B)
y  = arm mean within-arm row-space cosine of the LoRA updates, from the pair
     matrix results/_wd/pa_all.npy (written by scripts/pa_all_arms.py, the
     same matrix behind Table tab:subspace)

For each of 30 transformations (six of x, five of y) a line y' = a + b x' is
fitted on three arms and evaluated at the fourth; the prediction is mapped back
to the y scale and the miss |prediction - actual| is divided by the held-out
arm's within-arm range (largest minus smallest pair value). A transformation
diverges in a fold when the back-transformed prediction is not finite or the
ratio exceeds 50.

The table prints every fold, because the four folds are not alike: D and B are
the endpoints of the cosine axis, so holding either out is an extrapolation,
while A and F are interpolations.

Usage: python scripts/prereg_loo.py        (standard library only)
"""
import array
import itertools
import json
import math
import os
import statistics as st
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, 'results', '_wd')
PROBE = os.path.join(ROOT, 'results', 'audit', 'dose_response_prod_20260904_192407.json')
ARM_FILTER = {'D': 'qk_int4_block', 'A': 'qk_e2m1_block', 'F': 'qk_chan_int4_prod', 'B': 'qk_fp8_block'}
ORDER = ['D', 'A', 'F', 'B']
CHANCE = 0.0607          # mean cosine of two random 16-dim subspaces of R^3072
DIVERGE = 50.0


def load_npy(path):
    with open(path, 'rb') as fh:
        assert fh.read(6) == b'\x93NUMPY'
        major, _ = fh.read(2)
        hlen = struct.unpack('<H', fh.read(2))[0] if major == 1 else struct.unpack('<I', fh.read(4))[0]
        header = eval(fh.read(hlen).decode())
        rows, cols = header['shape']
        data = array.array('d'); data.fromfile(fh, rows * cols)
        return [[data[i * cols + j] for j in range(cols)] for i in range(rows)]


def arm_values():
    tags = json.load(open(os.path.join(WD, 'tags_all.json')))
    PA = load_npy(os.path.join(WD, 'pa_all.npy'))
    ix = {t: i for i, t in enumerate(tags)}
    members = {}
    for t in tags:
        members.setdefault(t.split('_')[0], []).append(t)
    probe = json.load(open(PROBE))['checkpoints']['step0_fresh']['filters']
    x, y, rng = {}, {}, {}
    for a in ORDER:
        vals = [PA[ix[p]][ix[q]] for p, q in itertools.combinations(members[a], 2)]
        x[a] = probe[ARM_FILTER[a]]['aggregate']['cos']
        y[a] = st.mean(vals)
        rng[a] = max(vals) - min(vals)
    return x, y, rng


FX = {
    'x':         lambda c: c,
    'log x':     lambda c: math.log(c),
    'log(1-x)':  lambda c: math.log(1 - c),
    'logit x':   lambda c: math.log(c / (1 - c)),
    '1/(1-x)':   lambda c: 1 / (1 - c),
    'sqrt(1-x)': lambda c: math.sqrt(1 - c),
}
GY = {  # forward, inverse
    'y':             (lambda v: v, lambda w: w),
    'log y':         (math.log, math.exp),
    'log(y-chance)': (lambda v: math.log(v - CHANCE), lambda w: math.exp(w) + CHANCE),
    'logit y':       (lambda v: math.log(v / (1 - v)), lambda w: 1 / (1 + math.exp(-w))),
    '1/y':           (lambda v: 1 / v, lambda w: 1 / w),
}


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sum((p - mx) ** 2 for p in xs)
    return my - b * mx, b


def main():
    x, y, rng = arm_values()
    print('arm   cos(g,g*)   mean subspace cos   within-arm range   position on the x axis')
    lo, hi = min(x.values()), max(x.values())
    for a in ORDER:
        pos = 'endpoint (extrapolation when held out)' if x[a] in (lo, hi) else 'interior (interpolation when held out)'
        print(f'{a}     {x[a]:.4f}       {y[a]:.5f}            {rng[a]:.5f}          {pos}')
    print(f'\nratio = |back-transformed prediction of the held-out arm mean - actual| / within-arm range;'
          f' divergent = not finite or > {DIVERGE:g}')
    print(f"\n{'f(x)':10s} {'g(y)':14s} " + ' '.join(f'hold {a:>2s}' for a in ORDER) + f" {'median':>8s} {'mean':>9s}")
    table = []
    for fn, f in FX.items():
        for gn, (g, ginv) in GY.items():
            ratios = []
            for held in ORDER:
                train = [a for a in ORDER if a != held]
                a0, b0 = fit([f(x[a]) for a in train], [g(y[a]) for a in train])
                try:
                    pred = ginv(a0 + b0 * f(x[held]))
                except (OverflowError, ValueError, ZeroDivisionError):
                    pred = float('nan')
                ratios.append(abs(pred - y[held]) / rng[held])
            table.append((fn, gn, ratios))
            cells = ' '.join(f'{r:7.2f}' if math.isfinite(r) else '    nan' for r in ratios)
            finite = [r for r in ratios if math.isfinite(r)]
            print(f'{fn:10s} {gn:14s} {cells} {st.median(finite):8.2f} {st.mean(finite):9.2f}')
    print(f'\nper-fold summary over the {len(table)} transformations')
    print(f"{'held-out':10s} {'min':>7s} {'median':>8s} {'max':>8s} {'divergent':>10s}")
    for k, a in enumerate(ORDER):
        col = [row[2][k] for row in table]
        ok = [r for r in col if math.isfinite(r) and r <= DIVERGE]
        print(f'{a:10s} {min(ok):7.2f} {st.median(ok):8.2f} {max(ok):8.2f} {len(col) - len(ok):10d}')


if __name__ == '__main__':
    main()
