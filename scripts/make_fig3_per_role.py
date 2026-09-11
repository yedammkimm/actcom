"""Figure 3 — where the two four-bit grids actually differ.

Each role is compressed in isolation while every other tensor passes through
uncompressed, so the cosine is attributable to that role alone. The x axis
starts at 0.30 because every role except Q/K sits against 1.0.

Every number is read from one probe session,
results/audit/mechanism_int4_vs_e2m1_20260831_080053.json (Llama-3.2-3B, B=1,
L=512, step 0 before fine-tuning):

  cosine   filters.<role>_int4 / filters.<role>_e2m1 -> aggregate.cos
  share    filters.dist_stats.dist_stats_by_role[<role>].numel_share, i.e.
           n_elements / dist_stats_total_numel (699,269,120): the role-attributed
           elements the probe saw, counted as autograd saves them. Training runs
           do not deduplicate saved tensors, so every saved copy is a stored copy
           and the shares are shares of what is stored. The six roles sum to
           80.6 %; the remaining 19.4 % is the normalized residual saved once by
           each of q/k/v_proj (3 x 6.3 %), the up_proj input and the o_proj
           output. Rotary tables and logits carry no role attribution and are
           outside the denominator.

Role groupings for the share column:
  residual pre-attn   residual_pre_attn
  attention output    o_proj.in
  residual pre-MLP    residual_pre_mlp
  MLP intermediate    gate_proj.out + up_proj.out + down_proj.in
  V head views        v_proj.out
  Q / K head views    q_proj.out + k_proj.out

Usage: python scripts/make_fig3_per_role.py [--out-dir figures]
"""
import argparse
import json
import os

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, 'results', 'audit',
                     'mechanism_int4_vs_e2m1_20260831_080053.json')
# (label, INT4 filter, E2M1 filter, dist_stats roles summed for the share)
SPEC = [
    ('residual pre-attn', 'residual_pre_attn_int4', 'residual_pre_attn_e2m1', ['residual_pre_attn']),
    ('attention output',  'o_int4',                 'o_e2m1',                 ['o_proj.in']),
    ('residual pre-MLP',  'residual_pre_mlp_int4',  'residual_pre_mlp_e2m1',  ['residual_pre_mlp']),
    ('MLP intermediate',  'mlp_int4',               'mlp_e2m1',               ['gate_proj.out', 'up_proj.out', 'down_proj.in']),
    ('V head views',      'v_int4_block',           'v_e2m1_block',           ['v_proj.out']),
    ('Q / K head views',  'qk_int4_block',          'qk_e2m1_block',          ['q_proj.out', 'k_proj.out']),
]
INT4, E2M1 = '#8A6520', '#1F7A70'


def load_roles(path=PROBE):
    d = json.load(open(path))
    f = d['checkpoints']['step0_fresh']['filters']
    by_role = f['dist_stats']['dist_stats_by_role']
    total = f['dist_stats']['dist_stats_total_numel']
    roles = []
    for label, fi, fe, share_roles in SPEC:
        share = 100.0 * sum(by_role[r]['n_elements'] for r in share_roles) / total
        roles.append((label, f[fi]['aggregate']['cos'], f[fe]['aggregate']['cos'], share))
    return roles, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args()
    roles, total = load_roles()
    six = sum(r[3] for r in roles)
    print(f'denominator {total:,} elements; six roles = {six:.2f} %')
    for name, a, b, share in roles:
        print(f'  {name:18s} INT4 {a:.7f}  E2M1 {b:.7f}  share {share:.2f} %')

    plt.rcParams.update({
        'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'font.family': 'DejaVu Sans',
        'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})

    fig, ax = plt.subplots(figsize=(5.9, 3.3))
    y = np.arange(len(roles)); h = 0.34
    ax.barh(y+h/2, [r[1] for r in roles], height=h, color=INT4, label='INT4  uniform grid')
    ax.barh(y-h/2, [r[2] for r in roles], height=h, color=E2M1, label='E2M1  non-uniform grid')

    for i, (name, a, b, share) in enumerate(roles):
        d = b - a
        txt = f'{d:+.3f}' if abs(d) >= 0.0005 else '±0.000'
        ax.text(1.005, i, txt, va='center', fontsize=7.4,
                color='#1B2027' if abs(d) > 0.05 else '#7E858E',
                fontweight='bold' if abs(d) > 0.05 else 'normal')
        ax.text(1.088, i, f'{share:.1f}%', va='center', fontsize=7.4, color='#5C636C')

    ax.text(1.005, len(roles)-0.42, 'Δcos', fontsize=7.2, color='#5C636C', fontweight='bold')
    ax.text(1.088, len(roles)-0.42, 'share of\nstored elements', fontsize=7.2,
            color='#5C636C', linespacing=1.15)
    ax.text(1.088, -0.52, f'six roles = {six:.1f} %', fontsize=6.9, color='#9AA0A8')

    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in roles])
    ax.set_xlim(0.30, 1.0); ax.set_ylim(-0.65, len(roles)-0.05)
    ax.set_xlabel('gradient cosine against the uncompressed backward pass')
    ax.xaxis.grid(True, color='#E4E7EB', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.20), ncol=2,
              frameon=False, handletextpad=0.6, columnspacing=2.0)

    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, 'fig3_per_role.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, 'fig3_per_role.png'), dpi=220, bbox_inches='tight')
    print('fig3 ok ->', args.out_dir)


if __name__ == '__main__':
    main()
