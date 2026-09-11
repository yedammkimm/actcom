"""Method figure — when compression happens, and what it dispatches on.

Two things a reader gets wrong about backward-only compression, and the figure
exists to prevent both.

(a) That the forward pass consumes the compressed value. It does not: the
    original tensor flows into the next operation, and the packed copy is only
    what sits in memory between the two passes. run_experiment.py enforces this
    before every run (INVARIANT-12): it computes the loss with hooks off and
    with hooks on and raises ParityError unless the difference is exactly 0.0,
    so this is an asserted invariant rather than a measured approximation.

(b) That the dispatch is a single rank test. It is a six-step chain whose order
    is load-bearing: after the cross-entropy reshape the logits are (B*L, vocab),
    i.e. two-dimensional and trailing-dim aligned, so a rank rule reached before
    the vocabulary filter would route them into four-bit quantization.

Numbers, all traced to source:
  - min_numel is 1024 (configs/base.py:92, and every result JSON). 128 is the
    group size and appears only on the INT4 route; the two must not be conflated.
  - the filter order is pack_hooks.py:252-280, verbatim.
  - the rank test is `tensor.dim() == 4`, not >= 4.
  - 15.04 / 84.96 % are numel_uniform_fp8_4d and numel_uniform_fp4_3d over
    numel_kept = 4,076,467,118,080, identical across all 3B runs.
  - per-filter tensor-count shares were removed: they use a different
    denominator (tensors reaching pack) from the routing split (stored
    elements), and two normalisations in one figure invite a reader to add
    them. The routing split is the only percentage the figure carries.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MID, GREY, FAINT = '#1B2027', '#5C636C', '#9AA0A8', '#C9CED4'
SKIP, INT4, PROT = '#9AA0A8', '#8A6520', '#6B4E9B'

plt.rcParams.update({
    'font.size': 8.5, 'font.family': 'DejaVu Sans',
    'pdf.fonttype': 42, 'ps.fonttype': 42})

fig = plt.figure(figsize=(6.4, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.10)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
for ax in (a, b):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(ax, x0, x1, y0, y1, ec, fc='none', lw=0.9, ls='-'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle='round,pad=0,rounding_size=1.4', ec=ec, fc=fc, lw=lw, ls=ls,
        mutation_aspect=0.5, zorder=3))

def arrow(ax, p, q, color=INK, lw=1.0, ls='-', head=2.6, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f'-|>,head_width={head/2.2},'
        f'head_length={head}', color=color, lw=lw, ls=ls,
        shrinkA=0, shrinkB=0, mutation_scale=3.2, zorder=z))

# ───────────────────────── (a) when ─────────────────────────
a.text(0, 99, '(a)', fontsize=10, fontweight='bold', color=INK)
a.text(7.5, 99, 'when', fontsize=9.2, color=MID)

# forward lane — the original tensor dominates it: solid, heavy
a.text(0, 87, 'F O R W A R D', fontsize=7.2, color=GREY, fontweight='bold')
a.text(6, 75, r'$t$', fontsize=12, color=INK, ha='center', va='center')
arrow(a, (10, 75), (36, 75), INK, lw=2.1, head=3.5)
box(a, 36, 58, 68.5, 81.5, INK, '#FFFFFF', lw=1.0)
a.text(47, 75, 'next op', fontsize=8.8, color=INK, ha='center', va='center')
arrow(a, (58, 75), (88, 75), INK, lw=2.1, head=3.5)
a.text(73, 85, 'original tensor, unquantized', fontsize=7.4, color=MID, ha='center')

# storing is a side effect: thin, dashed
arrow(a, (6, 70), (6, 57), GREY, lw=0.9, ls=(0, (2.4, 2.0)), head=2.4)
a.text(9, 63.5, 'pack()', fontsize=8, color=MID, style='italic')

box(a, 1, 41, 42, 56, INT4, '#FBF7EF', lw=1.1)
a.text(7.5, 49, r'$\tilde{t}$', fontsize=12, color=INT4, ha='center', va='center')
a.plot([14, 14], [44, 54], color=FAINT, lw=0.7)
a.text(17, 51.2, '4-bit nibbles', fontsize=7.6, color=INK)
a.text(17, 44.6, 'FP16 scale', fontsize=7.6, color=INK)
a.text(44, 49, 'the only copy held between the two passes',
       fontsize=7.6, color=MID, va='center')

a.plot([6, 6], [40, 28], color=FAINT, lw=0.8, ls=(0, (1, 2.4)))
a.text(9, 34, r'$\cdots$ time $\cdots$', fontsize=7.4, color=GREY, va='center')

# backward lane
arrow(a, (6, 26), (6, 17), GREY, lw=0.9, ls=(0, (2.4, 2.0)), head=2.4)
arrow(a, (7.5, 15.5), (27, 15.5), INK, lw=1.3, head=3.0)
a.text(17.5, 19, 'unpack()', fontsize=8, color=MID, style='italic', ha='center')
a.text(31, 15.5, r'$\hat{t}$', fontsize=12, color=INK, ha='center', va='center')
arrow(a, (35, 15.5), (50, 15.5), INK, lw=1.3, head=3.0)
a.text(57, 15.5, r'$\partial L/\partial x$', fontsize=10.5, color=INK,
       ha='center', va='center')
a.text(0, 3, 'B A C K W A R D', fontsize=7.2, color=GREY, fontweight='bold')

box(a, 68, 100, 8, 24, FAINT, '#F7F8F9', lw=0.8)
a.text(84, 18.5, r'$|\Delta L_{\mathrm{on}}-\Delta L_{\mathrm{off}}| = 0$',
       fontsize=9, color=INK, ha='center', va='center')
a.text(84, 12, 'asserted before every run', fontsize=7.2, color=MID, ha='center')

# ───────────────────────── (b) what ─────────────────────────
b.text(0, 99, '(b)', fontsize=10, fontweight='bold', color=INK)
b.text(7.5, 99, 'what', fontsize=9.2, color=MID)

SX = 22.0
b.text(SX, 95.0, 'tensor reaching pack()', fontsize=8.6, color=INK, ha='center')
b.plot([SX, SX], [92.0, 46.0], color=INK, lw=1.0, zorder=2)

FILT = [
    (88.0, 'a model parameter',                  'weights handed back by linear layers', 0),
    (80.5, 'fewer than 1024 elements',           'biases, norm statistics',              0),
    (73.0, 'not floating point',                 'dropout masks',                        0),
    (65.5, 'trailing dim = vocabulary size',     'logits, 2-D after the reshape',        1),
    (58.0, 'trailing dim not a multiple of 128', 'LoRA intermediates',                   0),
]
for y, cond, note, hot in FILT:
    col = PROT if hot else SKIP
    arrow(b, (SX, y), (35, y), col, lw=1.0 if hot else 0.9, head=2.4)
    b.text(36.5, y, 'skip', fontsize=8.2, color=col, va='center', fontweight='bold')
    b.text(46, y + 1.8, cond, fontsize=8.0, color=INK, va='center')
    b.text(46, y - 2.3, note, fontsize=7.2, color=PROT if hot else GREY, va='center')

# the order is load-bearing: bracket from the vocabulary filter to the rank test
b.plot([12.5, 8.5, 8.5, 12.5], [65.5, 65.5, 40.0, 40.0], color=PROT, lw=0.8,
       solid_joinstyle='miter', zorder=2)
b.text(7.0, 53.0, 'must precede', fontsize=7.0, color=PROT, rotation=90,
       va='center', ha='center')

box(b, 12.5, 33, 33.5, 46.0, INK, '#FFFFFF', lw=1.0)
b.text(22.8, 41.6, r'rank $\geq$ 4 ?', fontsize=8.8, color=INK, ha='center', va='center')
b.text(22.8, 36.6, 'head-view layout', fontsize=6.9, color=GREY, ha='center', va='center')

arrow(b, (33, 40.0), (41, 40.0), PROT, lw=1.3, head=2.8)
b.text(37, 42.4, 'yes', fontsize=7.4, color=PROT, ha='center')
box(b, 41, 96, 30.5, 49.5, PROT, '#F6F3FA', lw=1.1)
b.text(44, 44.4, 'PROTECTED', fontsize=8.4, color=PROT, fontweight='bold', va='center')
b.text(44, 39.6, 'FP8, or per-channel INT4', fontsize=7.8, color=INK, va='center')
b.text(44, 34.8, 'Q / K / V head views and rotary', fontsize=7.2, color=GREY, va='center')
b.text(93, 44.4, '15.0%', fontsize=9.6, color=PROT, ha='right', va='center',
       fontweight='bold')

b.plot([SX, SX], [33.5, 18.0], color=INK, lw=1.0, zorder=2)
arrow(b, (SX, 18.0), (41, 18.0), INT4, lw=1.3, head=2.8)
b.text(30, 20.4, 'no', fontsize=7.4, color=INT4, ha='center')
box(b, 41, 96, 8.5, 27.0, INT4, '#FBF7EF', lw=1.1)
b.text(44, 22.2, 'INT4', fontsize=8.4, color=INT4, fontweight='bold', va='center')
b.text(44, 17.4, 'groups of 128, per-group absmax', fontsize=7.8, color=INK, va='center')
b.text(44, 12.6, 'residual stream, MLP, attention output', fontsize=7.2, color=GREY,
       va='center')
b.text(93, 22.2, '85.0%', fontsize=9.6, color=INT4, ha='right', va='center',
       fontweight='bold')

fig.savefig('figures/fig_method.pdf', bbox_inches='tight')
fig.savefig('figures/fig_method.png', dpi=220, bbox_inches='tight')
print('wrote figures/fig_method.{pdf,png}')
