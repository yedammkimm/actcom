"""Figure 1 — activation memory against sequence length on 70B.

Measured points are markers on solid segments; everything past the last
measurement is a dashed linear extrapolation. Weights and optimiser state are a
39.65 GB constant that compression does not touch, which is why the compressed
curves converge on the baseline's slope rather than on a fraction of its height.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics as st

STD  = {1024: 71.37, 1536: 86.13}                 # L=2048 aborted at 100.62
STD_ABORT = (2048, 100.62)
FP8  = {1024: 63.23, 2048: 76.85, 3072: 90.95}
CHAN = {1024: 61.64, 2048: 74.74, 3072: 87.85}
CAP  = 128.0

def fit(d):
    xs = sorted(d); ys = [d[x] for x in xs]
    mx, my = st.mean(xs), st.mean(ys)
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    return sl, my - sl*mx

plt.rcParams.update({
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})

fig, ax = plt.subplots(figsize=(5.9, 3.9))
XHI = 6800
ax.axhspan(CAP, 145, color='#C4453A', alpha=0.07, lw=0)
ax.axhline(CAP, color='#B3372B', lw=1.1, zorder=3)
ax.text(1000, CAP+2.6, '128 GB', ha='left', fontsize=8, color='#B3372B',
        fontweight='bold')

LABEL_OFFSET = {
    'uncompressed':          (-6, -13, 'right'),
    '4-D FP8':               (-4, -13, 'right'),
    '4-D per-channel INT4':  ( 5,  -13, 'left'),
}
SERIES = [
    ('uncompressed',      STD,  '#4A5058', 'o'),
    ('4-D FP8',           FP8,  '#2B5D9E', 's'),
    ('4-D per-channel INT4', CHAN, '#1F7A70', 'D'),
]
for name, d, col, mk in SERIES:
    xs = sorted(d); ys = [d[x] for x in xs]
    sl, ic = fit(d)
    cross = (CAP - ic) / sl
    # Crossing lengths go in the legend rather than beside each marker: three
    # floating labels on ascending curves collide with the curves whichever
    # corner they are placed in.
    ax.plot(xs, ys, color=col, lw=1.5, marker=mk, ms=4.6, mec='white',
            mew=0.7, zorder=5, label=f'{name}  —  128 GB at L = {cross:.0f}')
    xe = [xs[-1], min(cross*1.06, XHI)]
    ax.plot(xe, [ic+sl*x for x in xe], color=col, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.plot([cross], [CAP], marker='v', ms=5, color=col, mec='white', mew=0.7, zorder=6)

# The uncompressed run that hit the watchdog. The marker is drawn; what it
# means goes in the caption, where there is room and no line to cross.
ax.plot([STD_ABORT[0]], [STD_ABORT[1]], marker='x', ms=7, mew=1.6,
        color='#4A5058', zorder=6, ls='none')

ax.set_xlim(900, XHI); ax.set_ylim(40, 145)
ax.set_xticks([1024, 2048, 3072, 4096, 5120, 6144])
ax.set_xlabel('sequence length, tokens')
ax.set_ylabel('peak reserved memory, GB')
ax.yaxis.grid(True, color='#E4E7EB', lw=0.6); ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(loc='lower right', frameon=False, handlelength=2.4,
          borderaxespad=0.9, labelspacing=0.5)

fig.savefig('figures/fig1_vram_scaling.pdf', bbox_inches='tight')
fig.savefig('figures/fig1_vram_scaling.png', dpi=220, bbox_inches='tight')
print('fig1 crossovers:', {n: round((CAP-fit(d)[1])/fit(d)[0]) for n, d, _, _ in SERIES})
