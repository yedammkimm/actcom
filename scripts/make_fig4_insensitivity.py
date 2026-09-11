"""Figure 4 — what the four measurement layers can and cannot see.

The same eight arm-A runs under four metrics. Colour marks the held-out verdict;
only the rightmost panel separates the colours. Each panel keeps its own y range
so the actual spread of each metric is visible rather than normalised away.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# seed, GSM8K accuracy (unified protocol: stop_strings, max_new_tokens 512,
# n = 200, unified protocol), training loss (mean of step_losses[-500:] from
# each run's result JSON, i.e. every one of the final 500 optimiser steps, not
# the checkpoint-sidecar samples), GSM8K test perplexity, WikiText-2 perplexity
RUNS = [
    (42,   56.5, 0.8341, 2.5636, 15.7796),
    (123,  55.0, 0.8427, 2.5637, 18.4320),
    (456,  51.0, 0.8352, 2.5630, 16.9550),
    (789,  51.0, 0.8425, 2.5636, 15.9558),
    (1011, 56.5, 0.8280, 2.5648, 16.1016),
    (2024, 54.0, 0.8323, 2.5651, 16.0627),
    (3033, 52.5, 0.8341, 2.5669, 17.0183),
    (4042, 54.5, 0.8303, 2.5650, 16.9350),
]
TAU = 16.65
SAFE, DMG = '#2B5D9E', '#B3372B'
PANELS = [
    (1, 'GSM8K accuracy', 'stop, 512 tok, n=200', None),
    (2, 'training loss',  'last 500 steps',    None),
    (3, 'GSM8K perplexity', 'in-distribution', None),
    (4, 'WikiText-2 perplexity', 'held out',  TAU),
]
# Panels whose metric is not yet available for all eight runs are dropped
# rather than filled from a different protocol.
PANELS = [q for q in PANELS if all(r[q[0]] is not None for r in RUNS)]

plt.rcParams.update({
    'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})

fig, axes = plt.subplots(1, len(PANELS), figsize=(1.78*len(PANELS)+0.5, 2.9))
axes = [axes] if len(PANELS) == 1 else axes
for ax, (idx, title, sub, hline) in zip(axes, PANELS):
    vals = [r[idx] for r in RUNS]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.30 or 0.01
    for r in RUNS:
        dmg = r[4] > TAU
        ax.plot(0, r[idx], marker='^' if dmg else 'o', ms=5.2 if dmg else 4.9,
                mfc=DMG if dmg else SAFE, mec='white', mew=0.7, ls='none', zorder=4)
    if hline is not None:
        ax.axhline(hline, color=DMG, ls=(0, (4, 3)), lw=0.9, zorder=2)
        ax.text(-0.50, hline, r'$\tau$', color=DMG, fontsize=7.8, va='bottom')
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.55, 0.55); ax.set_xticks([])
    ax.set_title(title, fontsize=8.2, pad=13)
    ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha='center',
            fontsize=7.2, color='#7E858E')
    ax.text(0.5, -0.10, f'spread {hi-lo:.4g}'.rstrip('0').rstrip('.') if hi-lo < 1
            else f'spread {hi-lo:.2f}', transform=ax.transAxes, ha='center',
            fontsize=7.4, color='#5C636C')
    ax.yaxis.grid(True, color='#E4E7EB', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(bottom=False)

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([], [], marker='o', ls='none', mfc=SAFE, mec='white', mew=0.6, ms=4.9,
           label='held-out verdict: safe'),
    Line2D([], [], marker='^', ls='none', mfc=DMG, mec='white', mew=0.6, ms=5.2,
           label='held-out verdict: damaged'),
], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.10))
fig.subplots_adjust(wspace=0.55)

fig.savefig('figures/fig4_insensitivity.pdf', bbox_inches='tight')
fig.savefig('figures/fig4_insensitivity.png', dpi=220, bbox_inches='tight')
print('fig4 ok')
