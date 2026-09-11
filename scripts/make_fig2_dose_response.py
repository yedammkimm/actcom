"""Figure 2 — damage as a function of Q/K gradient fidelity.

One mark per training run. The x coordinate is the gradient cosine measured on
a fresh model when only the Q/K head views are compressed the given way, all
from one probe session (2026-09-04) so the four coordinates share a scale.

The y axis is broken because arm D reaches 33.75: a linear axis tall enough for
it flattens the 15.8-18.4 band where the whole result lives, and a log axis
hides the difference between 15.8 and 18.4.

The two shaded bands carry no in-figure label. Labelling them inside the axes
put text over the F and B columns whichever corner it went in, and the caption
has room to say what they are.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TAU, CSTD, BASE = 16.65, 16.0765, 16.1934

# (x, short, long, values) — a list, not a dict: B and E share x = 0.9919.
ARMS = [
    (0.3712, 0.0,    'D', 'INT4 body\nblockwise 4-bit',
     [17.2634, 17.6943, 33.7466]),
    (0.5868, 0.0,    'A', 'E2M1 body\nblockwise 4-bit',
     [15.7796, 15.9558, 16.0627, 16.1016, 16.9350, 16.9550, 17.0183, 18.4320]),
    (0.9103, 0.0,    'F', 'E2M1 body\nper-channel 4-bit',
     [15.9908, 16.1257, 16.2414, 16.2876, 16.3664, 16.5298, 16.5877, 16.6579]),
    (0.9919, -0.014, 'B', 'E2M1 body\nFP8',
     [15.8887, 15.9331, 16.0008, 16.0253, 16.0370, 16.0927, 16.2828, 16.3029]),
    (0.9919, +0.014, 'E', 'INT4 body\nFP8',
     [15.9250, 16.0782, 16.4676]),
]

SAFE, DMG, GREY = '#2B5D9E', '#B3372B', '#7E858E'
JIT = 0.0060

plt.rcParams.update({
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 0.7,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# B and E are pushed apart so both columns are legible; a bracket under the
# axis records that they sit at one cosine.
ARMS = [(x, {'B': -0.021, 'E': +0.021}.get(sh, 0.0), sh, lg, v)
        for x, _o, sh, lg, v in ARMS]

XLO, XHI = 0.305, 1.075
fig, (top, bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6.3, 4.3),
    gridspec_kw={'height_ratios': [1, 5.6], 'hspace': 0.08})

for ax in (top, bot):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E4E7EB', linewidth=0.6)
    ax.set_xlim(XLO, XHI)

top.set_ylim(33.2, 34.3); top.set_yticks([33.5, 34.0])
bot.set_ylim(15.62, 19.15); bot.set_yticks([16, 17, 18, 19])

# Reference band: uncompressed adapter (16.08) to the un-fine-tuned model
# (16.19). Every safe run sits at or below it; every damaged run above.
bot.axhspan(CSTD, BASE, color=GREY, alpha=0.16, lw=0, zorder=1)
bot.axhline(TAU, color=DMG, ls=(0, (5, 3)), lw=1.0, zorder=3)
bot.text(XHI - 0.008, TAU + 0.06, r'$\tau = 16.65$', color=DMG, fontsize=8.2,
         ha='right', va='bottom')

# The empty band that makes arm A bimodal.
gap_lo, gap_hi = 16.1016, 16.9350
ax_frac = lambda v: (v - XLO) / (XHI - XLO)
bot.axhspan(gap_lo, gap_hi, xmin=ax_frac(0.5868 - 0.050), xmax=ax_frac(0.5868 + 0.050),
            color='#D9A93C', alpha=0.13, zorder=2, lw=0)


for x, off, short, long, vals in ARMS:
    xc = x + off
    n = len(vals)
    for i, v in enumerate(sorted(vals)):
        xs = xc + (JIT * (2 * i / (n - 1) - 1) if n > 1 else 0.0)
        dmg = v > TAU
        ax = top if v > 19.15 else bot
        ax.plot(xs, v, marker='^' if dmg else 'o', ms=4.8 if dmg else 4.5,
                mfc=DMG if dmg else SAFE, mec='white', mew=0.65, ls='none', zorder=5)
    nd = sum(1 for v in vals if v > TAU)
    bot.text(xc, 18.98, f'{nd}/{n}', ha='center', fontsize=8.6,
             color=DMG if nd else SAFE, fontweight='bold')


d = 0.012
for ax, y in ((top, -d * 5.6), (bot, 1 - d)):
    ax.plot([0, 1], [y] * 2, transform=ax.transAxes, color='k', lw=0.8,
            clip_on=False, marker=[(-1, -0.6), (1, 0.6)], markersize=5,
            mew=0.8, ls='none')
top.spines['bottom'].set_visible(False)
bot.spines['top'].set_visible(False)
for ax in (top, bot):
    ax.spines['right'].set_visible(False)
top.tick_params(bottom=False)

bot.set_xlabel('Q/K gradient cosine against the uncompressed backward pass',
               labelpad=6)
bot.set_ylabel('WikiText-2 perplexity')
bot.yaxis.set_label_coords(-0.075, 0.60)
# Arm letter and cosine on one tick label: B and E share the cosine, so they
# are named together rather than bracketed under a third row of text.
bot.set_xticks([0.3712, 0.5868, 0.9103, 0.9919])
bot.set_xticklabels(['D\n0.371', 'A\n0.587', 'F\n0.910', 'B, E\n0.992'])

# Inside the axes: the band between x = 0.62 and 0.88 holds no data.
bot.legend(handles=[
    Line2D([], [], marker='o', ls='none', mfc=SAFE, mec='white', mew=0.6, ms=4.5,
           label='safe'),
    Line2D([], [], marker='^', ls='none', mfc=DMG, mec='white', mew=0.6, ms=4.8,
           label='damaged'),
], loc='upper left', bbox_to_anchor=(0.30, 0.82), frameon=False,
   handletextpad=0.5, labelspacing=0.35)

fig.savefig('figures/fig2_dose_response.pdf', bbox_inches='tight')
fig.savefig('figures/fig2_dose_response.png', dpi=220, bbox_inches='tight')
print('wrote figures/fig2_dose_response.{pdf,png}')
