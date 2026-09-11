"""Rule 10: every sentence that promises something is in an appendix must be
matched against what the appendix actually contains.

LaTeX validates \ref but not the prose around it. Twice during the 2026-09-08
reframing a sentence said "Appendix A gives the packing layout" when the
appendix had no packing layout, because the block had been cut in the same edit
that wrote the sentence. This lists every promise so the pairing can be read.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import re, sys
src = open(ROOT + 'docs/OAMP/OAMP_paper.tex').read()
apx_at = src.index('\\appendix')
body, apx = src[:apx_at], src[apx_at:]

# appendix sections and their text
secs = [(m.start(), m.group(1)) for m in re.finditer(r'\\section\{([^}]*)\}', apx)]
sec_text = {}
for i, (p, name) in enumerate(secs):
    end = secs[i + 1][0] if i + 1 < len(secs) else len(apx)
    sec_text[name] = apx[p:end]
labels = {}
for name, t in sec_text.items():
    for m in re.finditer(r'\\label\{(app:[^}]+)\}', t):
        labels[m.group(1)] = name

print(f'{len(secs)} appendix sections: ' + ', '.join(n[:34] for _, n in secs))
print(f'labels: {labels}\n')
print('promises made in the body:\n')
n = 0
for m in re.finditer(r'[^.]*[Aa]ppendix[^.]*\.', body):
    s = ' '.join(m.group(0).split())
    ref = re.search(r'\\ref\{(app:[^}]+)\}', s)
    tgt = labels.get(ref.group(1), '??') if ref else '(no \\ref)'
    n += 1
    print(f'  [{n}] -> {tgt}')
    print(f'      {s[:190]}')
print(f'\n{n} promises. Read each against the section named, by hand: a \\ref that')
print('resolves proves the section exists, not that it contains what was promised.')
