"""Every value or phrase that has been retracted must be absent from the body.

Retractions accumulate across a long revision and a number that was correct in
one draft becomes wrong in the next without any edit touching it. This greps for
each retired item and prints its context so a hit can be judged rather than
assumed. A hit is not automatically a failure -- some are deliberately quoted as
"we do not use this" -- so the output is read, not counted.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'  # repository root
import re, sys
P=ROOT + 'docs/OAMP/OAMP_paper.tex'
src=open(P).read()
apx=src.index('\\appendix')
RETIRED=[
 ('bimodality as a claim',      r'distribution is bimodal|is bimodal,'),
 ('4.4 standard deviations',    r'4\.4 standard deviations'),
 ('p < 0.001 for a variance',   r'variance[^.]{0,80}p\s*<\s*0\.001'),
 ('670x attenuation',           r'670'),
 ('per-channel is free',        r'(free|no cost|costs nothing)[^.]{0,40}per-channel|per-channel[^.]{0,40}(is free|no bit cost)'),
 ('4/8 vs 1/8 as fix evidence', r'four of eight[^.]{0,60}one of eight'),
 ('1/8 vs 0/8 comparison',      r'one of eight[^.]{0,40}zero of eight'),
 ('CA trend as main test',      r'Cochran|Armitage'),
 ('continuous Spearman on ppl', r'Spearman[^.]{0,60}perplexity'),
 ('regression extrapolation',   r'R\^2|R\$\^2\$|extrapolat[^.]{0,40}FP8'),
 ('paired eight "also bimodal"',r'difference[^.]{0,30}also bimodal'),
 ('subspace run level as claim',r'subspace[^.]{0,80}predicts which run'),
 ('safeguard causal claim',     r'(safeguard|guard)[^.]{0,60}(causes|caused|responsible for)'),
 ('variance ratio 33.7 with p', r'33\.7[^.]{0,40}p\s*[=<]'),
 ('F-test as the quoted p',     r'F-test,\s*\$p'),
 # retired in the 2026-09-08 audit (AUDIT_v3_20260908.md), replacements in brackets
 ('0.447 bits (70B value quoted for 3B) [0.602 at 3B, 0.444 at 70B]',
                                r'0\.4(47|5)\s*(bits|\\,bits|bit)'),
 ('rotary 7.9% of promotion cost [5.9%]', r'7\.9\\%[^.]{0,60}(rotary|promot)|(rotary|promot)[^.]{0,60}7\.9\\%'),
 ('93-96% of head views attributed [88-94%]', r'9[36]\s*--\s*9[46]\s*\\%|9[36]\s*(to|and)\s*9[46]\s*\\%'),
 ('actcomp softmax collapse 0.4% [0.2%]', r'softmax[^.]{0,80}0\.4\\%|0\.4\\%[^.]{0,80}softmax'),
 ('39.8% step-time deviation (train_wall_s ratio) [x1.47 s/step]', r'39\.8'),
 ('variance ratio 33.66 -> 37.03 (App F) [38.47]', r'37\.03'),
 ('uncompressed runs differ by 0.007 [0.0094 / range 0.01]', r'differ by 0\.007'),
 ('seed 123 mean minus median 0.0008 [0.0007]', r'0\.0008'),
 ('70B extrapolated 100.89 GB [100.9]', r'100\.89'),
 ('LOO subspace 0.027, 0.042 (double rounding) [0.041]', r'0\.027,\s*0\.042'),
]
print(f'body is lines up to the appendix at char {apx}\n')
bad=0
for name,pat in RETIRED:
    hits=[m for m in re.finditer(pat,src[:apx],re.I)]
    if not hits:
        print(f'  clean   {name}')
        continue
    bad+=1
    print(f'  HIT     {name}  ({len(hits)})')
    for m in hits[:3]:
        a=max(0,m.start()-90); b=min(apx,m.end()+90)
        print('            ...'+' '.join(src[a:b].split())+'...')
print(f'\n{bad} of {len(RETIRED)} retired items appear in the body. Read each hit above:')
print('a deliberate "we do not quote this" mention is allowed, a live claim is not.')
