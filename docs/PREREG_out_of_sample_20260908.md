# Pre-registration — out-of-sample validation of the gradient-fidelity probe

**Written 2026-09-08, before any new arm is trained.** Two cells in §2 are
deliberately empty and are filled only after the probe sweep runs; the commit
that fills them is the one the paper cites, and training starts only after it.
Everything else is fixed now.

## 0. What is being validated

| | |
|---|---|
| already established | in our data, gradient cosine predicts dispersion — a finding |
| to be validated | given a configuration we have never trained, the probe orders the outcome — a tool |

The prediction is **ordinal, not quantitative.** Leave-one-out on the existing
four arms misses the fourth point by three to ten times an arm's internal range
(thirty transformations tried; the best was 3.2×). Predictions of the form
"cosine 0.75 implies sd 0.3" are therefore not registered.

## 1. What is predicted

### Primary — subspace dispersion (geometry)

Take `ΔW = B·A` per LoRA module, compute the principal-angle cosine between the
row spaces of two runs **per module**, and average by arm using the same
aggregation as the existing D/A/F/B/C computation. Per-module first:
concatenating across layers lets layer size dominate. The code path must be
byte-identical to the one that produced the existing numbers.

Primary because the within-arm range is 0.002 against between-arm gaps of 0.011
to 0.029, a factor of 4.5 to 14, with no overlap across 88 pairs — and because
the estimate stays tight at n=3 (arm D's range is 0.00175).

### Secondary — item-level disagreement (behaviour)

Pairwise disagreement rate over all benchmark items, averaged by arm. Existing
values: C 6.478, B 6.594, A 7.387, D 9.079 — monotone in reverse cosine order.
With MMLU the item count reaches 20,695.

Secondary because it is a *behavioural* measure, which forecloses the objection
that we predicted a quantity nobody uses. The primary validates the tool; the
secondary is the result.

### Tertiary — log sd of WikiText-2 perplexity

Registered, and registered as weak. At four seeds per arm the relative standard
error of log sd is 41%, and the observed F(0.233) against B(0.151) ratio of 1.54
gives F-test p = 0.247 at n=4. Adjacent arms may invert.

**Damage count is not registered.** It depends on τ, and 1/8 against 0/8 is
Fisher p = 1.000.

## 2. Selecting the new arms — filled after the probe sweep

### 2.1 Placement on the cosine axis

Resolution, not preference, fixes how many arms fit. The spacing must clear the residual measured in §5, not the within-arm range.
Two arms the probe calls identical still differ by 0.00256, so a new arm needs
about three times that, 0.0077, to be distinguishable from its neighbour:

| region | slope | spacing for 0.0077 of separation |
|---|---|---|
| cos < 0.6 | 0.135 | Δcos ≥ 0.057 |
| cos 0.6–0.9 | 0.034 | **Δcos ≥ 0.23** ← the binding constraint |
| cos > 0.9 | 0.132 | Δcos ≥ 0.058 |

Existing arms sit at 0.371, 0.587, 0.910, 0.992. Against these requirements the
0.587–0.910 window (0.323 wide, needing 0.46 for a point with clearance on both
sides) and the 0.910–0.992 window (0.082 wide, needing 0.116) are both closed.
**Two slots remain, not four:** below 0.371 and inside 0.371–0.587. The earlier
four-slot plan rested on the within-arm range of 0.002, which the arm E result
showed is not the relevant residual.

**Targets are replaced by whatever the probe actually produces.** If a target
cannot be hit, the actual value is used and the predicted rank is written to
match it. Nothing is forced onto a target.

| slot | measured cos(g, g*) | intervention | predicted rank among all 6 arms |
|---|---|---|---|
| below 0.371 | *(empty until the sweep)* | *(empty)* | *(empty)* |
| 0.371–0.587 | *(empty)* | *(empty)* | *(empty)* |

### 2.2 Diversity of intervention

Two arms that are both group-size variants would be one knob turned twice. The
two must come from **different interventions**, drawn from:
`q_only` protection, `k_only` protection, layer threshold (`layers < k` for
k = 1, 2, 4, 8), group size (g = 32, 256, 512), and head-view grid (E2M1 against
INT4, crossed with per-channel against blockwise).

Group-size arms move bits per element as 16/g — 0.5 at g=32 against 0.0625 at
g=256 — so they must not enter the bits-held-fixed contrast of the paper's §5.2,
which stays computed on the D/A/F subset and says so.

### 2.3 Ties — rule withdrawn 2026-09-08

The original rule registered two arms of equal cosine as a tie, predicted not to
resolve. The arm E result below falsifies that: two arms with identical probe
values resolved cleanly, with non-overlapping ranges. Equal cosine does not
imply equal outcome, so ties are no longer registered as predictions of
equality. What replaces the rule is a known quantity: **two configurations that
the probe cannot distinguish still differ by about 0.0026 in subspace
dispersion**, and that residual is what §2.1's spacing must clear.

## 3. Prediction and test

Registered per new arm: the measured `cos(g, g*)` from the probe (before any
training), the predicted rank among all six arms (or an interval such as
"between F and B"), and the basis — which is the cosine value and nothing else.

Primary test: Spearman correlation between cosine and subspace dispersion over
all six arms, with a permutation p. A perfect ordering gives one-sided exact
p = 1/6! = 1.4×10⁻³; a partial one gives the permutation p of the observed ρ.
Reported alongside: how many of the new arms landed in their predicted
interval, which needs no explanation.

Secondary and tertiary use the same form, with the tertiary carrying its
"adjacent arms may not resolve at n=4" caveat stated in advance.

### 3.4 Reporting rules

1. Every arm trained is reported. A bad result is not a reason to drop one.
2. Seeds per arm are fixed at registration and not adjusted after seeing results.
3. A run that fails (OOM, non-finite, interruption) is re-run at the same config
   and the failure is reported. Failure is not a reason to change the arm.
4. If the primary fails and the secondary succeeds, we do not write that the
   secondary was the primary.
5. If the sweep cannot produce a cosine in some region, that region is reported
   empty rather than filled with something else.

## 4. Budget

Two arms × four seeds = 8 runs, ≈ 3.3 h per run at 3B and 3,736 steps, so
≈ 26 GPU-h, plus ≈ 8 h of per-adapter evaluation, for ≈ 34 GPU-h total. The arm
E result halved this by closing two of the four slots.

Four seeds rather than three: three suffices for the primary (arm D reaches a
range of 0.00175 at n=3) but makes the tertiary meaningless. If the budget is
short, drop an arm rather than a seed.

## 5. Arm E — registered, measured, and wrong

The prediction was committed as `2479f1a` before the measurement was taken.

> **Registered.** Arm E's mean within-arm row-space cosine falls inside arm B's
> range, 0.13297 to 0.13495, because the probe assigned the two arms the same
> cosine (0.9919) and nothing else about arm E was used.

**Measured: 0.13137.** Outside the range, below its floor by 0.00160, and arm
E's own three pairs span only 0.13123 to 0.13149, so the two arms do not
overlap at all. The prediction failed.

Three explanations were checked.

**Sample size is ruled out.** Fitting `d_ij = mu + a_i + a_j + eps_ij` to the
28 pairs of each eight-run arm and propagating to the mean of three runs gives
`Var = (4/n)sigma_a^2 + sigma_eps^2/C(n,2)`, and the gap sits at z = 5.9 to 7.1
depending on which arm supplies the components. Arm E is also below all 56
three-run subsets of arm B.

**Metric definition is ruled out.** Row-space principal angles are exactly
scale-invariant — scaling one factor by 7.3 leaves the mean cosine at
1.0000000000 — so the update-norm difference cannot produce the subspace gap.
Taking the 16 angles individually, the divergence is concentrated in the leading
directions (−0.0099 at the first, +0.0003 at the sixteenth), so a spectrum-
weighted or top-k variant would widen the gap rather than close it.

**What remains is the probe's measurement scope.** The dose axis is built from
Q/K-only filters: each compresses the head views and passes the body through
untouched. Arms E and B differ only in body encoding, INT4 against E2M1, so
they are scored by *the same filter* and necessarily receive the same two
numbers, cosine 0.9919 and norm ratio 0.9518. Measured separately, every body
role is harmless — cosine above 0.99 for both grids on the residual stream, the
MLP intermediates, the attention output and the value views — and the four
held-out corpora agree, with arm E inside arm B's range on all of them. The
difference appears only in the endpoint geometry, and it comes with a change in
update norm: arm E's adapters end at 24.5 against 31.0 for arm B, the same
level as the uncompressed pair at 24.5.

The failure is therefore not a defect in the probe but a limit on what a
prediction from it can claim. The probe orders configurations by how faithfully
the backward pass survives compression of the head views. It does not determine
the endpoint, because at least one axis it does not measure — the body encoding
— moves the endpoint while leaving both probe numbers and four corpora
unchanged.

**Consequences, applied above.** §2.3's tie rule is withdrawn. §2.1's spacing
requirement rises from the within-arm range to the measured residual, and the
number of usable slots falls from four to two, halving the programme's cost.

## 6. Schedule

1. This document, with the arm E prediction, committed. ← the commit this
   paragraph is in
2. Arm E subspace computed and reported.
3. Probe sweep, after MMLU releases the GPU: four conditions plus the depth curve.
4. §2's empty cells filled, committed and timestamped. **Training starts only
   after that commit.**
5. Sixteen runs. All reported.

## Changelog

- **v1 — commit `2479f1a`, 2026-09-08 09:41:58 UTC.** Registered the design as
  first written (four new arms at targets ≈0.25 / 0.48 / 0.75 / 0.95, spacing
  derived from the within-arm range of 0.002, and a tie rule for arms of equal
  cosine) together with the arm E prediction of §5: mean within-arm row-space
  cosine inside arm B's range, 0.13297–0.13495. Arm E's subspace dispersion had
  not been computed at that point. Its first computation is
  `results/_wd/pa_all.npy`, written 2026-09-08 09:42:31 UTC by
  `scripts/pa_all_arms.py` (itself written 09:42:16 UTC); both files were
  untracked until 2026-09-08, so the timing rests on local file times, not on
  git history. The earlier matrices of 08:19 UTC cover arms A and B only.
- **v2 — commit `511dd3a`, 2026-09-08 11:48:37 UTC.** Recorded the arm E outcome
  in §5 (0.13137, outside the registered range; the registered prediction text
  itself is unchanged). Two arms with identical probe cosine turned out to
  differ by 0.00256, so §2.1's spacing rule was re-based from the within-arm
  range (0.002) to that residual (three times 0.0026 = 0.0077), which closes the
  0.587–0.910 and 0.910–0.992 windows and leaves two slots instead of four. The
  §2.3 tie rule (equal cosine → registered tie, predicted not to resolve) was
  withdrawn for the reason stated there: arm E falsified the premise that
  equal probe cosine implies equal outcome. No new arm had been trained.
- **v3 — this commit, 2026-09-08.** Carries the two-slot design through the
  rest of the document: the slot table (two rows), §2.2 (two interventions
  rather than four), §3 (six arms in total; a perfect ordering gives
  p = 1/6! = 1.4×10⁻³, not 1/8!), and the §4 budget (two arms × four seeds =
  8 runs, ≈ 34 GPU-h). §5 and the registered prediction are unchanged. None of
  the new arms has been trained; this is a design change made before the
  programme starts, and the programme will be pushed before training begins so
  that its registration carries an external timestamp.
- **Pushed to the private remote `github.com/yedammkimm/OAMP`, branch
  `downstream-mc-prereg`, 2026-09-09 08:35:27 UTC** (local head `6f40e83`;
  v1–v3 above and every result up to arm A's MMLU cells were on the remote
  from that moment). The paper's footnote keeps its placeholder until the
  repository is public.
