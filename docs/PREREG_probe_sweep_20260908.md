# Pre-registration — probe sweep, 2026-09-08

Written before any of the four conditions was run. The probe measures
`cos(g, g*)`, the cosine between the gradient under a compression filter and
the gradient with nothing compressed, on one forward–backward pass of
Llama-3.2-3B. Existing reference points: `qk_only` = 0.358, `v_only` and
`o_only` near 1, arm-level values 0.371 (D), 0.587 (A), 0.910 (F), 0.992 (B).

Every prediction below is recorded with the outcome that would weaken it, and
with what the result licenses. Where a prediction is directional the direction
is fixed here, before the run.

---

## 1. Q alone versus K alone

Grouped-query attention gives Llama-3.2-3B 24 query heads and 8 key/value
heads, so each key head is shared by three query heads.

- Compressing **Q** perturbs three times as many elements, but the
  perturbations are independent across query heads.
- Compressing **K** perturbs a third as many elements, but each error is
  **shared** by the three query heads that read that key head, so the errors
  are correlated across the score terms they enter.

The number of affected score entries is 24·L·L either way. Correlated error
does not average out over heads; independent error partly does.

**Prediction: `k_only` gives a lower cosine than `q_only`.**

| outcome | reading |
|---|---|
| `k_only` < `q_only`, clearly | the predicate refines from "on the softmax bilinear input path" to "on that path, weighted by GQA sharing"; the rank-4 rule could then be applied asymmetrically |
| the two within noise of each other | sharing structure does not matter; the predicate stays as published, and Q/K are symmetric passengers |
| `q_only` < `k_only` | the element count dominates the sharing; prediction wrong, and the asymmetric-rule idea is dead |

Noise scale: the probe reproduces a cosine to about ±0.01 across repeated runs
(the layer-0 spread of 0.31–0.42 across two probes eight minutes apart is the
worst case observed, and aggregate values are far more stable). A difference
below 0.02 is not to be read as a difference.

## 2. Additivity of the Q and K perturbations

Treating the gradient error as orthogonal to `g*` makes
`cos = 1/sqrt(1 + eps^2)` exact, so

```
eps^2 = 1/cos^2 - 1
```

is a variance-like quantity that can be compared additively.

- **Additive**, `eps^2_qk ~= eps^2_q + eps^2_k`: Q and K are independent
  passengers and the pair is just the sum of its parts.
- **Super-additive**, `eps^2_qk >> eps^2_q + eps^2_k`: perturbing both at once
  produces error through the cross term of `QK^T`, which is what the
  paper's mechanism claims.

The published `cos(qk_only) = 0.358` gives `eps^2_qk = 6.80`.

**Prediction: super-additive.** The mechanism in §5.1 says the softmax input is
a product of Q and K and is then exponentiated, so a joint perturbation has a
cross term that neither single perturbation has.

| outcome | reading |
|---|---|
| ratio `eps^2_qk / (eps^2_q + eps^2_k)` clearly above 1 | direct confirmation of the product mechanism, one level below the existing V-versus-Q/K contrast |
| ratio near 1 | Q and K are independent contributors; the rank-4 rule is still justified by magnitude but the product story is not supported |
| ratio below 1 | sub-additive, meaning the two perturbations partly cancel; this would need explaining before anything is claimed |

This test replaces the weaker monotonicity check (that `q_only` and `k_only`
should each sit **above** `qk_only`, since each compresses strictly less).
Monotonicity is still required as a sanity condition: a violation means tag
leakage, not a finding.

## 3. Layer-depth threshold

Implemented as `protect_layers < k`: layers below `k` are left uncompressed and
every layer from `k` upward has its Q/K head views compressed. `k = 1` is the
"protect layer 0 only" configuration, at 1/28 of the bit cost of protecting
all layers.

**Prediction: cosine rises steeply from `k = 0` to `k = 1` and then flattens.**
Layer 0's cosine sits 0.31–0.42 below the deeper layers under blockwise
four-bit, so most of the recoverable error should be at `k = 1`.

| outcome | reading |
|---|---|
| steep rise at `k=1`, flat after | protecting layer 0 alone is nearly as good as protecting everything, at 1/28 the cost — a configuration worth running as a training arm |
| gradual rise across `k` | depth is a continuum, no cheap configuration, and the depth curve becomes the result instead |
| no rise | layer 0's low cosine is a property of measuring that layer, not of compressing it |

### Precondition, not an output: tag coverage by layer

This sweep is only interpretable if the fraction of rank-4 Q/K tensors that
actually receive a tag is **uniform across layers**. The existing tagging
misses 4–7% of query head views, which is conservative for the paper's
published claim (an untagged tensor escapes compression, so the reported
cosine is an upper bound on the damage). In the depth sweep the same miss
works the other way: if layers 3 and 17 escape compression while running
"compress layers >= 1", the cosine rises and reads as "layer 0 was enough".
The bias runs toward the prediction.

So the coverage table — tagged rank-4 tensors per layer, separately for Q and
K — is computed and checked first. GQA makes the Q and K paths different, so
they can diverge. If coverage is not uniform, the depth curve is either
corrected for it or abandoned; it is not reported as measured.

## 4. Group size

`_GROUP_SIZE` was a module constant and is now a runtime knob, default
unchanged at 128. Scale overhead is `16/g` bits per element: 0.500 at g=32,
0.125 at g=128, 0.0625 at g=256.

**Prediction: cosine falls as `g` rises**, since a coarser block shares one
scale across more elements with a wider dynamic range.

**Constraint on how this may be used.** The bits-fixed contrast — that gradient
cosine orders the arms while bits per element does not — rests on arms D, A and
F spanning only 0.02 bits (4.105 to 4.125). A group-size sweep moves bits by up
to 0.44, twenty times that span. Group-size arms therefore must not enter that
contrast: it stays computed on the D/A/F subset, and any figure that plots both
must say so. Adding them silently would break the premise that bits are held
fixed.

`_GACT_GROUP_SIZE = 256` is a different quantity — GACT's flat reshape crosses
head boundaries and uses its own block size — and is deliberately not tied to
`_GROUP_SIZE`.

---

## Verification required before any of the four is believed

1. `--group_size 128` explicitly set must reproduce an unset run to the
   printed precision. The default was promoted from a constant; this shows the
   promotion is inert.
2. The `compressed` counts of `q_only` and `k_only` must sum to the count of
   `qk_only`. Splitting a role must not lose or double-count tensors.
3. `q_only` and `k_only` must each give a cosine **above** `qk_only`, since
   each compresses a strict subset. A violation is tag leakage.
4. With the layer threshold at `k = 0` (compress every layer), the result must
   equal the run with no layer condition at all, to the printed precision.
5. The per-layer, per-role tag coverage table must be uniform before the depth
   curve is read.
