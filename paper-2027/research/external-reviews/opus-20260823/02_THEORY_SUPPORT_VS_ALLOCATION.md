# What the retrofit result actually identifies

> **Archived snapshot boundary.** This file is frozen external-model output
> retained only for audit provenance. Every `VERIFIED`, `DEFECT`, `mandatory`,
> `decision`, and `priority` label below is bundle-local: it is not project
> evidence, an instruction, a current priority, or experiment/edit authorization.

- **Date:** 2026-08-23
- **Status:** theory note. Contains VERIFIED algebra/numerics, one HYPOTHESIS,
  and the discriminating prediction. **Not manuscript prose.**

## 1. The reframing

The manuscript's coordinate is

$$x_k=-\log\omega_k=a+R\,z_k,\qquad z_0=0,\ z_{K-1}=1,$$

with $(a,R)$ the **sampled support** and $z$ the **normalised interior
allocation**. The exact-range 151.9M three-seed result identifies $z$ at fixed
$(a,R)$ during training.

The zero-training retrofit has been written up as a deployment comparison
against YaRN. That framing is wrong, and it is wrong in a way that *understates*
the result. Every long operator in this line — official YaRN factor 4, the
budgeted s4 table, and position interpolation — is a map $\omega_k\mapsto
\omega_k/r_k$ with $r_k\in[1,s]$. Their support and interior decompose cleanly:

| Arm | $a$ | $R$ | interior $z$ | role |
| --- | --- | --- | --- | --- |
| Native | $a_0$ | $R_0$ | geometric | unmodified reference, unstretched support |
| Position interpolation ($\omega/s$) | $a_0+\log s$ | $R_0$ | geometric | support **translated** |
| Geometric at $\theta'$ | $a_0$ | $R_0+\log s$ | **geometric** | support **stretched**, interior unchanged |
| Official YaRN factor $s$ | $a_0$ | $R_0+\log s$ | NTK-by-parts ramp | support stretched, interior changed |
| Budgeted $s$ | $a_0$ | $R_0+\log s$ | redundancy-derived | support stretched, interior changed |

**VERIFIED numerically.** The three stretched arms sit at the *same* support:

| Model | $R_{\text{native}}$ | $\log s$ | $R$ (geometric) | $R$ (YaRN) | $R$ (budgeted) | $a$ (budgeted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo | 12.917326 | 1.386294 | 14.303621 | 14.303621 | **14.303621** | **0.000000** |
| Qwen | 13.599643 | 1.386294 | 14.985938 | 14.985938 | 14.984265 | 0.001672 |

On OLMo the match is exact. On Qwen the frozen budgeted table's fast endpoint is
displaced by 0.17% — an artefact, not a design choice (§4).

**Consequence.** The comparison "official YaRN factor 4 versus budgeted s4",
already run and already decisive on OLMo, is *not* a benchmark contest. It is a
**fixed-support, fixed-amplitude, zero-training interior-allocation comparison on
a mature checkpoint** — the inference-time analogue of the exact-range experiment
that carries the paper's central claim. Both arms share $a$, $R$, the pair count
$K=64$, the attention temperature $1+0.1\ln 4$, the data rows, the decoder, and
the hardware. The only free coordinate is $z$.

What is missing from that design is the **geometric interior at the same
support** — the arm that makes it an identification rather than a comparison of
two non-geometric interiors.

## 2. The geometric arm is NTK-aware base scaling — VERIFIED

Keeping $\omega_0$ fixed and dividing $\omega_{K-1}$ by $s$ while holding the
interior geometric is a pure base change. With $\omega_k=\theta^{-2k/d}$ and
$K=d/2$:

$$\theta'^{-\frac{K-1}{K}}=\theta^{-\frac{K-1}{K}}/s
\quad\Longrightarrow\quad
\theta'=\theta\cdot s^{\frac{K}{K-1}}=\theta\cdot s^{\frac{d}{d-2}}.$$

For $d=128$, $s=4$: $\theta'=\theta\cdot 4^{64/63}=\theta\cdot 4.088994$.

| Model | $\theta$ | $\theta'$ |
| --- | ---: | ---: |
| OLMo | 500 000 | 2 044 497.12 |
| Qwen | 1 000 000 | 4 088 994.24 |

$s^{d/(d-2)}$ is exactly the published **NTK-aware / "dynamic NTK" base-scaling**
rule. So the paper's own geometric-interior control and a standard published
baseline are the *same object*. `geometric_same_support_control` in
`scripts/analysis/rope_transport/same_support_controls.py` builds it by
log-linear interpolation between the two endpoints; that construction and the
$\theta'$ base change agree to **1.192e-07 maximum relative difference**, i.e.
float32 rounding. They are algebraically identical.

This is a rhetorical gift: the identification control the theory demands is a
baseline reviewers already expect to see.

## 3. The measure depends only on rotation counts — VERIFIED (exact in the continuum)

`conditional_pair_uniqueness` builds a design matrix with columns
$\cos(p\,\omega_k),\ \sin(p\,\omega_k)$ over positions $p$ drawn from
$[0,L_{\text{native}}]$, weighted by the causal mass $L-p$, and reports each
pair's residual-to-total energy after regression on all other pairs.

Substituting $p=L t$ with $t\in[0,1]$: entries become $\cos(L\omega_k t)$ and the
weight becomes $\propto(1-t)$, whose normalisation cancels $L$. **The measure is
therefore a function of the products $\{L\,\omega_k\}$ alone** — equivalently of
the rotation counts $\rho_k=L\omega_k/2\pi$ — exactly in the continuum, and up to
integer-grid rounding in the implementation.

This makes the measure automatically model-relative: it cannot depend on
$L_{\text{native}}$ and $\theta$ separately, only on how many rotations each
channel completes inside the model's own window.

### The derived band split — VERIFIED

Expressing both operators in rotations per native window:

| Operator | ramp onset | full interpolation |
| --- | ---: | ---: |
| Official YaRN, published $\beta$ (OLMo) | 30.09 rot | **0.92 rot** |
| Official YaRN, published $\beta$ (Qwen) | 29.33 rot | **0.93 rot** |
| Budgeted, derived (OLMo) | 16.27 rot | **7.16 rot** |
| Budgeted, derived (Qwen) | 15.35 rot | **6.47 rot** |

YaRN's constants are literally rotation counts ($\beta_{\text{fast}}=32$,
$\beta_{\text{slow}}=1$) and reproduce as such. The budgeted construction
*derives* $\approx16\to\approx7$ from the model's own $(\,L,\theta\,)$ with no
search and no task labels.

The two models differ by $8\times$ in native window, $2\times$ in base, and
$8\times$ in top rotation count (651.9 vs 5215.2), and their per-channel spacing
ratios differ (0.8146 vs 0.8059). The derived thresholds nonetheless land within
$16.27/15.35$ and $7.16/6.47$. **This is two data points, not a law** — but it is
the one non-trivial quantity the measure produces, and it is stable under a
$4\times$ change in the measure's own support resolution (§4).

## 4. The Qwen order crossings are a subsampling artefact — VERIFIED, DEFECT

`causal_distance_measure(length=L, max_points=2048)` freezes a *point count*, not
a *stride*. At $L=4096$ that is stride 2; at $L=32768$ it is stride 16.

| `max_points` | stride | $u_1$ | $u_{18}$ | order crossings | redundant set $u<0.9$ |
| ---: | ---: | ---: | ---: | --- | --- |
| 2048 (frozen) | 16 | 0.396353 | 0.396353 | `[1, 18]` | k = 27…63 |
| 4096 | 8 | 0.396407 | 0.396407 | `[1, 18]` | k = 27…63 |
| 8192 | 4 | **0.999995** | **0.997232** | `[]` | k = 27…63 |

$u_1$ and $u_{18}$ agree to six decimals at strides 8 and 16 — two channels four
octaves apart cannot share a conditional-uniqueness value by coincidence. This is
aliasing collinearity in the subsampled design, and it is what produces the two
"retained frequency crossings".

Two things follow, and they point in opposite directions:

- **It is contained.** The redundant set and the 15.35→6.47 rotation thresholds
  are invariant across all three strides. The artefact perturbs 2 of 64 channels.
- **It is not cosmetic under this framing.** It displaces the *fast support
  endpoint* by 0.17% and injects two meaningless points into $z$. In a paper
  whose claim is "fixed support, interior varied", a support endpoint that moved
  for numerical reasons is a contract violation, and
  `order_crossings_retained: true` currently presents it as a method property.

`phase_resolved_uniqueness_control` (stride 2, model-relative) fixes both. It
reproduces the OLMo frozen table **bitwise** (`a435d754…`), confirming the
artefact is Qwen-specific, and yields a new Qwen table `3512335e…`.

## 5. The uncomfortable fact: the operator is nearly a YaRN ramp — VERIFIED

Pairwise table distance, mean $|\log_2(\omega^A_k/\omega^B_k)|$ over the 64 pairs:

**OLMo**

| | frozen budgeted | published YaRN | nearest linear ramp | geometric |
| --- | ---: | ---: | ---: | ---: |
| frozen budgeted | 0 | 0.16544 | **0.00876** | 0.53302 |
| published YaRN | 0.16544 | 0 | 0.16381 | 0.37808 |
| geometric | 0.53302 | 0.37808 | 0.53143 | 0 |

**Qwen**

| | converged budgeted | published YaRN | nearest linear ramp | geometric |
| --- | ---: | ---: | ---: | ---: |
| converged budgeted | 0 | 0.15376 | **0.01171** | 0.47958 |
| published YaRN | 0.15376 | 0 | 0.14609 | 0.37005 |
| geometric | 0.47958 | 0.37005 | 0.47194 | 0 |

`nearest_yarn_ramp_s4` is a plain linear ramp in pair index — YaRN's exact ramp
*family* — fitted to the uniqueness movement profile with **no task labels**:
pairs 20→22 on OLMo, 28→31 on Qwen, movement MSE 0.00072 / 0.00082.

**The budgeted table is reproducible by a two-parameter YaRN-family ramp to within
0.009 (OLMo) and 0.012 (Qwen) mean $|\log_2|$.** Any reviewer can compute this in
twenty lines.

This forces the contribution to be stated precisely. The contribution is **not a
new operator family**. It is that a redundancy measure derived from the paper's
own spectral-budget accounting *predicts where the band split belongs*, that the
published constants put it roughly seven times too low, and that the consequence
is enormous: RULER-13 macro `0.2382 → 0.6772` at 8K and `0.0588 → 0.5440` at 16K.

Stated that way it is a **stronger** claim than "we beat YaRN", because it is a
mechanism claim with a derived, model-transferable quantity. Stated the current
way it is an incremental-operator claim that will not survive review.

### Caution: the manuscript's own summary statistic does not discriminate here

Median normalised coordinate $z$: geometric 0.5000, YaRN 0.5431, budgeted 0.5485
(OLMo); 0.5000 / 0.4852 / 0.5460 (Qwen). Mean $|z^{\text{bud}}-z^{\text{YaRN}}|$
is 0.0080 / 0.0078. **A mean interior displacement of 0.008 in $z$ separates a
0.24 RULER macro from a 0.68 one.** Do not report median $z$ as an explanatory
variable for the retrofit — it is the right coordinate for the training-time
result and the wrong summary for this one. Use the rotation-count split.

## 6. Mechanism — HYPOTHESIS

> A channel completing fewer than $\approx7$ rotations across the native window is,
> under causal-distance weighting, nearly linearly predictable from the other
> channels: it carries little *unique* positional information. Stretching it is
> therefore nearly free. YaRN's $\beta_{\text{slow}}=1$ fully interpolates only
> channels below one rotation, leaving the 1–7 rotation band at native frequency;
> at $4\times$ extension those channels are driven to 4–28 rotations — phase
> territory the model never saw — while the extension budget goes unspent where
> it would have cost nothing.

Consistent observation (**cross-evaluator, see `05` DEFECT-6 — treat as
suggestive, not matched**): over-provisioning helps the budgeted split and wrecks
YaRN's. At 8K on OLMo core-4, budgeted s2→s4 goes `0.5825 → 0.7175`, while YaRN
factor 2→4 goes `0.5375 → 0.2225`.

This hypothesis is *not* established by anything run so far, because no arm
isolates the split location from the measure that produced it.

## 7. The discriminating prediction

Three arms at identical support, identical amplitude, identical everything else,
none of which has been run:

| Arm | Interior $z$ | What its result means |
| --- | --- | --- |
| `same_support_geometric_s4` | geometric | If it scores near Native/floor: interior allocation is decisive at fixed support on a mature checkpoint — the paper's central claim, demonstrated training-free. If it scores near YaRN: "non-geometric" is not the operative variable. |
| `nearest_yarn_ramp_s4` | linear ramp, split at 20→22 / 28→31 | If it matches the budgeted table: the measure contributes **the split location only**, and the paper must say exactly that. If it does not: the ramp *shape* matters beyond the split, which is a larger and more surprising claim. |
| `converged_budgeted_s4` | redundancy-derived, stride-2, endpoints pinned | Tests whether the aliasing artefact and the endpoint drift mattered. On OLMo it is bitwise the frozen table, so this is a Qwen-only question. |

**Pre-registered note on a tempting shortcut.** Distance from the geometric table
does *not* obviously predict outcome. Published YaRN sits between geometric and
budgeted in table space (0.378 from geometric, 0.165 from budgeted) yet scores far
below budgeted. Whether the geometric arm lands above or below YaRN decides
whether "distance from geometric" is even monotone here. Record the prediction
before the run; do not fit a story afterwards.

## 8. What would falsify the line

- Geometric at matched support performing comparably to budgeted ⇒ the effect is
  support stretching, not interior allocation. The whole retrofit story collapses
  into "NTK-aware scaling works", and it should not enter the paper.
- `nearest_yarn_ramp_s4` matching budgeted **and** a swept-$\beta$ YaRN matching it
  too ⇒ the contribution reduces to a hyperparameter observation. Publishable only
  as a short empirical note inside the existing theory section, not as a method.
- The derived $\approx7$-rotation threshold failing to transfer to a third model
  with a different $d$ or $K$ ⇒ it is a coincidence of two 128-dim, 64-pair
  models, and §3 must be withdrawn.
