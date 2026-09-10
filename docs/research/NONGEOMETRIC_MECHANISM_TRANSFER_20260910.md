# Nongeometric allocation: mechanism and transfer investigation

2026-09-10. Active experiment record; the ten-direction screen remains in progress.
The objective is to outperform MrRoPE-Pro with a rule that transfers across tasks
and checkpoints. A selected frequency table and two development wins do not yet
establish that rule. The original [ten-candidate plan](NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md)
defines the fixed-weight experiment and permissive mixed-RULER screening protocol.

**Latest protocol correction:** the user's subsequent instruction supersedes
the old automatic full-panel completion rule. The active objective is long
context: first inspect target-length PPL and passkey, then selectively release
long downstream tests. No new 32K experiments or automatic 36-row completion
are queued. P2 and same-gain MrPro already scored 4/4 on the existing 128K
single-key passkeys; these are reused. The only newly queued job measures their
tail-512 PPL on four existing contiguous 128K document prefixes, with official
MrPro's cached references reused. The large P2/E1/Slower holdout jobs and
`finish_screen` have been deferred, pending this cheaper evidence and a revised
long-only scope. This also supersedes the immediate-expansion priority stated
in the preceding candidate-quality decision below.

**Current priority correction (2026-09-10, after the user's candidate-quality
critique):** the hand-built HighGap, G1/G2/G3 pair-gap, and BM_ScaleTaper branch
has been withdrawn from the active queue. The associated G binding confirmation
is also deferred. HighGapToLong's completed negative result remains below;
HighGapToMid stopped between rows at 16/36, and its partial data are retained.
Unrun proposals are not labeled empirical failures. Later historical descriptions
that say these jobs are queued or pending are superseded by this decision.
The immediate GPU priority is independent confirmation of full P2's measured
long VT/QA gains, followed by the original screen and measured-positive work.
The mistake was proposing arbitrary local redistributions from a verbal
analogy without a sufficient reason to expect improvement over existing effective
allocations. Further candidate generation should start from those effective
rules and identify a concrete mechanism for improvement. It need not prove
universal task superiority, but preserving endpoints or moving a nominal budget
alone does not supply that rationale. Queue contracts and the decision are
retained in `results/nongeometric_screen_20260909/deferred_queue/20260910_candidate_quality/`.

The 128K P2 precheck subsequently completed in 268.7 seconds. Over the same
four document prefixes and tail-512 targets, mean NLL / pooled PPL is
1.705390 / 5.503529 for official MrPro, 1.680924 / 5.370516 for same-gain
MrPro, and 1.678507 / 5.357549 for P2. P2's PPL changes are -2.6525% versus
official MrPro and -0.2415% versus same-gain MrPro; the latter consists of
two improved and two worsened documents. All paired input hashes and target
token arrays match. Together with the reused 4/4 128K passkeys, this permits
one small fresh QA-only pilot: four 128K rows evaluated with all three methods
(12 generations). It does not release the deferred 16-row expansions or any
32K work. The small matched PPL improvement is a screening result, not proof
of the earlier developmental QA gain.

## What needs explanation

On the pinned Qwen2.5-3B-Instruct checkpoint, changing only zero-based slot 28
from the MrPro exponent to its predecessor exponent gives +5.2083 percentage
points at 128K and a tie at 32K on the historical 36-row six-task panel. The two
improvements are multikey row 2 and multiquery row 3; no other row loses score.
This is development evidence. The 16-document 8/16/32K NLL differences are
+0.0001544, -0.0006042, and -0.0008822 nats/token.

Changing slot 29 to its successor exponent gives +8.3333 points at 32K and
-0.2083 points at 128K. The short gain is one QA answer; long VT improves by
one returned variable while another multiquery answer loses one value.

The two changes weaken compression in an earlier transition slot and strengthen
it in the next slot. This qualitatively resembles the earlier part/later part
contrast in the successful FullLagP2 deployment. It does not prove that a sharper
transition is the cause: the P2 table changes many slots and its original gain
differs. The queued two-slot combination tests the actual joint effect; equal
absolute-frequency-step reversal and adjacent-slot controls test specificity.

## Mathematical facts and the missing implication

For fixed raw Q/K in one head, write

\[
z_j(\nu)=\frac{g^2}{\sqrt{d_h}}\sum_k
 [A_{jk}\cos(\nu_k d_j)+B_{jk}\sin(\nu_k d_j)].
\]

For a target key against all competitors, define
\(M=z_* - \log\sum_{j\ne *}\exp z_j\). Its conditional derivative is

\[
\partial_{\nu_k}M=\partial_{\nu_k}z_*
-\sum_{j\ne *}\pi_j\partial_{\nu_k}z_j,\qquad
\partial_{\nu_k}z_j=\frac{g^2d_j}{\sqrt{d_h}}
[-A_{jk}\sin(\nu_kd_j)+B_{jk}\cos(\nu_kd_j)].
\]

The signs depend on content coefficients and competitors. Increasing frequency
does not universally increase useful separation. Even the content-free
two-position distance \(4\sin^2(\nu d/2)\) is nonmonotone in frequency once
phases leave a small neighborhood. Changing gain changes logit magnitude, not
the phase arc. A phase-only explanation cannot absorb gain results as independent
confirmation of the same mechanism.

Moreover, target attention mass is not the task objective. The values, output
projection, residual stream, and later layers determine whether the attended
information supports the answer. Changing a table during prefill also changes
the later raw Q/K/V. Our conditional selected-key replay holds these states and
omitted-key contributions fixed. Its exact finite trigonometry does not make
it an exact end-to-end predictor. E8's favorable proxy but worse generation is
already a counterexample to treating that proxy as sufficient.

The usable theoretical object is a **signed decision margin under the actual
model computation**, with separate finite contributions from prefix formation
and readout. Geometry constrains possible effects but does not supply their sign.
The following intervention measures those contributions on observed positive
cases before proposing a universal explanation.

## Positive-case interventions

Rebuild all prefix keys from captured pre-RoPE K; never invert rounded cached K.
Keep each source prefix's values and content keys. Cross the prefix-formation
table with the final query/decoding table. Each source's full generation must
match its archived output. Compare crossed generation to the same cached-query
path, and separately report whether that path preserves the original phenomenon.

| Case / method | Mr prefix, Mr read | Mr prefix, E1 read | E1 prefix, Mr read | E1 prefix, E1 read |
|---|---:|---:|---:|---:|
| MK2 128K row 2 / slot 28 | 0 | 0 | 0 | 1 |
| Multiquery 128K row 3 / slot 28 | .75 | 1 | .75 | 1 |
| VT 128K row 2 / slot 29 | .8 | 1 | 1 | 1 |
| QA 32K row 1 / slot 29, cached path | 0 | 0 | 0 | 0 |

The QA same-table cached path loses the full-prefill E1 success. It cannot be
used to explain that success. Its full archived output was reproduced before
the intervention; the difference arises when the last query is partitioned.

The MK binary table alone might suggest a strong interaction, but its existing
token trace permits a more precise explanation without another GPU run. At the
first differing digit, all four arms have generated the same preceding space.
The correct digit is `6` (token 21), and the competing digit is `9` (token 24).
Both appear in every top-five trace. Their correct-minus-incorrect logit margins
in the four column orders above are **-2.125, -1.125, -1.000, +0.250**.

Thus readout alone improves the margin by 1.000 nat and prefix formation alone
by 1.125 nats. Together they cross the greedy decision boundary; the continuous
interaction remainder is only +0.250 nat, at the resolution of BF16 logits.
It is incorrect to infer a large nonadditive internal mechanism from the binary
success pattern alone. A transferable method should improve this signed margin
on new inputs, rather than merely fit the two observed decision crossings.

Multiquery is readout-sensitive under this intervention. VT can be repaired by
either component on this case. These are distinct measured pathways, not proof
that all tasks share one cause. Prefix interventions jointly change K/V content;
they do not isolate keys from values or identify a specific layer.

## Numerical approximation audit

E7's local predicted output NMSE is 7.93668e-8. On the same support and fixed
baseline states, exact finite ideal-relative computation gives 7.93640e-8;
FP32 absolute-coordinate computation gives 7.93255e-8. BF16 absolute-coordinate
rotation gives 1.28074e-5. The roughly 160-fold discrepancy here is dominated
by finite-precision rotation, not a measured 160-fold failure of the local
linear approximation. This does not prove that the rest of the full model's
E7 degradation is caused by rounding, or that all first-order methods are valid.

Two proposal records also exposed a separate implementation problem: normalizing
`old_probability * exp(delta)` by the largest delta could underflow every term.
Log-weight normalization repairs the two nonfinite records. The E2 selection
is unchanged. E5's second-ranked layer changes from 27 to 32; layer 27's actual
model outputs remain archived, but its nomination was invalid. Layer 32 is
being measured. These are computational corrections, not scientific failures.

## Frozen transfer and independent tasks

- Qwen2.5-7B: use the existing 18-row, six-task 32K/128K panel and its own
  MrPro baseline, plus four documents from its existing NLL panel. The native
  grid and MrPro bounds match 3B, so the slot-28 table transfers exactly.
  Preserve the baseline's 4096-token positionwise MLP implementation.
- OLMo-2-0425-1B-Instruct: use its existing 36-row six-task panel and its own
  MrPro/NLL references. Freeze the source rule as the fraction 5/17 through the
  MrPro transition, round to the nearest target slot, and replace its exponent
  by its predecessor's. Bounds 14/32 map to slot 19. Preserve target gain and
  every other frequency. No target outcomes select or tune this mapping.
- New 3B tasks: four task types (multikey, multiquery, VT, QA), 16 examples per
  task and length initially, 32K/128K, seed 20260910. The QA first batch samples
  one question from each of 16 articles, excluding development articles.
  Source-selected tables are frozen before evaluation. Relevant positive
  directions can advance to the preprepared second batch of 16 examples.
- New continuous text: PG19 and Proof-Pile arXiv, 64K/128K prefixes, first two
  documents per source measured. Slot 28's differences are -0.00155/-0.00121
  on Proof-Pile and +0.00359/+0.00420 on PG19; slot 29 stays within +0.00304.
  These small samples measure tail-512 NLL, not full-document perplexity or
  established equivalence. Eight documents per source are already prepared.

This separates fixed-rule transfer from checkpoint-specific recalibration.
If fixed transfer fails but recalibration succeeds, the claim is about an
adaptation procedure; it is not a universally better frequency table.

## Receipts and current scope

Remote run: `/root/autodl-tmp/nongeometric_screen_20260909` on the authorized
seetacloud server. Results are in `results/`, `long_nll/`, `causal_cases/`,
`selection/replay_repair.json`, and (as completed) `transfer/` and
`holdout_results/`. Code is in `experiments/nongeometric_screen/`.
No weights, adapters, or optimizer parameters have been trained. Large raw
activation files remain on the server. The parallel 20x10 plan is under a
separate evidence review; its proposed experiments are not automatically added
to the GPU queue.

## Subsequent synthesis with the supplied Pro analyses

The user supplied a GPT-5.6 Pro local-gap analysis and a GPT-6 Pro fixed-budget
analysis. Both retain the signed, value/readout-dependent interpretation and
propose concrete counterfactuals. They are compatible at the level of budget
accounting, but make different predictions about useful allocation shape.

Under the retained identity/high-frequency and `/S`/low-frequency endpoints,
write \(\epsilon_i=m_i-m_{i-1}\), \(\sum_i\epsilon_i=1\). The additional
log-frequency gap is \(\epsilon_i\log S\). The internal cumulative budget is

\[
B=\sum_{q=1}^{N-1}m_q
=\sum_{i=1}^N(N-i)\epsilon_i
=N-\sum_{i=1}^Ni\epsilon_i.
\]

Thus B fixes the centroid of the additional-gap allocation. The conserved
total gap is conditional on those endpoints; uniform PI changes all
frequencies together and does not enlarge the total frequency span.

The GPT-5.6 Pro arms use delta=6/306 around `(m28,m29)=(30,42)/306`:

| Arm | New numerator pair /306 | B | Increment roughness |
|---|---|---:|---:|
| MrPro | (30,42) | 5.333333 | .013071895 |
| G1 gap widen | (24,48) | 5.333333 | .020761246 |
| G2 gap narrow | (36,36) | 5.333333 | .020761246 |
| G3 shift pair slower | (36,48) | 5.372549 | .014609765 |

G1 changes the three neighboring increments by `(-delta,2delta,-delta)`;
G2 reverses it. Both preserve the geometric-mean pair frequency and total B.
Because the baseline increments are locally linear, both increase the
Dirichlet roughness by exactly `20*delta^2`. They also have equal exponent
perturbation norms. Their sign comparison therefore controls these quantities,
but still changes the two exterior gaps along with the middle one. It does
not isolate one central gap while keeping all others fixed. G3 preserves the
central gap and shifts the pair's geometric center, also changing B.

The GPT-6 Pro arm minimizes increment roughness with nonnegative increments,
sum one, and B fixed to MrPro's 16/3. An independent CPU implementation with
full primal/dual KKT checks yields roughness .004886399, active zero increments
1 and 2, equality residual below 9e-16 and free-stationarity residual below
6e-17. At the larger B=8 it recovers BM to numerical precision. These certify
the convex construction, not model performance. `smooth_budget.py` prepares
the table; missing Qwen3B MrUni is also queued to complete the comparison.

All newly added model experiments use Qwen3B following the user's cost
constraint. The earlier Qwen7B and OLMo E1 transfers are complete and are not
extended. The new shape/budget table and the three gap arms reuse the existing
36-row development panel; new binding inputs remain separately identified.

### Completed gain factorial

| Table and gain | 32K | 128K |
|---|---:|---:|
| MrPro, coefficient .1 | 87.2222% | 78.1250% |
| BM, coefficient .1 | 91.6667% | 70.8333% |
| MrPro, coefficient .074 | 98.3333% | 75.3472% |
| BM, coefficient .074 | 100.0000% | 70.0000% |

The large short improvement of BM/gain074 is mostly shared by the MrPro gain
control. The table effect at matched gain074 is +1.6667 pp short and -5.3472 pp
long. Lowering BM's gain all the way to 1 gives 89.5833%/58.8194%; its long
loss versus BM at original gain is 12.0139 pp, while its NLL improves. The total
gap to MrPro is not an isolated gain effect. No broader gain grid is queued.

### Additional diagnostics

The original multikey failure is a genuine competing-record answer:
`toothsome-citrus` has value 6683176, while MrPro outputs 9424151 belonging to
`toothsome-scene`. Swapping the two equal-token-length values preserves the
token multiset and physical positions while changing the answer required by
the same query. After the swap, both methods fail: MrPro emits 6624365 and E1
emits 6624369, neither of which occurs in the prompt. The first-digit contrast
between the original two values is -2.0/+0.25 for MrPro/E1 before the swap and
+2.0/+1.625 afterward. Other answers compete, so this contrast cannot be
treated as a binary choice probability. This single counterfactual does not
establish robust binding recovery or rule out every positional bias.

For this restricted first-digit contrast only, write the exchange-sensitive
component as `(D_plus-D_minus)/2` and the shared component as
`(D_plus+D_minus)/2`. They are (-2.0, 0.0) for MrPro and (-0.6875, 0.9375)
for E1. E1's original +2.25 margin advantage consequently decomposes into
+1.3125 in the exchange-sensitive component and +0.9375 in the shared one.
The former still has the wrong sign in both methods, and the shared component
is not independently identified as a content-free bias. These are algebraic
components of two counterfactual logit contrasts, not percentages of a
validated causal mechanism or evidence that the full answer follows the key.

A global +1 position shift flips this original E1 multikey success to failure,
while the matched MrPro failure stays unchanged. However, E1's improvement in
the correct-versus-competing first-digit margin is +2.25 at origin zero and
+2.0 at origin one. Thus the argmax crossing is numerically fragile while the
directional margin improvement survives this diagnostic. E1's multiquery and
short QA improvements retain their scores under this shift. These are three
retrospective cases, not an estimate of general robustness. All margins above
use full-prefill records; a separate cached-path MrPro margin is not substituted.

The pure native S=1/gain=1 short-generation reference completes 12 rows with
83.3333% macro score versus MrPro's 87.2222%. Native QA scores zero, while VT
and FWE score one; the six-task average is sensitive to the two QA samples.
Existing native NLL is reused. The archived NativeWindowMrPro experiment
changes its final read to MrPro and cannot be mislabeled as this reference.

E10's duplicated Mr/Mr arithmetic control completes the same four-document
8K/32K NLL subset. Its differences from stock MrPro are -0.000709/+0.000030
nats. E10's Mr/BM mixture minus that control is -0.005180/-0.012242, so the
small observed NLL improvement is not explained by this arithmetic control.
The initial 12 RULER scores tie MrPro; the full development panel is pending.

### Waterbed interpretation and the target tradeoff

The original waterbed budget identities constrain the chosen spectral
parameterization or surrogate cost. They do not prove conservation of task
accuracy between 32K and 128K. The current objective prioritizes 128K behavior
subject to retaining useful 32K performance. A short-context gain can motivate
testing a different allocation, but it does not establish that a corresponding
amount of task ability can be transferred to long context. Conversely, MrPro
is not presumed Pareto-optimal for every checkpoint. The separate OLMo result
already permits simultaneous short and long improvements over its MrPro
baseline, without implying the same outcome on Qwen3B.

## Distance dependence and the search for a better allocation rule

The user identified an important error in the recent research emphasis: fixed
budget, smoothness, and neighboring-gap geometry were being analyzed without
making protection and improvement of long-distance interactions the primary
criterion. The original E1 proposal aggregation also weights short and long
task cells equally. These choices are not equivalent to prioritizing 128K
subject to preserving useful short-context behavior.

The overall research objective remains a better allocation rule than MrRoPE,
with explanatory and transferable value. The user explicitly cautions against
turning a local long-distance diagnostic into the entire research program.
High/middle/low frequency roles are a starting point, not an immutable final
partition, a proof that short and long must trade off, or a restriction to two
perturbation directions. A large extrapolation gain is a useful possible first
breakthrough; simultaneous improvements are also admissible. The four-slot
comparison below is one concrete diagnostic of this wider objective.

On the actual MrPro table, zero-based slots 36--39 have periods of 33,983,
47,875, 68,059, and 97,632 tokens. They are inside the nominal middle transition
but already operate on the scales that bridge 32K to 128K. Slot 40 has period
141,332 tokens. Preserving slots 40--63 does not preserve the whole long-distance
part of the spectrum. In the multikey case above, the final prompt query is
88,725 tokens from the first token of the target value. BM changes slot 38's
phase by -1.676 radians at that distance despite preserving the low-frequency
tail. This is a concrete perturbation, not by itself proof that slot 38 caused
the answer failure.

The previous E1 slot-28 change has an unwrapped phase shift of 3.144 radians
at 32K and 12.575 at 128K. Its shift at the actual 88,725-token evidence
distance is 8.512 radians. A near-complete number of turns at the maximum
length does not imply small changes at intervening distances. For a fixed
raw Q/K pair the relevant phase is `distance * frequency`, and the rotation
difference is `2*abs(sin(distance*delta_frequency/2))`; neither exponent-table
smoothness nor a single endpoint phase controls the task effect.

### Distance-scaled direction comparison

`long_bridge.py` selects MrPro frequencies whose periods lie between 32K and
128K (slots 36--39 here). It adds or subtracts exactly `1/131072` radians per
token on those slots, retaining every other frequency and the attention gain.
Thus the two directions have matched fixed-state phase magnitudes: one radian
at 128K and one quarter radian at 32K. All frequencies remain positive and
ordered after FP32 conversion. Slower periods become 35,446/50,830/74,190/
110,764 tokens; faster periods become 32,637/45,245/62,864/87,285 tokens.

One radian is an explicit exploratory scale, not a derived optimum. This
comparison asks whether shifting the long transition toward slower clocks
helps or instead loses useful discrimination. It does not assume that slower
is universally better, that the budget remains fixed, or that a fixed-state
phase bound controls the whole network. Both arms are queued on Qwen3B for
the same 36-row development panel and 16-document NLL controls. Fresh-input
confirmation is separate; no holdout outcomes were used to construct them.

Slower now completes all 36 generation rows and 48 NLL measurements. Its
32K/128K macro scores are **80.5556%/80.0694%**, differences of
**-6.6667/+1.9444 pp** from MrPro. At 128K, VT improves from .75 to .95;
FWE declines from .75 to .6667; multikey remains .75 through one lost and
one recovered answer; single-key, multiquery, and QA retain their averages.
At 32K, one multikey answer is lost while VT rises .9 to 1. NLL differences
at 8K/16K/32K are +.001088/-.000231/-.000228 nats. There are three improved
and three worsened rows overall. This is a conditional long-development
signal with a short-context cost, not a large or general improvement.
Faster also completes all 36 rows and 48 NLL measurements. Its 32K scores
all tie MrPro (macro 87.2222%). At 128K it scores 73.9583%, or -4.1667 pp:
the sole changed score is QA row 2, from 1 to 0. All other 35 scores tie,
giving zero wins and one loss. NLL differences at 8K/16K/32K are
+.000267/-.000708/-.000889 nats. The equal-magnitude opposite direction
therefore does not reproduce Slower's positive long development signal.
These are still small-panel task outcomes, not an optimal-direction theorem.
Fresh-input confirmation remains pending.

The +20 pp development VT signal qualifies for the declared focused
confirmation even with the short-context cost. Two queued jobs freeze the
Slower table on 16 new 128K VT rows and 16 new multikey rows at each of 32K
and 128K: 48 candidate generations in total. Their MrPro references reuse
the preceding E1 confirmation cohort. VT is the primary benefit test;
multikey checks the prominent short cost and the long binding tradeoff.
This is the same declared source/template family with a new seed. The
development FWE cost remains disclosed and is not covered by this focused
confirmation. These jobs do not delay delivery of the Pro theory brief.

### Completed fixed-budget smooth result

Smooth(MrBudget) finishes at 87.2222%/68.3333% for 32K/128K, versus MrPro's
87.2222%/78.1250%. The long difference is -9.7917 pp. At 128K, multikey drops
from .75 to .25 and QA from .5 to .25; multiquery rises .9375 to 1 and VT
.75 to .85. NLL differences at 8K/16K/32K are +.000975/+.001925/+.003423.
This table is not a long-context improvement on Qwen3B. The result rejects
minimum roughness at fixed MrPro budget as a sufficient prescription here;
it does not identify one frequency band as the cause or rule out other
distance-directed allocations.

### What the paired long-band shift identifies

For fixed raw Q/K, write the contribution of the selected band as
`A(d) = sum_j [a_j cos(d nu_j) + b_j sin(d nu_j)]`, with the sine sign
absorbed into `b_j`. Define its quadrature
`B(d) = sum_j [-a_j sin(d nu_j) + b_j cos(d nu_j)]`.
Adding the same absolute frequency `delta` to every selected slot gives the
exact identity `A_delta(d) = cos(d delta) A(d) + sin(d delta) B(d)`.
The opposite direction reverses the sine term. This follows from angle
addition; a CPU numerical check at six relevant distances and both signs
agrees within 8.9e-16 in float64.

Consequently, LongBridge changes the band's common phase relative to the
content and the remaining slots, while preserving all within-band absolute
frequency differences. It does not test a redistribution of clock spacing
inside that band. A gain would motivate testing whether a broader rule can
preserve that useful phase alignment across distances; a loss would leave
within-band redistribution, transition-boundary changes, and the already
queued full P2 rule untested by this particular diagnostic. No end-to-end
ranking follows from this fixed-state identity, since prefill also changes
the coefficients. The equal and opposite frequency steps therefore supply
a concrete direction comparison without exhausting the allocation problem.

The calibration position records also distinguish physical relation length
from input length. For the first prompt-end query of the two 128K multikey
rows, correct-answer digit tokens lie 117,487--117,493 and 95,676--95,682
tokens away. The single-key rows instead place them 51,403--51,409 and
29,024--29,030 tokens away. Multiquery row zero mixes distances from 70 to
120,933. These are literal target-token geometry, not measured internal read
paths. Even a nearby target can remain difficult in a long input because of
intervening computation and distractors. A better allocation rule should
therefore address both far relations and competition within long inputs;
calling every 128K score a direct test of a 128K relation would obscure this
distinction.

### One further whole-table rule: BM with a physical-scale taper

While the paired direction test runs, a further candidate directly combines
the measured useful BM allocation with retention of MrPro's long-period
clocks. Let `T_j = 2*pi/nu_Mr_j` and
`w_j = clip(log(W/T_j)/log(S), 0, 1)`. Set
`nu_j = nu_Mr_j * (nu_BM_j/nu_Mr_j)**w_j`, using the official MrPro gain.
This uses BM at periods up to `W/S`, blends back in log-frequency over
`[W/S,W]`, and keeps MrPro at periods at least `W`. The physical interval is
an explicit design hypothesis using existing W and S, not a derived optimum.
It is not selected using fresh-input results.

For the current model, slots 24--31 equal BM, 32--35 taper with weights
.8972/.6762/.4486/.2144, and 36--63 equal MrPro. The unchanged high-frequency
slots equal both references. The resulting table is positive and ordered.
The question is whether some measured BM benefit can survive while reducing
its changes to the slow middle band. Retaining that band's frequencies is
not a guarantee of long-context preservation: changed faster frequencies
still act at long distances and alter upstream states.

`scale_taper.py` queues **BM_ScaleTaper**, one full 36-row development panel
and the existing 16-document 8K/16K/32K NLL controls. Both MrPro and BM at
gain .1 already exist and are reused. Long improvement with useful short
behavior would justify fresh-input confirmation; BM-like long losses would
show that this taper is insufficient, without identifying a unique causal
band. There is no amplitude or boundary grid, and no result is yet available.

A CPU-only construction on the existing OLMo reference arrays uses its own
W=4096 with the same S=4 rule: slots 15--23 equal BM, 24--27 taper, and
28 onward retain MrPro. The table remains positive and ordered. Thus the
implementation can transfer through physical period coordinates rather
than hardcoded Qwen slot IDs. This is construction evidence only; no OLMo
model evaluation of this rule has been queued or claimed.

## EVQ-inspired transfer of a gap budget from fast to long-period clocks

The user's latest correction is to investigate a resource exchange: tolerate
some loss at short and intermediate scales if high-frequency capacity can
be reassigned to produce a meaningful long-context improvement. The weak
Slower development signal does not yet establish that outcome. Most recent
arms preserve the native high band and therefore do not test this exchange.

The existing EVQ theory and implementation identify the resource more
precisely. With `omega=b**(-phi)`, the Cosh quantile map is
`phi(u)=1-asinh((1-u)*sinh(tau))/tau`. Its derivative at the high-frequency
endpoint is `tanh(tau)/tau < 1`, while at the slow endpoint it is
`sinh(tau)/tau > 1`. At tau=1 these are .761594 and 1.175201. Hence it
narrows the high-side exponent gaps and widens the slow-side gaps. This is
a transfer of **log-frequency spacing**, not a transfer of channel counts:
the Cosh density actually increases at the high end and decreases at the
slow end. It also raises interior frequencies relative to the geometric
quantiles. Increasing long-scale separation is different from making every
slow clock slower.

The useful hypothesis is that the trained model can tolerate more crowded
fast clocks while benefiting from separation among slower clocks. Neither
the Cosh variational optimum nor these derivative identities establish that
the frozen Qwen checkpoint has this tolerance. The old from-scratch Cosh
results and the unsuccessful short-LoRA transfers remain different adaptation
regimes. This new test changes only the frozen model's static table.

References inside the repository: `docs/theory/EVQ_COSH_THEORY.tex`, the
quantile construction in `scripts/lib/rope/learnable_evq.py`, and the evidence
boundaries in `docs/research/COSH_REDESIGN_EVIDENCE_REVIEW.md`. The continuous
EVQ endpoint identity does not make every k/N discretization endpoint-matched;
the new experiment explicitly fixes the actual first and last frequencies.

### Two matched resource allocations

For the actual MrPro table define `g_j=log(nu_j/nu_(j+1))`. Their sum is the
full log-frequency range, distinct from the earlier middle-band cumulative
compression budget B. The native-identical prefix contains slots 0--23,
providing donor gaps 0--22. Remove a total of one mean donor gap,
`c=0.2158673516`, uniformly from these 23 gaps. This is 4.3478% of their
combined log range, an explicit initial budget quantum rather than an
optimized amount.

Redistribute c uniformly among recipient gaps selected by their reference
geometric-mean period `sqrt(T_j*T_(j+1))`:

| Arm | Recipient period interval | Actual recipient gaps |
|---|---|---|
| HighGapToLong | [W, SW] = [32768, 131072] | 36--39 |
| HighGapToMid | [W/S^2, W/S] = [2048, 8192] | 26--31 |

Reconstruct the complete table by cumulative log gaps. Both arms have exactly
the same high-band frequencies, the same donated budget, the same official
gain, and the same first and last frequencies. All deployed frequencies are
positive and ordered; the tail after the final recipient is copied exactly
from MrPro. The Long arm changes slots 1--39, while Mid changes 1--31.
Thus this is a global redistribution affecting high and middle frequencies,
not another isolated common shift of four slow clocks. The recipient window
is one distance-directed instantiation, not a claim that the deeper slow tail
or other allocations are irrelevant.

For Long, periods at slots 36--40 become approximately 27,385/40,719/61,095/
92,503/141,332 tokens, versus 33,983/47,875/68,059/97,632/141,332 for MrPro.
The slow-band span widens by moving its faster edge; the slow boundary stays
fixed. The maximal 26-token single-slot phase change is .4245 radians in
both arms. This describes the intervention size, not a bound on model loss.

`gap_budget_transfer.py` constructs both arms with explicit receipts. They
are queued immediately after the current FullLagP2 transfer, each on all 36
development generations and 16 documents at 8K/16K/32K. Existing MrPro
references are reused. Short or intermediate losses alone do not disqualify
a meaningful long-task gain. Long versus MrPro tests net utility; Long versus
Mid tests whether this particular recipient placement is useful at identical
high-band cost. A win would motivate fresh-input confirmation and broader
rule development; these two tables alone cannot validate the entire EVQ
mechanism or a universal high-frequency redundancy claim.

HighGapToLong has now completed all 36 development generations and all
48 NLL measurements. Its 32K macro is 70.1389% versus MrPro's 87.2222%
(-17.0833 pp); its 128K macro is 67.3611% versus 78.1250% (-10.7639 pp).
There are zero improved and seven worsened rows across both lengths.
At 128K the task scores are single-key 1, multikey .5, multiquery .875,
VT .75, FWE .6667, and QA .25. No long task improves. NLL differences are
+.008165/+.017769/+.018295 at 8K/16K/32K. Thus this explicit gap transfer
does pay a short-context cost, but fails to buy a long-context benefit.
The same-budget Mid control is still running. This result rejects the
utility of the tested table on this panel; it does not reject every
high-band resource exchange or the EVQ training-regime theory.

## Completed full P2 transfer on the current Qwen3B protocol

The unmodified historical 64-frequency table and gain .074 finish all 36
development generations and 48 NLL measurements. Both existing MrPro
references are reported because the attention gain has a substantial effect:

| Method | 32K macro | 128K macro |
|---|---:|---:|
| MrPro, official gain coefficient .1 | 87.2222% | 78.1250% |
| MrPro, matched coefficient .074 | 98.3333% | 75.3472% |
| FullLagP2, original coefficient .074 | 72.9167% | 81.6667% |

P2's long advantage is +3.5417 pp over the official reference and +6.3194 pp
over the same-gain reference. The corresponding short differences are
-14.3056 and -25.4167 pp. At 128K, P2 has single-key 1, multikey .5,
multiquery 1, VT .9, FWE .75, and QA .75. Same-gain MrPro has 1/.75/.9375/
.75/.8333/.25 in the same order. Thus the positive long result is driven
by QA and VT, with multikey and FWE costs; it is not a uniform improvement.
There are five improved and seven worsened rows versus same-gain MrPro
over both lengths, or six improved and six worsened versus official MrPro.

NLL differences at 8K/16K/32K are -.004285/+.000477/-.008876 versus the
official reference but +.014000/+.017771/+.012728 versus matched gain.
All 48 matched NLL input hashes and target-token arrays agree. The apparent
default-reference NLL benefit cannot be attributed solely to the P2 table.

This is a new current-protocol developmental tradeoff, not a reproduction
of the old 1.5B/64K protocol or independent confirmation. A focused queued
job tests 16 new 128K VT and 16 new 128K QA rows. It uses the E1 cohort and
is prioritized immediately after the two high-gap transfers, before the
larger E1 confirmation. Its 32 official MrPro references are computed first
and reused by E1 later; the 32 same-gain references are also computed once,
alongside 32 frozen P2 generations. This ordering adds no duplicate work.
Short losses and long multikey/
FWE costs remain disclosed; this confirmation only tests the positive cluster.

### Does P2's positive long result already support the high-band exchange?

A direct comparison of the deployed arrays says no. P2 changes the native
high prefix (slots 0--23) by at most .0775% in relative frequency. Its total
gap change over high gaps 0--22 is +.0007754, whereas HighGapToLong explicitly
removes .2158674 from that band. P2 therefore does not supply evidence that
taking substantial high-band spacing funds its measured long-task gains.

P2 instead concentrates a large positive log-gap change at gap 29 (+.809123),
with smaller additions at 28 (+.149445) and 30 (+.131703). Gaps 31--39 become
narrower. For example, slots 29/30/31 move from periods 3,977/5,258/7,016 to
4,468/13,267/20,193 tokens, while slot 40 retains 141,332. This creates a sharp
transition and shifts many middle clocks toward longer periods. It is a
different allocation mechanism from spreading long-period clocks by drawing
spacing out of the high band. The promising P2 task result and the EVQ-inspired
hypothesis should therefore remain separate until each has direct evidence.
The complete descriptive table is in
`results/nongeometric_screen_20260909/planned_controls/p2_gap_comparison.json`;
these numerical identities do not identify which slots caused the task gains.
