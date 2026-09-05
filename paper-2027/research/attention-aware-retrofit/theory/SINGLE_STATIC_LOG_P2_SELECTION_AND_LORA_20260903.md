# Single-static log-p2 selection and same-substrate LoRA

- **Date:** 2026-09-03
- **Status:** `COMPLETE HISTORICAL-SELECTION SYNTHESIS / ZERO-TRAINING
  NUMBERS OWNED BY EXISTING RESULTS / LORA SPECIFICATION LATER EXECUTED`
- **Evidence labels:** the candidate selection in Sections 2--5 is a synthesis
  of existing **Observations** under the paper's capability-first engineering
  contract. Equations are **Derived identities** of the already executed table.
  Section 7 preserves the pre-outcome **Working method**; its later execution
  is owned by
  [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md`](../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md).
- **Exact question:** Under a single-table, single-attention-path deployment
  constraint, which completed zero-training form is retained by the repository's
  sequential Native-retention and capability replacement rules, and what LoRA
  continuation is defined on that same substrate?
- **Engineering constraint:** one table and one fixed gain are installed before
  prefill and used at every evaluated length. There is no request routing,
  Native/long branch, table switch, cache rewrite, prefix handoff, head mask, or
  piecewise/ramp operator.
- **Execution:** repository reading and deterministic arithmetic only. No model
  load, inference, training, GPU, remote access, parameter search, or new
  experiment occurred in this 2026-09-03 owner. The later LoRA result is a
  separate execution and does not retroactively make this synthesis prospective.
- **Numerical owners:**
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](../results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md)
  and
  [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md`](../results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md).

## 1. Decision

For the ICLR paper's capability-first static deployment question, the sequential
historical decision retains the already executed

> **full 64-slot legacy-u p2 movement, installed by log-s4, with the OLMo
> 1x-selected fixed gain coefficient `c=.074`.**

This is one global table tensor, not a three-zone approximation. The selection
is a retrospective reading of a **sequential stop tree**, not a jointly executed
factorial over all static candidates:

1. full-p2 log entered as the incumbent after passing both registered OLMo 1x
   natural-retention gates and producing strong 2x/4x natural and RULER results;
2. matched-gain arithmetic p2 and fitted C2 were stopped when their registered
   PG-19 retention gates failed; missing later cells are not counted as losses;
3. Native-isotonic passed the reused 1x gates but failed the prospective
   replacement comparison against incumbent p2 on fresh structured 4K/8K;
4. the Qwen recomputation supplies long-capability construction transfer only,
   not Native compatibility or an ordered-profile guarantee.

Therefore full-p2 remains the **retrospectively retained capability-first
engineering reference among these named historical candidates**. The
Native-isotonic table has better natural likelihood and more 1x natural margin,
but is materially worse on fresh structured 4K/8K. This is an observed
two-candidate endpoint tradeoff, not a formal Pareto frontier or deployment
utility.

No global training-free SOTA, universal optimum, or minimum possible LM loss is
claimed. The exact result is that the existing evidence already selects a
single-path engineering profile; request routing was an invalid substitution
for this question.

This owner does not close the separate prospective direction in the root
README. Because p2 and `.074` were selected with historical outcomes, the
retained incumbent must not be relabeled as a newly theory-derived,
pre-outcome candidate.

## 2. Exact single-table form

Let the Native table contain `K` positive frequencies

\[
\omega_0>\cdots>\omega_{K-1}>0,
\qquad x_k=-\log\omega_k=a+Rz_k.
\]

The executed movement uses the **legacy numerical procedure**, not the later
exact-u projector. Starting from the float32 Native frequencies, it builds lags
`0..4095` with causal pair-count weights, reduces them to at most 2048 points by
the recorded weighted block-centroid rule, forms the float64 sin/cos design, and
uses `np.linalg.lstsq(..., rcond=1e-10)`. That argument is an SVD cutoff, not a
ridge penalty. Let `U_k^legacy` be the resulting clipped residual-energy
fraction for pair `k` against all other pairs. With nonzero range, define

\[
\widetilde U_k^{\rm legacy}=
\frac{U_k^{\rm legacy}-\min_j U_j^{\rm legacy}}
{\max_j U_j^{\rm legacy}-\min_j U_j^{\rm legacy}},
\qquad
m_k=(1-\widetilde U_k^{\rm legacy})^2.
\tag{1}
\]

The historical builder sets the normalized vector to zero in the degenerate
zero-range case. That branch is part of the procedure identity even though the
executed OLMo vector has nonzero range. Equation (1) therefore denotes the
frozen legacy procedure/vector and does not silently substitute full-lag
high-precision `exact-u`.

The selected `s=4` table and amplitude are

\[
\boxed{\omega_k^\star=\omega_k4^{-m_k}},
\qquad
\boxed{g^\star=1+0.074\log4=1.102585782722872}.
\tag{2}
\]

For the released OLMo-2-0425-1B-Instruct checkpoint, the executed float32 table
SHA-256 is
`56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`.
The same tensor and scalar are installed at 1x, 2x, and 4x. Equation (2) adds no
learned parameter and uses no conditional dispatch.

For the **verified OLMo tensor**, CPU reconstruction gives `m_0=0`,
`m_{K-1}=1`, strict transformed-frequency order, and the hash above. Under
those executed conditions, the fast endpoint remains fixed and the slow
endpoint expands the sampled log-frequency span from `R` to `R+log 4`, so its
support-normalized allocation is

\[
\boxed{
z_k^\star=\frac{Rz_k+(\log4)m_k}{R+\log4}
}.
\tag{3}
\]

Equation (3) is the precise sense in which this OLMo table changes `z`. It is
not a generic consequence of min--max normalization: a different checkpoint
need not place the extrema at its endpoints or preserve frequency order. Unlike the
discarded ramp draft, all `m_k` come from the full frozen profile; no frequency
partition or segment boundary is introduced.

The log action was selected after the original p2 profile existed. It removes
the arithmetic movement-saturation defect and satisfies

\[
-\frac{\log(\omega_k^\star/\omega_k)}{\log4}=m_k.
\]

This is a scale-coordinate identity, not a proof that p2 or `c=.074` is an
analytic constant. The p2 link was originally retained after reading 8K
outcomes, and `c=.074` was selected only on OLMo 1x PG-19.

## 3. Historical selection in actual time order

| Date | Completed evidence | Consequence for the single-table choice |
| --- | --- | --- |
| 2026-08-22 | Arithmetic p2 plus `c=.1` established table-by-gain interaction; p2 beat p1 at 8K and tied it at 16K within `.0025` | Retain p2 as the capability source profile; do not call it analytically optimal |
| 2026-08-23/24 | Same-support coarse arithmetic ramps matched detailed long profiles, while two fixed-Native-support analytic tables failed 1x badly | Detailed shape was compressible at those long endpoints, but a ramp was not established as the final static solution |
| 2026-08-25/26 | Analytic Cosh doses and an oracle direction exposed continuous short/full/tail tradeoffs; no analytic arm passed the joint gate | Native cost and long benefit must be checked jointly; geometry alone does not select a table |
| 2026-08-31 | The full p2 mask with log action and `c=.074` passed the formal static double gate, improved long natural endpoints, retained RULER, and transferred construction to Qwen | Current single-static numerical owner |
| 2026-09-01 | C2 compression preserved much long behaviour but missed OLMo and Qwen Native gates; K32/K128 normalized-index tables are a distinct family | Do not replace the full p2 owner with C2 or use cross-K C2 results as p2 evidence |
| 2026-09-03 | Exact-u Native-isotonic improved natural likelihood but lost fresh core-4 at 4K/8K | Record the two-candidate endpoint tradeoff; retain p2 under the capability-first replacement rule |

## 4. Native-window cost

All OLMo rows below use one static table and the fixed `c=.074` amplitude.

| Candidate | 1x PG-19 PPL retention | 1x five-task retention | Registered status |
| --- | ---: | ---: | --- |
| arithmetic p2 | `0.869584` | not the deciding owner | PG gate failed |
| fitted C2 log compression | `0.870971` | `0.904932` | PG gate failed |
| **full p2 log** | **`0.875302`** | **`0.915103`** | both gates passed |
| exact-u Native-isotonic log | `0.885660` | `0.969075` | both gates passed; separate fresh capability negative |

The isotonic row proves that full-p2 does not globally minimize Native cost.
The capability-first statement instead uses the fresh structured endpoint:

| Fresh core-4 | Full p2 log | Native-isotonic log | Isotonic minus p2, paired 95% CI |
| --- | ---: | ---: | --- |
| 4K | **`.8375`** | `.6925` | `-.1450 [-.2350,-.0550]` |
| 8K | **`.7225`** | `.5475` | `-.1750 [-.2600,-.0925]` |
| 16K | `.4025` | `.4100` | `+.0075 [-.0650,+.0750]` |

Thus the sequential replacement rule retains full-p2 as the capability-first
incumbent among these observed candidates. There is no independent scalar cost
functional, common candidate-by-endpoint matrix, or claim that it minimizes
Native loss over every utility or untested table.

## 5. Extrapolation evidence

### OLMo static reference comparison with arm-specific gains

| Endpoint | Native (`g=1`) | official-equation YaRN-4 (`c=.1`, `g=1.138629...`) | Full p2 log (`c=.074`, `g=1.102586...`) |
| --- | ---: | ---: | ---: |
| PG-19 NLL, 2x | `7.100855` | `3.441096` | **`3.083278`** |
| six-task macro, 2x | `.058883` | `.218527` | **`.307614`** |
| PG-19 NLL, 4x | `7.205538` | `3.793878` | **`3.081946`** |
| six-task macro, 4x | `.021517` | **`.263677`** | `.260055` |
| RULER-13, 8K | `0` | `.24308` | **`.66705`** |
| RULER-13, 16K | `.00385` | `.10564` | **`.49859`** |

This is a comparison of complete executed method bundles, not a matched-gain or
pure-table contrast. It is strong multi-endpoint extrapolation, not uniform dominance: YaRN is
slightly higher on the six-task 4x macro, and matched arithmetic p2 is slightly
higher on RULER-13 at 16K (`.50481` versus `.49859`) while failing the 1x PG
gate. The log profile is selected by the complete constraint set, not by winning
every cell.

### Qwen construction transfer

Without reading Qwen outcomes, the same algorithm recomputed `m` from the Qwen
Native `L=32768`, `b=10^6`, `K=64`, reused `s=4` and the OLMo-selected
`c=.074`, and scored `.7000/.5875` on 64K/128K core-4 versus
`.6025/.4650` for the executed YaRN-4 control. This establishes long-capability
construction transfer only: the Qwen realization has crossings at pairs 1 and
18 and has no matched Qwen 1x PPL gate.

K32/K128 normalized-index successes use the distinct C2-derived transport and
must not be presented as full-p2 replication.

## 6. Why this is the current answer rather than another formula

- **No routing:** exact Native is not claimed; measured single-table retention
  owns the cost.
- **No ramp or segmentation:** the complete 64-slot `m` vector is retained.
- **No new surrogate:** p2/log/`.074` are the already executed operating
  choices, with their post-outcome history disclosed.
- **No metric pooling:** natural NLL, generated natural tasks, fresh core-4, and
  RULER remain separate.
- **No unsupported cross-K merge:** the successful normalized-index C2 family is
  retained as separate evidence.
- **No universal optimum:** Native-isotonic remains a genuine alternative when
  likelihood, rather than structured capability, is the deployment priority.

## 7. Same-substrate LoRA working specification

The declared same-substrate adaptation estimand is **defined** to keep Equation
(2) installed for every training and evaluation request. This is a causal-control
choice, not something the historical 2x2 proved necessary. There is no Native
training branch and no inference switch.
For every adapted layer,

\[
W_Q^\star=W_Q+B_QA_Q,
\qquad
W_K^\star=W_K+B_KA_K,
\tag{4}
\]

while the base weights and V/O projections remain frozen in the first version.
The fixed rotary amplitude `g^star` stays explicit at its existing runtime
action point; it is not silently delegated to LoRA.

The closest completed protocol evidence is deliberately kept in two pieces:

1. a rank-64 Q/K LoRA trained for 300 steps on Native-table 4K natural LM was
   evaluated in the full weights-by-table 2x2. With that weight state, static
   log-p2 scored NLL `2.913996/2.900898/2.915873` at 1x/2x/4x versus
   Native-table `2.927003/7.114380/7.091656`;
   This is a cross-substrate readout perturbation: it shows no sign reversal for
   PG-19 NLL, not same-substrate log-p2 adaptation;
2. the July matched Stage-A study showed that continuing existing Q/K while
   freezing inherited V/O can repair task capability and retain length transfer
   on its own Native/EVQ substrates. It did not test log-p2 and its inherited
   adapters cannot be transplanted here.

The same-substrate continuation is therefore:

- install and freeze the exact log-p2 tensor and `c=.074` amplitude before
  adaptation;
- train only Q/K low-rank residuals while keeping that table active at every
  length;
- reuse the existing complete-answer-plus-immediate-EOS and natural replay
  contract if task adaptation is later authorized;
- evaluate the resulting adapter only with the same table/gain identity.

This paragraph records the declared specification as it existed before
execution. The later rank-8/alpha-16/96-step matched run improved PG-19 NLL but
did not improve the measured five-task or fresh core-4 macros; see the result
owner linked above. Rank, alpha, schedule, and budget remain protocol fields,
not proven optima, and that negative capability result does not alter the
zero-training table decision in Sections 1--6.

## 8. Adversarial audit

| Tempting statement | Verdict |
| --- | --- |
| Full-p2 has zero Native damage | False; 1x PG retention is `.875302` |
| Full-p2 globally minimizes Native cost | False; Native-isotonic has better natural retention |
| Full-p2 wins every long endpoint | False; YaRN wins the six-task 4x macro narrowly and arithmetic p2 wins one 16K RULER comparison |
| p2/log/`.074` are theory-derived constants | False; their outcome-selection history is explicit |
| Coarse-ramp or C2 scores belong to full-p2 | False; they are different tensors and installation laws |
| K32/K128 normalized-index confirms full-p2 | False; those are C2-derived assets |
| Qwen proves short retention | False; no matched Qwen 1x PPL gate exists |
| The executed log-p2 LoRA solves long-context capability | False; paired PG-19 improved, but generated-task and fresh core-4 macros did not |
| This is a formal Pareto frontier or global training-free SOTA | False/unresolved; no deployment utility or matched current-method comparison exists |

## 9. Manuscript increment and claim ceiling

The current manuscript's mature pure-`z` equation describes the earlier
arithmetic same-support experiment, while the zero-training system reports the
earlier routed deployment. Neither should be silently relabeled as Equation
(2). If admitted later, this owner contributes a separate single-static result:

> Under one static table and fixed gain at every length, the frozen full-p2
> log profile passes the registered OLMo 1x retention gates and preserves strong
> 2x/4x likelihood and capability. A fresh structured comparison retains p2 over
> a Native-isotonic alternative despite the latter's lower natural NLL, exposing
> an endpoint-dependent tradeoff rather than a universal optimum.

Allowed claims are limited to the exact checkpoint, tensor, gain, rows, tasks,
scorers, and transfer boundaries above. This owner does not alter the EVQ-Cosh
surrogate theorem, does not turn static geometry into a behavioural predictor,
and does not authorize an experiment.
