# EVQ-Cosh Rebuttal Playbook

Last audited: 2026-07-26
Mode: `outcome-first / reviewer-specific / claim-local disclosure`
Readiness: **`sendable`**, provided the core route below is used and the
conditional rows remain excluded.

This is the independent response guide for Submission 11628. It is not a
paper revision and does not replace the standalone evidence owners. The only
authoritative concern entry is
`00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`.

The response objective is:

1. move `Dz6s` from 4 to 5 by closing the limited-evaluation concern without
   disturbing the mechanism framing that reviewer already values;
2. move `27bE` from 3 to 4 by directly supplying the requested \(\tau\),
   fixed-schedule, DAPE-budget, and larger-run evidence;
3. answer all four explicit `zWsa` score-move conditions so the AC can
   distinguish the remaining scope limits from a fatal novelty/evaluation
   gap; and
4. satisfy the AC's conjunction: technical distinction, direct control, and
   stronger evaluation.

The governing rule is simple: answer the concern asked, lead with the result
that changes the decision, and attach only the limitation needed to keep that
specific claim true. Do not volunteer unrelated internal failures, but do not
omit a boundary whose absence would make the selected claim misleading.

## 1. Internal status and source hierarchy

### 1.1 Review provenance

- Reviewer `27bE` is the retained payload-hashed official source.
- AC `XLtL`, `Dz6s`, and `zWsa` are author-pasted official OpenReview exports;
  no independent payload hashes are retained for them. Preserve this
  provenance limit internally.
- Do not route simulated reviews, internal audits, or historical reviewer
  paraphrases into the response.

### 1.2 Evidence vocabulary

| Tier | Meaning |
| --- | --- |
| `SUBMITTED` | Present in the submitted manuscript; keep its original seed, control, and endpoint tier. |
| `POST_SUB_RAW_HASH_BACKED` | Completed after submission with a standalone owner and retained raw/artifact hashes; reviewer-usable with its stated boundary. |
| `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction or aggregate is known, but the portable raw/per-seed owner is incomplete; do not quote its exact numbers. |
| `CONDITIONAL` | A named provenance or metadata gate remains; exclude from the core response. |
| `DESIGN_ONLY` | Plan or proposed experiment, not evidence. |
| `NEGATIVE` | A completed result that bounds a selected positive claim. |

### 1.3 Current send decision

The core response is **sendable now**. It can use:

- submitted 454M EVQ×YaRN, 432M MLA, 750M strict autoregressive, 8B LoRA,
  and exploratory video-DiT evidence;
- post-submission \(\tau\), fixed-schedule, DAPE-budget, and raw-backed
  seed-42 exact-range controls;
- post-submission 1.485B OLMo counterfactual NLL and strict NIAH;
- post-submission OLMo 13-task RULER-family evidence;
- post-submission matched LLaMA-3-8B natural-LM and RULER evidence; and
- the 1.485B step-0 to step-1,000 scratch comparison.

Do not make the core response depend on:

- the three-seed exact-range aggregate;
- exact held-out-base numbers;
- future-dated OLMo long-gap metadata;
- the fresh EVQ-only LLaMA counterfactual arm; or
- any planned matched LLaMA counterfactual experiment.

### 1.4 The OLMo scratch status is resolved

The step-1,000 result is **not conditional**. Its canonical standalone owner,
`theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`, records the
completed paired evaluation plus checkpoint, raw-result, per-token-NLL,
paired-comparison, manifest, and evaluation-anchor hashes. The sibling JSON is
explicitly scoped to `released_native_rope_baselines_only`; its statement that
it contains no EVQ result describes that earlier Native-only snapshot and does
not override the later paired owner.

Use this unified classification everywhere:

> **POST-SUBMISSION RAW/HASH-BACKED SUPPORTING EVIDENCE — paired evaluation
> complete; one early-training trajectory; same initialization, scientific
> recipe, counted-token budget, data-order prefix, and evaluation rows, but
> different trainer stacks; natural-language-modeling evidence only.**

Different trainer stacks are a scientific claim boundary, not a provenance
conflict. The result must not be called bitwise-paired, multi-seed, a pure
interior-shape isolation, or a capability result.

### 1.5 NeurIPS response mechanics

The 2026 Main Track handbook permits up to 10,000 characters **per review**,
plain Markdown only, no additional files, and no links; new results must
clarify questions raised by reviewers or the AC. Keep all source paths in this
internal playbook, not in the posted rebuttal, and verify the response UI
before sending. Internal policy source:
`https://neurips.cc/Conferences/2026/MainTrackHandbook`.

The 2026 AC pilot explicitly uses the initial metareview to focus the author
response on the most decision-relevant issues. That is why this playbook
answers the AC's three-part conjunction before attempting exhaustive
point-by-point coverage. Internal process source:
`https://blog.neurips.cc/2026/03/23/refining-the-review-cycle-neurips-2026-area-chair-pilot/`.

## 2. Result-first opening

Use this as the AC/discussion opening, then route each reviewer to the
paragraphs relevant to their own concerns.

> We added the scale, capability, and attribution tests requested in the
> reviews. First, on OLMo-2-0425-1B-Instruct (1.485B actual parameters), every
> LoRA backward pass was capped at 4K. In one matched Native/EVQ
> counterfactual-training pair, 8K strict autoregressive exact on the official
> `niah_single_1` task changed from 0/100 to 69/100; a second independently
> trained EVQ seed scored 67/100 on the same evaluation rows, while the
> initial 16K screen was only 0/20 versus 1/20. Second, under
> identical physical-8K supervision over the same 13 RULER families, the
> matched LLaMA-3-8B Native/EVQ comparison obtained 0.295% versus 14.03%
> official macro at 16K. At 8K Native was stronger (94.44% versus 77.60%),
> and at 32K EVQ was 0% while the ten completed Native task cells were also
> zero, so this is task-family-adapted 2× transfer rather than broad unseen-task
> or reliable 4× capability.
>
> We also ran the requested larger pre-specified training comparison from the
> public OLMo-2 step-0 initialization. After 1,000 steps and exactly
> 2.097B counted tokens, Geo/EVQ PPL on the same 128 document-disjoint PG-19
> rows was 161.19/167.45 at 4K, 163.88/156.87 at 8K, and 182.73/159.64 at
> 16K. Finally, independent \(\tau\), fixed analytic-schedule, and matched-range
> controls separate the allocation effect from learned capacity and scalar
> range selection. These results support EVQ-Cosh as a zero-parameter
> training-time frequency-grid allocation method; they do not claim universal
> long-context superiority or replacement of target-aware range methods.

Why this opening works:

- it answers the AC's scale/evaluation gate before revisiting small-model
  diagnostics;
- it keeps strict generation, RULER, and PPL as separate endpoints;
- it uses the completed scratch result at its correct evidence tier; and
- it states the precise 2×/4× capability boundary next to the capability
  claim.

## 3. The four score-critical questions

### 3.1 Does the effect persist at mature scale and reach real generation?

**Concern IDs:** `RDz6s.1`, `RzWsa.3`, `RzWsa.4`, `R27bE.2`,
`R27bE.5`, `AC.2`, `AC.4`.

**Direct answer:** Yes at the bounded endpoints actually measured:

- 1.485B OLMo supplies matched 4K-trained natural-text NLL and strict 8K
  autoregressive NIAH;
- a separate OLMo continuation supplies all 13 official RULER families, but
  without a matched Native continuation;
- 8B LLaMA supplies a matched natural-LM comparison and, in a separate
  protocol, a matched 13-family RULER comparison;
- the submitted paper already supplied a single-seed 8B probability-scale
  anchor and a 750M strict autoregressive endpoint; and
- 1.485B scratch step-1,000 supplies the requested larger pre-specified
  training-run branch.

**Mandatory boundary:** These protocols establish task-family-adapted 2×
capability and long-position probability effects, not clean unseen-task
transfer. They do not justify deriving capability from NLL. The mature 4×
RULER endpoints are weak or zero.

**Best owners:**

- `theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md`
- `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`
- `theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`
- `theory_results/LLAMA8B_MATCHED_RULER_MIX_20260726.md`
- `theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`
- submitted Table 12 and Appendix D, Table 23

**Recommended English:**

> We agree that NLL and teacher-forced retrieval alone do not establish usable
> context. We therefore evaluated pretrained 1.485B and 8B models with strict
> autoregressive and official RULER endpoints. With every backward pass capped
> at 4K, the matched OLMo Native/EVQ pair scored 0/100 versus 69/100 at 8K
> strict first-number exact; an independent EVQ training seed scored 67/100,
> while the initial 16K screen was only 0/20 versus 1/20.
> Separately, with identical physical-8K supervision over the same 13 RULER
> families, LLaMA-3-8B Native/EVQ official macro was 0.295%/14.03% at 16K.
> Native remained stronger at 8K (94.44%/77.60%), and neither arm showed a
> defensible 32K capability result. We therefore claim task-family-adapted 2×
> transfer, not unseen-task or universal downstream superiority.

### 3.2 Is EVQ technically distinct from FMRoPE, and is there a direct control?

**Concern IDs:** `RzWsa.1`, `RzWsa.2`, `AC.1`, `AC.4`.

**Direct answer:** Concede the missing Oka et al. citation and the overlapping
motivation. Define the narrower technical contribution:

\[
\text{Geo/FMR:}\quad \omega_i=b(T)^{-u_i},
\qquad
\text{EVQ:}\quad \omega_i=b^{-\phi_\tau(u_i)} .
\]

FMRoPE selects or retargets a geometric spectral range through the base/range
choice for a declared context. EVQ keeps a fixed nominal base and applies a
closed-form nonlinear index-to-exponent allocation inside the frequency grid
before training. Do not claim that EVQ is the first method ever to change
frequencies or that the two families are disjoint in motivation.

**Direct evidence:** In the raw/hash-backed seed-42 exact-range control, the
sampled extrema and log span match and only 30 interior frequencies differ.
Endpoint-normalized fixed-range Cosh-minus-uniform-FMRoPE NLL is
`-0.47750/-0.20499/-0.11284` at 512/1K/2K. When the schedules are instead
retargeted to the declared target length, FMRoPE is stronger. This is the
expected boundary between an interior-allocation control and target-aware
range selection, not a universal win or an additive-synergy claim.

**Status:** `POST_SUB_RAW_HASH_BACKED`, single seed.

**Owner:** `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md`.

**Recommended English:**

> We agree that FMRoPE is directly relevant and should have been cited and
> compared. The overlap is the importance of the RoPE frequency set; the
> tested parameterizations differ. FMRoPE changes the geometric base/range
> associated with a declared context, whereas EVQ uses a fixed nominal base
> and a closed-form nonlinear allocation of the interior exponent locations
> before training. In a new seed-42 control, sampled extrema, log span,
> initialization, token order, optimizer, budget, and evaluation anchors are
> matched; only the 30 interior frequencies differ. Endpoint-normalized
> fixed-range Cosh-minus-uniform-FMRoPE NLL is -0.4775/-0.2050/-0.1128 at
> 512/1K/2K. When both schedules are target-retargeted, FMRoPE is stronger.
> We therefore claim an identifiable training-time allocation variable, not
> replacement of or universal superiority over target-aware range methods.

### 3.3 Are finite \(\tau\), Cosh, and the DAPE result actually attributable?

**Concern IDs:** `R27bE.1`, `R27bE.3`, `R27bE.4`, `RDz6s.3`,
`AC.3`.

**Direct answer on finite \(\tau\):**

The theory supplies the scaling structure, not an exact finite-\(\tau\)
constant:

\[
\tau = c(\Pi)\,\frac{d_{\mathrm{eff}}}{\sqrt{L_{\mathrm{train}}}} .
\]

Here \(c(\Pi)=O(1)\) is convention- and protocol-dependent. Under the
submitted convention, \(c=1\) is an operating default, not a theorem or global
optimum. If a selection objective explicitly weights a deployment target,
\(L_{\mathrm{target}}/L_{\mathrm{train}}\) can enter \(\Pi\), together with
the base/log span, number of rotary channels, head geometry, architecture,
data, and objective. It need not alter the claimed
\(d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) scaling structure.

**Empirical calibration:** In the direct sweep, selected \(\tau=5\) gives
selection NLL `5.9898`; the rule value \(5.657\) gives `6.0017`, a difference
of `0.0119`. Across the 99-run reanalysis, the rule beats Geo in 7/9
configuration means and 18/27 seed pairs, but beats the neighboring empirical
pilot in only 3/9 means and 8/18 available pairs. It is therefore a useful but
fallible basin selector.

**Direct answer on DAPE tuning:** Yes. The submitted DAPE row received
positional-parameter learning-rate multipliers of `10×` and `100×`; the
reported `100×` choice gives PPL@8K `455.3`, versus `477.7` at `10×`, both
seed 42. This directly answers the tuning-budget question.

**Shape attribution:** Learned capacity remains a confound regardless of the
budget answer. The fixed, zero-parameter schedule study keeps the positional
operator and training protocol fixed. EVQ-minus-Geo mean tail NLL is
`-0.256/-0.305/-0.223/-0.238` at 1K/2K/4K/8K, with 3/3 seeds in the same
direction. Exponential and attention-derived two-band schedules are stronger
at some lengths. Thus allocation matters; Cosh is not claimed to be
universally optimal.

**Best owners:**

- `theory_results/EXPERIMENT_REPORT_20260724.md`
- `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`
- `docs/exp/2026-02-24_128tok_baseline_report.md`

**Recommended English:**

> The finite-\(\tau\) rule is an empirical operating rule with a derived
> scaling structure, not an exact optimizer. More precisely, the analysis
> motivates \(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\), where the
> convention-dependent \(O(1)\) coefficient is calibrated empirically. In a
> direct sweep, the selected \(\tau=5\) and the rule value 5.657 differ by only
> 0.0119 selection NLL. The broader 99-run audit also shows why we do not call
> the rule optimal: it often selects a useful basin but is not consistently
> better than a neighboring empirical choice.
>
> The submitted DAPE row was separately tested with 10× and 100×
> positional-parameter learning-rate multipliers; the reported 100× setting
> obtained 455.3 PPL@8K versus 477.7 at 10×. To remove learned capacity from
> the allocation-shape question, we additionally compared fixed
> zero-parameter schedules under the same operator and training protocol.
> EVQ improved over Geo at all four extrapolation lengths, while exponential
> and two-band schedules were stronger at some lengths. This supports
> allocation as a design axis, not universal Cosh optimality.

### 3.4 Does EVQ complement range scaling without claiming tuned dominance?

**Concern IDs:** `RDz6s.2`, `AC.1`, `AC.4`.

**Direct answer:** The submitted result establishes substrate-dependent
leverage under the **same fixed** YaRN transformation, not superiority to an
exhaustively optimized Geo+YaRN or target-aware range search.

In submitted Table 3, with YaRN scale \(s=8\) fixed for both substrates and
three training seeds, Geo+YaRN versus EVQ+YaRN gives:

- teacher-forced NLL-gap PK@8K: `61±3%` versus `100±0%`;
- PK@12K/16K: `59/51%` versus `79/68%`;
- PPL@8K/16K: `82.9/157.7` versus `70.9/107.5`.

This shows that the training-time grid changes what the same inference-time
transformation acts on. The exact-range result separately shows that
target-aware FMRoPE can be stronger when the target range itself is retuned.

**Recommended English:**

> We agree that a fixed matched YaRN scale does not establish dominance over a
> fully optimized range search. Its purpose is narrower: it tests whether the
> training-time frequency substrate changes the leverage of the same
> inference-time transformation. In submitted Table 3, with \(s=8\) fixed for
> both arms and three training seeds, Geo+YaRN versus EVQ+YaRN obtained
> \(61\pm3\%\) versus \(100\pm0\%\) teacher-forced NLL-gap retrieval at 8K
> and 82.9 versus 70.9 PPL. We therefore retain a substrate-dependent
> complementarity claim, not tuned-YaRN dominance.

## 4. Compact evidence table

All endpoints below must remain separate. `PPL/NLL`, teacher-forced NLL-gap,
strict autoregressive exact, and RULER are not interchangeable.

| Evidence | Tier | Exact reviewer-usable result | Claim-local boundary | Owner |
| --- | --- | --- | --- | --- |
| 454M EVQ×YaRN, submitted Table 3 | `SUBMITTED` | Fixed \(s=8\), 3 seeds: PK@8K `61±3→100±0%`; PPL@8K `82.9→70.9`; PPL@16K `157.7→107.5` | PK is teacher-forced NLL-gap; fixed scale, not tuned-range dominance | submitted Table 3; curated Primary-I JSON |
| 432M MLA, submitted Table 18 | `SUBMITTED` | 500M tokens, 8K training, 16 rotary channels, 3 seeds: PPL@16K `138.8±5.5→95.6±4.1`; same \(s=4\) EVQ+YaRN `71.1±4.1` | Tested scarce-channel PPL configuration; not downstream capability | submitted Table 18; `table18_mla_3seed_aggregate.json` |
| 750M continuation, submitted Table 12 | `SUBMITTED` | Teacher-forced 8K retrieval `100%→100%`; strict AR exact `0→77.5%`; PPL@16K `45.1→24.4` | Single-seed, task-specific supporting continuation; not multi-seed scale closure | submitted Table 12 |
| Video DiT, submitted Table 14 | `SUBMITTED` | 129M/382M exploratory breadth evidence | Cross-modal scope only; not production video or primary evidence | submitted Table 14 |
| LLaMA-3-8B LoRA, submitted Appendix D Table 23 | `SUBMITTED` | untouched Base/EVQ-LoRA PPL: 8K `7.42/9.63`, 16K `176.3/21.5`, 32K `1942.5/104.3` | Single seed; unmatched PPL scale anchor, not attribution or capability | submitted Appendix D, Table 23 |
| Direct \(\tau\) sweep | `POST_SUB_RAW_HASH_BACKED` | selected `5` vs rule `5.657`: `0.0119` selection-NLL gap | One direct sweep; rule is a fallible basin prior | `EXPERIMENT_REPORT_20260724.md`; `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` |
| Fixed analytic schedules | `POST_SUB_RAW_HASH_BACKED` | EVQ−Geo NLL `-0.256/-0.305/-0.223/-0.238` at 1K/2K/4K/8K, 3/3 direction | Other fixed schedules are stronger at some lengths | `EXPERIMENT_REPORT_20260724.md` |
| Exact-range Cosh vs FMRoPE | `POST_SUB_RAW_HASH_BACKED` | seed 42 Cosh−FMR NLL `-0.47750/-0.20499/-0.11284` at 512/1K/2K | Endpoint-normalized, single seed; target-retargeted FMR is stronger | `MATCHED_RANGE_COSH_500M_S42_20260724.md` |
| OLMo-2 1.485B matched CF natural LM | `POST_SUB_RAW_HASH_BACKED` | Native/EVQ NLL: 4K `2.235/2.548`, 8K `3.735/2.703`, 16K `4.851/2.925` | Teacher-forced NLL; one matched training seed; in-window cost | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| OLMo-2 1.485B strict NIAH | `POST_SUB_RAW_HASH_BACKED` | 8K exact Native/matched-EVQ `0/100 vs 69/100`; independent EVQ seed `67/100` | Same generator family, disjoint rows/values; 16K screen only `0/20 vs 1/20` | same owner |
| OLMo-2 13-task RULER continuation | `SUPPORTING` | official macro 4K/8K/16K `37.51/21.29/6.13%` | Single EVQ continuation; RULER-family supervised; no matched Native continuation | `OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md` |
| LLaMA-3-8B matched ordinary LM | `SUPPORTING` | EVQ−Native NLL 8K/16K/32K `+0.390/-1.510/-2.048` | Single-seed teacher-forced probability result; separate from RULER protocol | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| LLaMA-3-8B matched RULER | `SUPPORTING` + `NEGATIVE` at 4× | Native/EVQ macro 8K `94.44/77.60%`; 16K `0.295/14.03%`; 32K EVQ `0%`, Native 10/13 completed and all zero | One continuation seed; same 13 families; task-adapted 2×, not unseen-task | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` |
| OLMo-2 1.485B scratch step-1,000 | `POST_SUB_RAW_HASH_BACKED` | Geo/EVQ PPL 4K `161.19/167.45`, 8K `163.88/156.87`, 16K `182.73/159.64`; `122/128`, `126/128` docs favor EVQ at 8K/16K | One early-training trajectory; different trainer stacks; LM only | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| Held-out base/head aggregate | `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction available internally | Exact numbers excluded until a dedicated raw/per-seed owner exists | `EXPERIMENT_REPORT_20260724.md` §4 |
| Three-seed exact-range aggregate | `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction available internally | Use raw-backed seed-42 control instead | `MATCHED_RANGE_COSH_500M_3SEED_20260724.md` |

## 5. Reviewer-specific response order

Do not paste the same omnibus block to every reviewer.

### 5.1 `Dz6s` — protect the 4 and give a reason for 5

Order:

1. thank the reviewer for recognizing the operator/table separation and
   EVQ×YaRN mechanism;
2. lead with mature strict NIAH and matched 8B RULER;
3. surface submitted 750M strict AR and submitted 8B PPL as prior scale
   anchors;
4. state that fixed \(s\) establishes substrate leverage, not tuned dominance;
5. explicitly separate surrogate proof, exact-kernel diagnostics, and trained
   results.

**Copy-ready response:**

> Thank you for recognizing the operator/frequency-table separation and the
> fixed-scale EVQ×YaRN result. We agree that the submitted diagnostics alone
> did not establish broadly usable context. We therefore added pretrained
> 1.485B and 8B experiments with strict autoregressive and official RULER
> endpoints. With 4K-only backward passes, the matched OLMo Native/EVQ pair
> changed 8K strict NIAH exact from 0/100 to 69/100; an independent EVQ seed
> scored 67/100, while the initial 16K screen was only 0/20 versus 1/20. In a
> separate matched LLaMA-3-8B experiment with identical
> physical-8K supervision over the same 13 RULER families, 16K official macro
> was 0.295%/14.03% for Native/EVQ. Native was stronger at 8K, and 32K
> remained unsolved, so we claim bounded task-family-adapted 2× transfer.
>
> The submission also contained a 750M strict autoregressive endpoint
> (0%/77.5% at 8K) and an 8B LoRA PPL scope check; we retain both at their
> original supporting tiers. Its exploratory 129M/382M video-DiT experiments
> additionally provide cross-modal scope evidence, not a production-video or
> stronger-language-benchmark claim. For YaRN, the fixed \(s=8\) comparison shows
> substrate-dependent leverage, not superiority over an exhaustively tuned
> Geo+YaRN search. Finally, we will keep the conditional surrogate theorem,
> exact-kernel diagnostics, and trained-model observations as three separate
> epistemic layers.

### 5.2 `27bE` — answer the requested ablations exactly

Order:

1. finite-\(\tau\) scaling versus coefficient;
2. direct \(\tau\) sweep and 99-run limitation;
3. DAPE `10×/100×` budget answer;
4. fixed non-Cosh schedules;
5. 1.485B \(d_{\rm head}=128\), base-500K scratch run;
6. state held-out-base numbers remain excluded rather than substituting a
   different result.

**Copy-ready response:**

> Thank you for separating the exact statement from the finite-\(\tau\)
> operating regime. We agree that the small-\(\tau\) argument does not
> uniquely determine a finite optimum. Our narrower statement is
> \(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\): the analysis supplies the
> scaling structure, while the convention-dependent \(O(1)\) coefficient is
> calibrated empirically. In a direct sweep, selected \(\tau=5\) and the rule
> value 5.657 differ by only 0.0119 selection NLL. Across the broader 99-run
> audit the rule is useful but fallible, so we do not call it globally or
> approximately optimal.
>
> The submitted DAPE row did receive a dedicated tuning budget: positional
> learning-rate multipliers 10× and 100× were tested, and the reported 100×
> setting obtained PPL@8K 455.3 versus 477.7 at 10×. Because learned capacity
> still confounds shape attribution, we added fixed zero-parameter controls
> under the same operator and training protocol. EVQ-minus-Geo NLL is
> -0.256/-0.305/-0.223/-0.238 at 1K/2K/4K/8K with 3/3 seeds agreeing, while
> exponential and two-band schedules are stronger at some lengths. This
> establishes allocation as an independent axis without asserting universal
> Cosh optimality.
>
> We also completed the requested larger pre-specified run on OLMo-2
> (1,484,916,736 parameters, \(d_{\rm head}=128\), base 500K) from the public
> step-0 initialization. Both trajectories use 1,000 steps, global batch 512,
> the pinned scientific recipe and data-order prefix, and exactly
> 2,097,152,000 counted tokens. Geo/EVQ PPL on the same 128 PG-19 documents is
> 161.19/167.45 at 4K, 163.88/156.87 at 8K, and 182.73/159.64 at 16K. The
> trainer implementations differ, so this is same-initialization/same-recipe
> single-trajectory LM evidence, not a bitwise-paired or multi-seed estimate.

### 5.3 `zWsa` — satisfy the four score-move conditions for the AC record

Order the reply exactly like the reviewer's four questions:

1. cite Oka et al. and define the narrow distinction;
2. give the raw-backed seed-42 matched-range control;
3. give matched 8B RULER plus bounded OLMo RULER;
4. give 1.485B and 8B scale evidence.

Do not debate whether the reviewer should have noticed the submitted 8B row.

**Copy-ready response:**

> Thank you for identifying FMRoPE; we agree it should have been cited and
> directly compared. Both works emphasize that the RoPE frequency set matters.
> The narrower EVQ contribution is a fixed-base, closed-form nonlinear
> allocation of interior exponent locations before training; FMRoPE selects or
> retargets a geometric range for a declared context. In a seed-42 control
> matching sampled extrema, log span, initialization, data order, optimizer,
> budget, and evaluation anchors, endpoint-normalized fixed-range
> Cosh-minus-uniform-FMRoPE NLL is -0.4775/-0.2050/-0.1128 at 512/1K/2K.
> When both schedules are
> target-retargeted, FMRoPE is stronger, so our claim is distinct
> interior-allocation leverage rather than FMRoPE replacement.
>
> We also added the two requested evaluation axes. On matched
> LLaMA-3-8B Native/EVQ adaptations, official 13-family RULER macro at 16K is
> 0.295%/14.03% after physical-8K training; Native remains stronger at 8K and
> 32K remains unsolved. On OLMo-2 (1.485B actual parameters) with 4K-only
> backward passes, strict 8K NIAH exact is 0/100 versus 69/100 in the matched
> pair, with an independent EVQ seed at 67/100; the initial 16K screen was
> only 0/20 versus 1/20. The submission already
> contained a bounded 8B PPL scope check; the new 1.485B step-0 to step-1,000
> run additionally shows lower 8K/16K natural-text PPL under the tested early
> training budget. These results directly address the requested FMRoPE,
> RULER, and \(\geq1\)B checks while preserving their seed and task-family
> limits.

### 5.4 AC `XLtL` — adjudicate the conjunction, not every detail

Use one compact table internally to ensure the final AC note closes all three
gates:

| AC gate | Status | Decisive evidence | Remaining boundary |
| --- | --- | --- | --- |
| `AC.1` novelty + direct comparison | Answered | formula-level distinction; raw-backed seed-42 exact-range control | single seed; target-retargeted FMR can be stronger |
| `AC.2` stronger scale/evaluation | Answered at supporting tier | 1.485B strict NIAH, matched 8B RULER, 1.485B scratch, submitted 750M/8B anchors | task-family 2×; mature 4× unsolved; mostly single seed |
| `AC.3` attribution | Answered | direct \(\tau\) sweep, 99-run audit, DAPE budget, fixed analytic schedules | rule fallible; Cosh not universal |
| `AC.4` recommendation conjunction | Answered with bounded claims | all three gates above are backed by completed evidence | no universal SOTA or range-method replacement claim |

**Copy-ready AC note:**

> The rebuttal now addresses the three conditions identified in the
> metareview. First, we cite FMRoPE, define the narrower fixed-base
> interior-allocation contribution, and provide a matched-range control whose
> opposite target-retargeted outcome also bounds the claim. Second, we add
> pretrained 1.485B and 8B strict autoregressive/RULER evidence plus a
> completed 1.485B step-0 to step-1,000 training comparison. Third, independent
> \(\tau\), DAPE-budget, and fixed analytic-schedule studies separate the
> operating rule, learned capacity, and allocation shape. We therefore ask
> that EVQ-Cosh be assessed as a simple zero-parameter training-time allocation
> axis with bounded mature-model evidence, not as a claim of universal
> long-context SOTA or replacement of target-aware scaling.

## 6. LLaMA counterfactual classification

This section is internal and prevents three different LLaMA protocols from
being merged.

| Item | Actual status | Permitted use |
| --- | --- | --- |
| Fresh EVQ-only LLaMA counterfactual arm | Completed post-submission, single seed, single arm; strict NIAH 8K/16K/32K `20/20`, `6/20`, `0/20`; no matched Native arm | Bounded feasibility only; exclude from the core response because it adds no causal comparison and weakens beyond 8K |
| Matched LLaMA natural-LM study | Completed; ordinary full-token LM, not counterfactual | Teacher-forced NLL/probability evidence only |
| Matched LLaMA RULER study | Completed; answer-only 13-family continuation, not counterfactual | Task-family-adapted RULER comparison |
| Matched Native/EVQ LLaMA counterfactual pair | `DESIGN_ONLY`; not completed | Not evidence, not needed for the current rebuttal, and **do not run** |

The core strategy is already complete without a new LLaMA counterfactual
experiment: OLMo supplies the matched counterfactual capability endpoint;
LLaMA supplies separate matched natural-LM and task-family RULER endpoints.

## 7. Claims to avoid

| Avoid | Why it backfires | Safe replacement |
| --- | --- | --- |
| “EVQ is the first method to optimize frequencies/exponents.” | LongRoPE and other work search or alter per-channel frequencies; the priority claim is unnecessarily broad. | “EVQ provides a fixed-base, closed-form nonlinear allocation of interior exponent locations.” |
| “EVQ is better than/replaces FMRoPE or YaRN.” | The target-retargeted FMR control reverses; submitted YaRN is fixed-scale. | “The tested allocation and range choices are separately identifiable; the same fixed range transform has substrate-dependent leverage.” |
| “Cosh or \(\tau=d_{\rm eff}/\sqrt L\) is optimal.” | Fixed alternatives can win and the 99-run rule is fallible. | “Cosh is the solution to the stated surrogate; the rule is an empirically calibrated basin prior.” |
| “The theory exactly predicts finite \(\tau\).” | It supplies a scaling structure, not the finite \(O(1)\) coefficient. | “\(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\), with \(c(\Pi)\) calibrated empirically.” |
| “PPL improvement proves long-context ability.” | NLL/PPL is teacher-forced probability, not generation. | Report strict NIAH or RULER separately. |
| “The two OLMo EVQ seeds are 136/200.” | They share the same evaluation rows and only one is paired with Native. | “One matched pair is 0/100 versus 69/100; an independent EVQ training seed scores 67/100.” |
| “OLMo full RULER is a matched EVQ improvement.” | No Native arm received that continuation mixture. | “A separate single-seed EVQ RULER-family continuation reached …” |
| “LLaMA shows unseen-task or zero-shot transfer.” | Training and evaluation rows are disjoint, but the 13 generator families are shared. | “Task-family-adapted 2× transfer.” |
| “The submitted 8B row is a matched control.” | It compares untouched Base with EVQ-LoRA. | “A submitted single-seed PPL scale anchor; a separate post-submission protocol provides matched Native/EVQ controls.” |
| “The scratch run is conditional or lacks EVQ raw evidence.” | This reverses the owner hierarchy and discards completed hashed evidence. | Use the unified `POST_SUB_RAW_HASH_BACKED` classification in §1.4. |
| “The scratch trajectories are bitwise matched.” | Geo and EVQ use different trainer stacks. | “Same initialization, scientific recipe, counted-token budget, and data-order prefix; different trainer implementations.” |
| “LLaMA is counterfactual-trained.” | The matched natural-LM and RULER protocols are not counterfactual. | Reserve “counterfactual-trained” for the OLMo routing stage or explicitly label the excluded fresh EVQ-only LLaMA arm. |

Do not mention internal alias bugs, abandoned schedules, GPU execution details,
or failed unrelated searches unless a posted claim would otherwise become
misleading. They do not answer a retained reviewer concern.

## 8. Final send gate

The core response remains `sendable` only if all items below pass:

1. Use the raw-backed seed-42 exact-range result; do not quote the
   unpromoted three-seed exact-range aggregate.
2. Do not quote exact held-out-base numbers until a dedicated raw/per-seed
   owner exists. The completed 1.485B scratch run answers the separate
   larger-run branch, not the held-out-base branch.
3. Keep the scratch row as `POST_SUB_RAW_HASH_BACKED`; retain
   single-trajectory, different-trainer-stack, early-budget, and LM-only
   boundaries next to that claim.
4. Do not use future-dated OLMo long-gap results until their metadata is
   reconciled.
5. Keep submitted and post-submission evidence explicitly separated.
6. Keep OLMo natural NLL, strict NIAH, and RULER continuation as three
   distinct endpoints/protocol statements.
7. Keep LLaMA ordinary-LM NLL and the 516-step RULER continuation separate.
8. State the LLaMA 32K/4× negative result next to the positive 16K RULER
   result; state the OLMo 16K/4× weakness next to its 8K capability result.
9. Answer the DAPE tuning-budget question with `10×/100×`; use fixed
   schedules for allocation-shape attribution.
10. Do not use the fresh EVQ-only LLaMA counterfactual arm in the core
    response, and do not run the design-only matched counterfactual plan.
11. Verify every posted number once against the named standalone owner after
    final character-limit editing.
12. Keep each reviewer response within 10,000 characters, plain Markdown, no
    links or attachments.

If an optional conditional row is added, the package immediately changes to
`needs_author_input` until that row's stated owner/provenance gate is closed.
