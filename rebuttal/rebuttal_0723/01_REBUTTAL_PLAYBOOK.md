# EVQ-Cosh rebuttal playbook

Last updated: 2026-07-26
Mode: `revise`
Readiness: `needs_author_input`

This is the compact reviewer-facing decision guide for Submission 11628. It is
not a response to paste verbatim. The authoritative concern text is
`00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`; the compact positive and internal
evidence indices are:

- `theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md`;
- `theory_results/INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md`.

Standalone reports retain protocol, raw/curated provenance, uncertainty, and
claim boundaries. This playbook selects the strongest directly responsive
facts and omits irrelevant internal work. It never omits a limitation whose
absence would make the selected claim misleading.

## 1. Source and status boundary

- The full payload-hashed formal review retained here is Reviewer `27bE`.
- The AC text was supplied by the author and is not independently source-hashed
  in this workspace.
- Do not assign claims to unretained `zWsa`, `Dz6s`, or “Reviewer 2” sources.
- The submitted paper already contains supporting LLaMA-3-8B LoRA evidence.
  It is explicitly listed in the evidence-tier table and reported in
  Appendix A4, Table `tab:lora-8b`. The correct response is to surface that
  exact anchor and add the matched control that the appendix itself requested,
  not accuse the reviewer of overlooking it.
- Exact-range three-seed and OLMo step-1,000 scratch results remain
  `CONDITIONAL` until their promotion conflicts are resolved.
- No completed LLaMA training uses counterfactual loss. The completed
  counterfactual-trained capability result is on OLMo-2 1.485B.

## 2. Core strategy: results first, then claim boundary

The response should lead with three concrete advances:

1. **Mature 1.485B LoRA:** 4K-only counterfactual training produces strong 8K
   probability and strict autoregressive capability, with a separate complete
   13-task task-family-adapted matrix.
2. **Mature 8B LoRA:** the submission already contained 8B supporting evidence;
   a post-submission matched Native/EVQ control confirms 16K/32K probability
   transfer, and a matched 13-family continuation gives non-zero 16K RULER
   capability for EVQ while Native collapses.
3. **Controlled scale trajectory:** after its curated/raw owner is reconciled,
   the 1.485B step-0→step-1,000 EVQ trajectory adds from-scratch scale-transfer
   evidence at the same 2.097B-token budget.

Then disclose one narrow limitation:

> The evidence now closes a meaningful 2× capability endpoint under
> task-adapted protocols; broad unseen-task transfer and reliable 4×
> autoregressive capability remain open.

This is strategically stronger than leading with internal failures. It is also
accurate: at 4×, LLaMA-8B is zero on the current 32K RULER matrix, while the
4K-trained OLMo continuation retains only a small `6.13%` 16K macro.

### 2.1 Compact result table

| Result | Training boundary | In-window | 2× | 4× | Required wording |
| --- | ---: | ---: | ---: | ---: | --- |
| Submitted LLaMA-3-8B supporting PPL, untouched Base→EVQ-LoRA | 8K | `7.42 → 9.63` | `176.3 → 21.5` at 16K | `1942.5 → 104.3` at 32K | Appendix A4, Table `tab:lora-8b`; single seed, no matched Native-LoRA control, and submitted RULER did not improve |
| OLMo-2 1.485B counterfactual LoRA natural-text NLL / PPL, Native→EVQ | 4K | `2.235/9.35 → 2.548/12.78` | `3.735/41.90 → 2.703/14.93` at 8K | `4.851/127.88 → 2.925/18.62` at 16K | Matched frequency-substrate plus LoRA co-adaptation; EVQ pays an in-window cost |
| OLMo-2 1.485B strict NIAH exact, Native vs two EVQ seeds | 4K | — | `0/100 vs 69/100, 67/100` at 8K | Initial 16K screen only `0/20 vs 1/20` | Same official NIAH generator family; disjoint rows/values; not unseen-task transfer |
| OLMo-2 1.485B complete 13-task RULER macro | 4K | `37.51%` | `21.29%` | `6.13%` | Separate single-seed task-family continuation; no matched Native continuation |
| LLaMA-3-8B matched temporal NLL / PPL, Native-LoRA→EVQ-LoRA | 8K | `1.919/6.82 → 2.309/10.07` | `4.691/108.96 → 3.181/24.07` | `6.899/991.47 → 4.851/127.91` | Single seed; ordinary LM training; probability/source-use evidence |
| LLaMA-3-8B matched 13-family official RULER macro, Native-LoRA vs EVQ-LoRA | 8K | `94.44% vs 77.60%` | `0.295% vs 14.03%` | Native partial-zero; EVQ `0%` | Same task-family supervision; not unseen-task; no solved 32K capability |
| OLMo-2 1.485B scratch Geo vs EVQ PPL | 4K | `161.19 vs 167.45` | `163.88 vs 156.87` | `182.73 vs 159.64` | `CONDITIONAL`: single trajectory and different trainer stacks; do not send until owner conflict is fixed |

### 2.2 Preferred scale-and-capability paragraph

Use after the cited worktree owners have been promoted to canonical `main`
evidence:

> We agree that the strongest controlled multi-seed evidence in the submission
> remained below 1B parameters. The scale record is nevertheless broader than
> that tier alone: Appendix A4, Table `tab:lora-8b` already reported a
> single-seed LLaMA-3-8B-Instruct LoRA experiment, where 16K/32K PPL changed
> from 176.3/1942.5 for the untouched base to 21.5/104.3 for EVQ-LoRA. We do
> not use that submitted row alone for attribution because it lacked a matched
> Native-LoRA control and did not improve RULER. We have now supplied the
> missing matched control and stronger autoregressive endpoints. First, on
> OLMo-2-0425-1B-Instruct (1.485B actual parameters), every LoRA backward pass
> was capped at 4K. Counterfactual routing changed 8K strict exact from 0/100
> for Native-LoRA to 69/100 and 67/100 for two independently trained EVQ-LoRA
> seeds, while the first matched pair changed 4K/8K/16K natural-text NLL from
> 2.235/3.735/4.851 to 2.548/2.703/2.925. A separate 4K-only
> task-family-matched continuation reached 37.5%/21.3%/6.1% macro over all 13
> RULER tasks at 4K/8K/16K. Second, in a matched seed-42
> LLaMA-3-8B-Instruct adaptation, EVQ-minus-Native-LoRA NLL was
> +0.390/-1.510/-2.048 at 8K/16K/32K. Under identical physical-8K
> 13-family supervision, Native-LoRA was stronger at 8K, while EVQ-LoRA
> retained 14.03% official macro at 16K compared with 0.295% for Native-LoRA.
> We present these as task-adapted length-transfer results: training and test
> rows are disjoint, but task generator families are shared. Reliable 4× and
> broad unseen-task capability remain open.

### 2.3 Conditional scale paragraph

Add only after the step-1,000 curated/raw conflict is resolved:

> We also trained EVQ from the public OLMo-2 step-0 initialization for the same
> 1,000 steps and 2.097B counted-token budget as the released native-RoPE
> milestone. On the same 128 PG-19 documents, EVQ pays 0.0381 NLL at the 4K
> training length but improves 8K and 16K NLL by 0.0437 and 0.1351; 122/128
> and 126/128 documents favor EVQ at those lengths. Because the EVQ and
> released controls use different trainer stacks, we treat this as
> single-trajectory same-initialization/same-recipe support, not a bitwise
> paired or multi-seed causal estimate.

## 3. Concern tracker

| ID | Direct answer | Best evidence | Remaining limitation |
| --- | --- | --- | --- |
| `R27bE.1` | Separate conditional theorem, surrogate choice, small-\(\tau\) argument, and empirical operating rule. Agree that the submitted trained-model attribution was incomplete. | Submitted theory map; fixed schedules; \(\tau\) sweep; exact-range after promotion | Small-\(\tau\) does not justify \(\tau\approx4\) exactly; Cosh is unique only for the stated surrogate |
| `R27bE.2` | Correct the “small model only” impression with the submitted evidence tiers, then lead with mature 1.485B and 8B results. | OLMo counterfactual/RULER; matched LLaMA NLL/RULER; submitted 8B/750M/MLA/video | Clean multi-seed controlled scale evidence remains below 1B; mature-model results are task-adapted and mostly single seed |
| `R27bE.3` | Concede that the historical learned-frequency comparison mixes shape, capacity, and tuning; use only fixed zero-parameter controls for attribution. | Three-seed fixed schedules and exact-range control after promotion | Historical “DAPE” row is layer-shared learnable inverse frequencies, not an audited official DAPE operator |
| `R27bE.4` | Report independently selected \(\tau\) and matched non-Cosh schedules directly. | \(\tau\) sweep; uniform/power/exponential/native/two-band controls | Cosh is effective but not empirically dominant; rule is a fallible basin prior |
| `R27bE.5` | Held-out base/head is partially answered; mature 1.485B/8B evidence strengthens scale. Add scratch only after promotion. | Held-out base/head; OLMo and LLaMA mature-model results | Held-out base/head is still 151.9M; scratch is one trajectory with trainer confound |
| `AC.1` | Concede the missing FMRoPE citation/comparison, then distinguish training-grid allocation from target range transport. | Exact-range after promotion; fixed schedules | Target-retargeted FMRoPE is stronger in the reported mean; no additive superiority |
| `AC.2` | Lead with mature 1.485B/8B probability and autoregressive evidence. | Result-first paragraph | Same-family task adaptation is not unseen-task generalization; 4× remains weak |
| `AC.3` | Reuse the R27bE.1/.4 four-layer separation and attribution experiments. | Fixed schedules, \(\tau\), exact-range | No universal Cosh or global-\(\tau\) claim |
| `AC.4` | Deliver novelty definition, direct control, and mature-model result in that order. | Exact-range, 1.485B counterfactual, 8B matched RULER | Package remains conditional until provenance gates pass |

The AC rows are based on author-supplied text without an independently retained
source hash.

## 4. Point-by-point response language

### 4.1 `AC.1`: FMRoPE and novelty

Use only after the exact-range raw promotion:

> We agree that FMRoPE is directly relevant and should have been cited and
> compared explicitly. This omission does not establish technical equivalence.
> The submitted contribution is not the prior observation of dead or
> under-utilized channels. It formulates the finite training-time RoPE grid as
> an explicit allocation problem and instantiates the surrogate solution as a
> fixed, closed-form, zero-learned-parameter inverse-CDF table before training.
> FMRoPE instead selects or retargets a geometric spectral range for a declared
> context. In a new three-seed exact-range control, sampled extrema, log-span,
> initialization, token order, optimizer, budget, and evaluation anchors match
> within every seed pair; only 30 interior frequencies differ. Fixed-range
> Cosh-minus-uniform-FMRoPE NLL is
> -0.3159/-0.1949/-0.1674 at 512/1K/2K, with 3/3 seeds agreeing. When both
> grids are target-retargeted, uniform FMRoPE is stronger in the reported
> three-seed mean. We therefore claim a separately identifiable training-time
> allocation variable, not replacement of or additive superiority over
> target-aware range transport.

Until promotion, replace the numeric exact-range paragraph with:

> Our fixed zero-parameter schedule ablations already show that changing
> interior allocation changes trained-model behavior when operator and
> optimization capacity are fixed. We are retaining the new three-seed
> exact-range aggregate as conditional evidence until its per-seed raw package
> is promoted.

### 4.2 `R27bE.1`, `R27bE.4`, `AC.3`: theory and ablations

> We agree that four epistemic layers must remain separate: the conditional
> minimizer of the stated convex surrogate, the pure-tether modeling choice,
> the small-\(\tau\) asymptotic motivation, and the empirical operating rule.
> The rule \(\tau=d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) is a practical
> basin prior, not a global trained-model optimum. We therefore added an
> independently selected \(\tau\) sweep and fixed uniform, power,
> exponential, native-span, and attention-derived schedules. The rule is close
> to the selected point in one direct sweep, but is fallible across broader
> configurations. Cosh is effective, while matched exponential and two-band
> schedules are stronger at some lengths. These results support allocation as
> a real design axis and narrow, rather than inflate, the empirical
> interpretation of the Cosh surrogate.

### 4.3 `R27bE.3`: historical “DAPE” comparison

> We agree that a learned-frequency comparison mixes allocation shape with
> learned capacity and optimization effort. We therefore do not use that row
> for shape attribution. The historical implementation is more accurately
> described as layer-shared learnable inverse frequencies. Our attribution
> instead relies on fixed, zero-learned-parameter schedule controls with the
> same positional operator, model, data order, optimizer, budget, and
> evaluation. This addresses whether fixed allocation matters; it does not
> claim a new head-to-head result against the official DAPE operator.

### 4.4 `R27bE.2`, `R27bE.5`, `AC.2`: scale and stronger evaluation

Use the result-first paragraph in §2.2, then add:

> This evidence also clarifies the boundary between language modeling and task
> capability. EVQ improves 2×/4× NLL in both mature-model studies, but we do
> not infer generation from PPL. The OLMo counterfactual arm supplies a strict
> 8K autoregressive endpoint; the LLaMA matched RULER continuation supplies a
> mature-8B 16K endpoint. Both are task-adapted rather than unseen-task
> transfer. At 4×, LLaMA-8B remains at zero RULER and the 4K-trained OLMo
> continuation remains weak, so reliable broad 4× capability is an explicit
> limitation.

This calmly corrects the scale record with a verifiable paper anchor:
Appendix A4, Table `tab:lora-8b`, also indexed by the submitted evidence-tier
table. The rebuttal then supplies the matched control and stronger capability
endpoint that the submitted appendix explicitly left open. Do not write that
the reviewer “failed to read” the 8B experiment.

### 4.5 `AC.4`: concise adjudication note

> The valid concern is that FMRoPE was omitted and the strongest controlled
> multi-seed tier was small. The requested evidence changes both parts of the
> record. Exact-range controls identify interior allocation separately from
> scalar range, while mature 1.485B and 8B matched adaptations show that the
> intervention affects both long-position probability and 2× autoregressive
> capability. We retain the deployment boundary that target-aware FMRoPE is
> stronger in the tested retargeted setting and the capability boundary that
> broad 4× transfer remains open. We therefore ask that the contribution be
> assessed as a finite-grid allocation method with bounded evidence, rather
> than as a claim of universal long-context superiority.

## 5. What to disclose and what to omit

### Always adjacent

- exact-range is not sendable until raw promotion;
- target-retargeted FMRoPE is stronger in the reported three-seed mean;
- mature-model results are single-seed or task-family matched;
- EVQ pays an in-window cost in the matched OLMo and LLaMA studies;
- LLaMA 32K RULER is zero;
- no completed LLaMA training is counterfactual;
- the 1.485B scratch comparison uses different trainer stacks.

### Omit unless directly triggered

- failed progressive morph/KL/natural-span searches;
- hybrid alias-bug history;
- residual, DC, tail, or EVQ-v2 proposals;
- band-proxy and pruning diagnostics;
- GPU utilization, compile, or remote-machine details;
- pre-submission chronology that does not answer a retained concern.

These omissions are selection, not concealment: none is needed to support the
narrow claims above. If the response expands to unseen-task transfer,
universal optimality, additive synergy, or no forgetting, the corresponding
negative guardrail in the internal ledger becomes mandatory.

## 6. One remaining score-changing experiment

The only new experiment that cleanly strengthens this strategy is the matched
LLaMA-3-8B counterfactual continuation in
`theory_results/LLAMA8B_COUNTERFACTUAL_REBUTTAL_PLAN_20260726.md`.

It starts from the existing matched seed-42 Native/EVQ RULER-family adapters
(whose chains begin at the matched LongAlpaca parents), keeps both frequency
tensors fixed, gives both arms the same 300 physical-8K counterfactual steps,
and evaluates 8K/16K/32K RULER plus source swaps and temporal NLL. It does not
use progressive morph, virtual positions, long backward passes, YaRN, or a new
EVQ mechanism.

If it passes, add one matched LLaMA counterfactual paragraph. If it fails,
retain the already valid LLaMA NLL and ordinary task-family 2× result and do
not spend GPU time inventing another retrofit path during rebuttal.

## 7. Send gate

Before external submission:

1. Promote the exact-range three-seed raw/per-seed/hash package or remove its
   numeric paragraph.
2. Reconcile the 1.485B scratch Markdown/curated-JSON/raw conflict or remove
   the scratch paragraph.
3. Correct future-dated OLMo metadata before citing those owners.
4. Commit every selected standalone owner and curated JSON to `main`.
5. Verify every number against its standalone owner after the final edit.
6. Keep submitted and post-submission evidence explicitly separated.
7. Preserve seed count, task-family status, training length, and evaluation
   endpoint in every scale claim.
8. Do not call LLaMA counterfactual-trained unless the new matched experiment
   completes.
9. Do not report a complete Native-LoRA 32K macro; three tasks are missing.
10. Keep the 4× capability limitation in the same response section as the 2×
    mature-model result.

Package status remains `needs_author_input` until these gates are resolved.
