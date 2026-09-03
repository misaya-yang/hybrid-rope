# Reviewer-usable evidence ledger

Last audited: 2026-07-26
Purpose: one compact index of positive evidence that can answer the retained
AC and three-reviewer panel.

This file does not replace the standalone evidence owners. Each owner retains
the full protocol, numbers, hashes, uncertainty, and claim boundary. The
playbook should select from this ledger; it should not reconstruct results from
older summaries.

## 1. Status vocabulary

- `READY_INTERNAL`: protocol and retained evidence support internal drafting.
  If the owner is still untracked, commit/promotion remains required before the
  result is treated as canonical `main` evidence.
- `POST_SUB_RAW_HASH_BACKED`: completed post-submission evidence with a
  standalone protocol/result owner and retained raw or artifact hashes; usable
  externally with the stated endpoint, seed, and control boundary.
- `POST_SUB_WORKSTATION_RAW_BACKED`: completed post-submission evidence with a
  standalone report here and raw/per-run owners retained on the author
  workstation; usable after the selected numbers are checked once against
  that owner.
- `SUPPORTING`: usable only with the stated single-seed, task-family, or
  endpoint boundary.
- `CONDITIONAL`: do not quote externally until the named promotion conflict is
  resolved.
- `SUBMITTED`: evidence already present in the submitted paper; describe it at
  its original evidence tier.

## 2. Positive evidence registry

| ID | Concern | Result | Status | Mandatory adjacent boundary | Evidence owner |
| --- | --- | --- | --- | --- | --- |
| `E-M4-EXACT-FACTORIAL` | `R27bE.1/.2/.3/.4`, `AC.1/.3`, `RzWsa.1/.2` | In a three-seed factorial over two bases, two training lengths, and three head dimensions, all schedules share exactly the sampled extrema and log-span. Formula-Cosh, \(1.25\times\) Cosh, and deformation-matched exponential minus uniform Geo weighted \(2\times/4\times/8\times\) OOD NLL are `-0.009879/-0.012100/-0.010619`; \(1.25\times\) Cosh favors 10/12 configurations. | `POST_SUB_WORKSTATION_RAW_BACKED` | This is a 50.9M short-budget mechanistic NLL control, not scale or capability. Formula-Cosh and the matched exponential are not empirically separated; the best Cosh multiplier varies across configurations. | `M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` |
| `E-SHAPE-3S` | `R27bE.1/.3/.4`, `AC.3` | In the 151.9M fixed-schedule study, EVQ improves over Paper-Geo at 1K/2K/4K/8K by `-0.256/-0.305/-0.223/-0.238` mean tail NLL, with 3/3 seeds in the same direction. Native-endpoint controls also preserve a positive allocation effect. | `READY_INTERNAL` | Matched exponential and an attention-derived two-band schedule are stronger at some lengths. This supports allocation as a design variable, not universal Cosh optimality. | `EXPERIMENT_REPORT_20260724.md` §§3,7 and `native_attention_shape_l128_results_20260724.json` |
| `E-TAU` | `R27bE.1/.4`, `AC.3` | In one independently selected sweep, \(\tau=5\) is selected and the rule value \(5.657\) is within `0.0119` NLL. | `READY_INTERNAL` | Other configurations show that the rule is fallible. It is a practical basin prior, not a trained-model optimum or theorem. | `EXPERIMENT_REPORT_20260724.md` §2 and `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` |
| `E-HELDOUT` | `R27bE.2/.5`, `AC.2` | At held-out base \(1\)M and \(d_{\mathrm{head}}=128\), EVQ-minus-Paper-Geo NLL is `-0.8018/-0.6640/-0.4329/-0.2871/-0.2117` at 1K/2K/4K/8K/16K, with 3/3 seeds agreeing. | `CONDITIONAL` | EVQ is worse at the 512-token training length by `+0.0694` NLL; base, head dimension, and training length change together; the checkout lacks a dedicated raw/per-seed aggregate owner. | `EXPERIMENT_REPORT_20260724.md` §4 |
| `E-EXACT-RANGE-S42` | `AC.1/.3`, `RzWsa.1/.2`, `R27bE.3/.4` | With sampled extrema and log-span exactly matched and only the 30 interior frequencies changed, fixed-range Cosh-minus-uniform-FMRoPE NLL is `-0.47750/-0.20499/-0.11284` at 512/1K/2K. | `POST_SUB_RAW_HASH_BACKED` | Single seed, 151.9M, endpoint-normalized Cosh. When both schedules are target-retargeted, uniform FMRoPE is stronger. This identifies an interior-allocation effect; it does not establish FMRoPE replacement, additivity, or multi-seed superiority. | `MATCHED_RANGE_COSH_500M_S42_20260724.md` |
| `E-EXACT-RANGE-3S` | `AC.1/.3`, `RzWsa.1/.2`, `R27bE.3/.4` | With sampled extrema and log-span matched and only 30 interior frequencies changed, fixed-range Cosh-minus-uniform-FMRoPE is `-0.3159/-0.1949/-0.1674` NLL at 512/1K/2K; 3/3 training seeds favor Cosh. | `CONDITIONAL` | The aggregate is author-confirmed but lacks promoted per-seed raw values, hashes, and training-seed CI. Target-retargeted FMRoPE is stronger in the reported three-seed mean. The arm is endpoint-normalized Cosh, not submitted raw midpoint EVQ. | `MATCHED_RANGE_COSH_500M_3SEED_20260724.md` and `matched_range_cosh_500m_3seed_result_20260724.json` |
| `E-DAPE-TUNING` | `R27bE.3`, `AC.3` | The submitted DAPE row received a dedicated positional-parameter learning-rate sweep at `10x` and `100x`; the better `100x` setting was reported (`455.3` PPL@8K versus `477.7` at `10x`, both seed 42). | `SUBMITTED` | This answers the tuning-budget question. Learned capacity remains a shape-attribution confound, so allocation-shape attribution uses fixed zero-parameter schedules under the same operator and training protocol. | `docs/exp/2026-02/2026-02-24_128tok_baseline_report.md` §4 and the submitted PE-dominant table |
| `E-OLMO-CF` | `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2` | On OLMo-2-0425-1B-Instruct (1.485B actual parameters), every backward pass is capped at 4K. In the matched Native/EVQ training seed, counterfactual rank-64 Q/K/V/O LoRA gives 8K `niah_single_1` strict exact `0/100` versus `69/100`; a second independently trained EVQ seed gives `67/100` on the same evaluation rows. Natural-text NLL at 4K/8K/16K is `2.235/3.735/4.851` for Native and `2.548/2.703/2.925` for the matched EVQ seed. | `POST_SUB_RAW_HASH_BACKED` | Training and evaluation rows and values are disjoint but use the same official NIAH generator family. The two EVQ results are training-seed stability on shared test rows, not `136/200`; NLL and strict autoregressive exact remain separate endpoints. | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| `E-OLMO-QUERY-GAP-FINAL` | `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2` | On the same 1.485B checkpoint, the final EVQ query-gap +100 plus answer-and-EOS +32 chain obtains literal complete answer-string exact plus terminal EOS of `100/100`, `98/100`, and `60/100` at 4K/8K/16K. The matched Native parent under the same +100 and +32 downstream protocol obtains `95/100`, `18/100`, and `0/100`. Every backward pass uses at most 4K physical tokens. | `POST_SUB_RAW_HASH_BACKED` | Training explicitly exposes target-range relative phases through position IDs; this is not phase-free 4K training. One seed and the same numeric NIAH family, not clean unseen-task transfer. The 16K far-gap subset is `31/66`; the EVQ 13-family 4K retention matrix has a localized `niah_single_2` drop from `0.70` to `0.55`. | `EVQ_QUERY_GAP_FINAL_DIAGNOSTIC.md` |
| `E-OLMO-LONG-GAP` | `R27bE.2/.5`, `AC.2` | On a fresh 8K set where every source-to-generation gap exceeds the 4K routing-training support, Native is `0/100` and the same two EVQ adapters are `49/100` and `48/100` strict exact. | `CONDITIONAL` | The report filename is dated after the current audit date and must be metadata-corrected before external use. It remains the same NIAH task family and does not establish full RULER or unseen-task transfer. | `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md` |
| `E-OLMO-RULER-FAMILY` | `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2` | With identical physical-4K, 13-family continuation, Native/EVQ official macro is `82.16%/37.51%` at 4K, `0.08%/21.29%` at 8K, and `0%/6.13%` at 16K. Native ends with the lower supervised validation NLL, while EVQ retains the 2×/4× capability. | `POST_SUB_RAW_HASH_BACKED` | One continuation seed per arm and explicit RULER-family supervision. This is task-adapted length transfer, not unseen-task transfer or pure interior-shape attribution; the large Native 4K advantage must remain adjacent. | `OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md` |
| `E-LLAMA8B-PROB` | `RDz6s.1`, `RzWsa.4`, `R27bE.2/.5`, `AC.2` | In a matched seed-42 LLaMA-3-8B-Instruct ordinary-LM adaptation, EVQ-minus-Native-LoRA NLL is `+0.390/-1.510/-2.048` at 8K/16K/32K. | `SUPPORTING` | Single-seed teacher-forced probability evidence, not generation or RULER. This 300-step protocol is separate from the later 516-step RULER-family continuation. | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| `E-LLAMA8B-RULER` | `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2` | With identical physical-8K, 13-family continuation, Native-LoRA/EVQ-LoRA official macro is `94.44%/77.60%` at 8K but `0.295%/14.03%` at 16K. Normalized exact is `17.69%/21.54%` at 8K and `0%/1.54%` at 16K. | `SUPPORTING` | One continuation seed and task-family supervision. This supports the tested 2x endpoint, not unseen-task transfer or pure Cosh attribution. At 32K EVQ is zero across all 13 tasks; Native-LoRA completed only 10/13 and all ten are zero. | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` and curated JSON |
| `E-OLMO-SCRATCH` | `RzWsa.4`, `R27bE.2/.5`, `AC.2/.4` | In one 1.485B same-initialization/same-recipe step-1,000 comparison at a 2.097B-token budget, Geo/EVQ PPL is `161.19/167.45` at 4K, `163.88/156.87` at 8K, and `182.73/159.64` at 16K. EVQ-minus-Geo NLL is `+0.0381/-0.0437/-0.1351`; `122/128` and `126/128` documents favor EVQ at 8K/16K. | `POST_SUB_RAW_HASH_BACKED` | One early-training trajectory and a natural-text LM endpoint, not capability or multi-seed uncertainty. EVQ used a reviewed HF single-GPU loop and released Geo used AI2's distributed trainer; call it same-initialization/same-scientific-recipe, not bitwise paired. The sibling JSON is intentionally native-only and is not the owner of the later paired result. | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| `E-SUBMITTED-BREADTH` | `RDz6s.1/.2`, `RzWsa.4`, `R27bE.2/.5`, `AC.2` | The submitted evidence tiers include 454M three-seed MHA, 432M three-seed MLA, a 750M continuation with 8K strict autoregressive exact `0%→77.5%`, 129M/382M video DiT, and LLaMA-3-8B LoRA. In submitted Appendix D, Table 23, untouched-Base/EVQ-LoRA PPL is `7.42/9.63`, `176.3/21.5`, and `1942.5/104.3` at 8K/16K/32K. | `SUBMITTED` | Keep each tier separate. The 8B row is a single-seed unmatched PPL scale anchor; the 750M row is a single-seed task-specific supporting observation; video is cross-modal scope evidence. | Submitted evidence-tier table; Appendix D, Tables 3, 12, 14, 18, and 23 |

## 3. Result-first scale paragraph

This is the preferred internal draft for `R27bE.2/.5` and `AC.2`. It may be
used only after every cited owner is promoted to canonical `main` evidence.

> We additionally evaluated mature 1.485B- and 8B-parameter models. On
> OLMo-2-0425-1B-Instruct (1.485B actual parameters), every LoRA backward pass
> was capped at 4K. Matched counterfactual routing gave 8K strict
> autoregressive exact of 0/100 for Native-LoRA versus 69/100 and 67/100 for
> two independently trained EVQ-LoRA seeds; the same adapters changed
> 4K/8K/16K natural-text NLL from 2.235/3.735/4.851 for Native to
> 2.548/2.703/2.925 for the first EVQ seed. A separate 4K-only,
> RULER-family-matched continuation reached 37.5%/21.3%/6.1% macro over all
> 13 tasks at 4K/8K/16K. On LLaMA-3-8B-Instruct, the matched seed-42
> adaptation changed EVQ-minus-Native-LoRA NLL by
> +0.390/-1.510/-2.048 at 8K/16K/32K. Under identical physical-8K
> 13-family adaptation, EVQ-LoRA retained 14.03% official macro at 16K,
> compared with 0.295% for Native-LoRA. These are task-adapted length-transfer
> results: the synthetic training and test rows are disjoint, but generator
> families are shared, and neither experiment establishes unseen-task or
> universal downstream superiority.

The following additional sentence is reviewer-usable:

> Starting from the public OLMo-2 step-0 initialization, a separate
> same-recipe 2.097B-token EVQ trajectory pays 0.0381 NLL at the 4K training
> length but improves 8K and 16K NLL by 0.0437 and 0.1351 on the same 128
> PG-19 documents relative to the released native-RoPE step-1,000 checkpoint.

## 4. What “counterfactual” means here

- The OLMo routing stage is genuinely counterfactual-trained: the loss
  combines answer CE with a paired gold-versus-alternate source margin.
- The OLMo full 13-task continuation inherits that parent adapter but uses
  ordinary answer-only CE. Do not call the continuation itself
  counterfactual-loss training.
- A fresh single-seed EVQ-only LLaMA arm now uses counterfactual loss, but no
  matched Native/EVQ LLaMA counterfactual pair exists. The matched LLaMA
  natural-LM and RULER studies use ordinary full-token LM and answer-only
  task-family supervision, respectively.

## 5. Current promotion gates

1. Promote the older 151.9M exact-range per-seed raw values, hashes, and
   training-seed uncertainty before quoting `E-EXACT-RANGE-3S` externally;
   use `E-M4-EXACT-FACTORIAL` for the core multi-configuration attribution.
2. Add a dedicated owner before treating the older `E-HELDOUT` result as
   portable reviewer evidence; use the M4 factorial for the core two-base,
   multi-head-dimension mechanism claim.
3. Correct future-dated OLMo report metadata before external use.
4. Commit the selected standalone owners, curated JSONs, playbook, and these
   ledgers to `main`; untracked worktree artifacts are not canonical evidence.
5. Do not describe the matched LLaMA Native/EVQ natural-LM or RULER studies as
   counterfactual-trained; only the separate fresh EVQ-only arm used that loss.
