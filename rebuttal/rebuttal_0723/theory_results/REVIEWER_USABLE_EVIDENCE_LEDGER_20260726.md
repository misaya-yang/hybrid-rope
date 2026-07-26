# Reviewer-usable evidence ledger

Last audited: 2026-07-26
Purpose: one compact index of positive evidence that can answer the retained
Reviewer 27bE and author-supplied AC concerns.

This file does not replace the standalone evidence owners. Each owner retains
the full protocol, numbers, hashes, uncertainty, and claim boundary. The
playbook should select from this ledger; it should not reconstruct results from
older summaries.

## 1. Status vocabulary

- `READY_INTERNAL`: protocol and retained evidence support internal drafting.
  If the owner is still untracked, commit/promotion remains required before the
  result is treated as canonical `main` evidence.
- `SUPPORTING`: usable only with the stated single-seed, task-family, or
  endpoint boundary.
- `CONDITIONAL`: do not quote externally until the named promotion conflict is
  resolved.
- `SUBMITTED`: evidence already present in the submitted paper; describe it at
  its original evidence tier.

## 2. Positive evidence registry

| ID | Concern | Result | Status | Mandatory adjacent boundary | Evidence owner |
| --- | --- | --- | --- | --- | --- |
| `E-SHAPE-3S` | `R27bE.1/.3/.4`, `AC.3` | In the 151.9M fixed-schedule study, EVQ improves over Paper-Geo at 1K/2K/4K/8K by `-0.256/-0.305/-0.223/-0.238` mean tail NLL, with 3/3 seeds in the same direction. Native-endpoint controls also preserve a positive allocation effect. | `READY_INTERNAL` | Matched exponential and an attention-derived two-band schedule are stronger at some lengths. This supports allocation as a design variable, not universal Cosh optimality. | `EXPERIMENT_REPORT_20260724.md` §§3,7 and `native_attention_shape_l128_results_20260724.json` |
| `E-TAU` | `R27bE.1/.4`, `AC.3` | In one independently selected sweep, \(\tau=5\) is selected and the rule value \(5.657\) is within `0.0119` NLL. | `READY_INTERNAL` | Other configurations show that the rule is fallible. It is a practical basin prior, not a trained-model optimum or theorem. | `EXPERIMENT_REPORT_20260724.md` §2 and `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` |
| `E-HELDOUT` | `R27bE.2/.5`, `AC.2` | At held-out base \(1\)M and \(d_{\mathrm{head}}=128\), EVQ-minus-Paper-Geo NLL is `-0.8018/-0.6640/-0.4329/-0.2871/-0.2117` at 1K/2K/4K/8K/16K, with 3/3 seeds agreeing. | `CONDITIONAL` | EVQ is worse at the 512-token training length by `+0.0694` NLL; base, head dimension, and training length change together; the checkout lacks a dedicated raw/per-seed aggregate owner. | `EXPERIMENT_REPORT_20260724.md` §4 |
| `E-EXACT-RANGE-3S` | `AC.1/.3`, `R27bE.3/.4` | With sampled extrema and log-span matched and only 30 interior frequencies changed, fixed-range Cosh-minus-uniform-FMRoPE is `-0.3159/-0.1949/-0.1674` NLL at 512/1K/2K; 3/3 training seeds favor Cosh. | `CONDITIONAL` | The aggregate is author-confirmed but lacks promoted per-seed raw values, hashes, and training-seed CI. Target-retargeted FMRoPE is stronger in the reported three-seed mean. The arm is endpoint-normalized Cosh, not submitted raw midpoint EVQ. | `MATCHED_RANGE_COSH_500M_3SEED_20260724.md` and `matched_range_cosh_500m_3seed_result_20260724.json` |
| `E-OLMO-CF` | `R27bE.2/.5`, `AC.2` | On OLMo-2-0425-1B-Instruct (1.485B actual parameters), every backward pass is capped at 4K. Counterfactual rank-64 Q/K/V/O LoRA gives 8K `niah_single_1` strict exact `0/100` for Native-LoRA versus `69/100` and `67/100` for two EVQ-LoRA training seeds. Natural-text NLL at 4K/8K/16K is `2.235/3.735/4.851` for Native and `2.548/2.703/2.925` for the first EVQ seed. | `READY_INTERNAL` | Training and evaluation use the same official NIAH generator family; values and rows are disjoint, but this is task-family-matched capability conversion. The 8K aggregate contains both within- and beyond-training-gap examples. Held-out UUID/VT transfer is negative. | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| `E-OLMO-LONG-GAP` | `R27bE.2/.5`, `AC.2` | On a fresh 8K set where every source-to-generation gap exceeds the 4K routing-training support, Native is `0/100` and the same two EVQ adapters are `49/100` and `48/100` strict exact. | `CONDITIONAL` | The report filename is dated after the current audit date and must be metadata-corrected before external use. It remains the same NIAH task family and does not establish full RULER or unseen-task transfer. | `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md` |
| `E-OLMO-RULER-FAMILY` | `R27bE.2/.5`, `AC.2` | A separate single-seed continuation, still using only physical 4K backward passes, reaches official 13-task RULER macro `37.51%/21.29%/6.13%` at 4K/8K/16K; at 8K it reaches VT `31%`, CWE `22%`, and FWE `60%`. | `SUPPORTING` | The continuation uses the same RULER generator families and ordinary answer-only CE, not counterfactual loss. There is no matched Native continuation, QA remains low, and the result is not pure EVQ attribution. | `OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md` and curated JSON |
| `E-LLAMA8B-PROB` | `R27bE.2/.5`, `AC.2` | In a matched seed-42 LLaMA-3-8B-Instruct LongAlpaca adaptation, EVQ-minus-Native-LoRA NLL is `+0.390/-1.510/-2.048` at 8K/16K/32K. At true 16K, target-block hit@16 changes `18.75%→64.06%`, gold-block deletion changes EVQ NLL by `+1.5055`, and median correct-token rank improves `33,774.5→2,043`. | `SUPPORTING` | Training is ordinary full-token LM, not counterfactual. Exact generation remains zero and QA macro F1 is `0.1126` for EVQ versus `0.2110` for Native. The contrast combines endpoint/midpoint quantization, Cosh, and LoRA co-adaptation. | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| `E-LLAMA8B-RULER` | `R27bE.2/.5`, `AC.2` | With identical physical-8K, 13-family continuation, Native-LoRA/EVQ-LoRA official macro is `94.44%/77.60%` at 8K but `0.295%/14.03%` at 16K. Normalized exact is `17.69%/21.54%` at 8K and `0%/1.54%` at 16K. | `SUPPORTING` | Single seed and explicitly task-family supervised. EVQ is worse on the 8K official macro. EVQ and untouched Native are zero across all 13 tasks at 32K; Native-LoRA completed only 10/13 32K tasks and those ten are zero. Training is not counterfactual. | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` and curated JSON |
| `E-OLMO-SCRATCH` | `R27bE.2/.5`, `AC.2/.4` | In one 1.485B same-initialization/same-recipe step-1,000 comparison at a 2.097B-token budget, Geo/EVQ PPL is `161.19/167.45` at 4K, `163.88/156.87` at 8K, and `182.73/159.64` at 16K. EVQ-minus-Geo NLL is `+0.0381/-0.0437/-0.1351`; `122/128` and `126/128` documents favor EVQ at 8K/16K. | `CONDITIONAL` | EVQ used a reviewed HF single-GPU loop; the released Geo checkpoint used AI2's distributed trainer. The current curated JSON still states that no EVQ result is included, and local raw EVQ outputs are unavailable for independent recomputation. This is single-trajectory LM evidence, not capability or pure interior attribution. | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`; curated owner requires reconciliation |
| `E-SUBMITTED-BREADTH` | `R27bE.2/.5`, `AC.2` | The submitted evidence tiers include 454M three-seed MHA, 432M three-seed MLA, 750M continued pretraining, 129M/382M video DiT, progressive training, QuALITY gold-answer NLL, and LLaMA-3-8B LoRA. In Appendix A4, Table `tab:lora-8b`, untouched-Base/EVQ-LoRA PPL is `7.42/9.63`, `176.3/21.5`, and `1942.5/104.3` at 8K/16K/32K. | `SUBMITTED` | The strongest controlled multi-seed tier remains below 1B. The submitted 8B row is single seed, lacks a matched Native-LoRA control, and did not improve RULER; supporting rows cannot be pooled into production-scale or downstream closure. | Submitted evidence-tier table and Appendix A4, Table `tab:lora-8b` |

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

The following sentence is **conditional** and must remain out of a submitted
response until `E-OLMO-SCRATCH` is reconciled:

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
- No completed LLaMA training arm uses counterfactual loss. LLaMA gold-block
  deletion and frequency/source swaps are evaluation interventions.

## 5. Current promotion gates

1. Promote exact-range per-seed raw values, hashes, and training-seed
   uncertainty before quoting `E-EXACT-RANGE-3S` externally.
2. Add a dedicated held-out base/head raw or curated aggregate before treating
   `E-HELDOUT` as portable reviewer evidence.
3. Reconcile the OLMo step-1,000 Markdown with its curated JSON and recover or
   re-verify raw EVQ results before using `E-OLMO-SCRATCH`.
4. Correct future-dated OLMo report metadata before external use.
5. Commit the selected standalone owners, curated JSONs, playbook, and these
   ledgers to `main`; untracked worktree artifacts are not canonical evidence.
6. Do not describe LLaMA as counterfactual-trained unless that separate
   matched experiment is actually completed and admitted.
