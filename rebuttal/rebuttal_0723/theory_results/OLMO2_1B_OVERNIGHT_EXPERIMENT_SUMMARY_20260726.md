# OLMo-2 1B overnight experiment summary

Status: **consolidated evidence ledger; no new experiment claim**  
Primary concerns: `R27bE.2`, `R27bE.5`, `AC.2`

## Executive conclusion

This experiment cycle did substantially more than produce one final table. It
mapped the full probability-to-capability failure, rejected seven plausible
training fixes, completed multiple full RULER matrices, and finally found a
4K-only task-adaptation recipe that produces measurable 8K/16K
autoregressive capability.

The result is useful but narrower than a method-superiority claim:

- EVQ adapters can learn held-out-identity long-range NIAH and can acquire
  broader RULER task capability using only physical 4K backward passes.
- The final task-adapted EVQ model improves the complete 39-cell RULER macro
  from 14.72% for the prior two-seed EVQ-adapter mean to 21.65%.
- The strongest new task results are FWE
  66.67%/60%/50%, VT 68%/31%/0%, and CWE 38%/22%/1% at
  4K/8K/16K.
- QA remains weak, clean unseen-task transfer remains negative, and the final
  EVQ adapter does not exceed the untouched Native/official-YaRN
  length-matched controls in aggregate.
- No matched Native adapter received the final task-family continuation, so
  the final gain identifies a successful training recipe around EVQ, not a
  pure EVQ-versus-Native causal advantage.

## 1. What was completed

| Package | Compute/evaluation completed | Outcome | Rebuttal role |
| --- | --- | --- | --- |
| Fresh all-long-gap NIAH | Native plus two EVQ training seeds; 300 fresh 8K autoregressive generations | Native 0/100; EVQ 49/100 and 48/100 | Positive benchmark-family-matched 2x capability conversion |
| Clean LongAlign+Tulu EVQ | One complete 13-task x 3-length x 20 matrix, 780 generations | 9.74%/4.01%/2.05% at 4K/8K/16K | Negative guardrail: low NLL alone does not create task capability |
| Seven non-RULER training arms | Seven arms x 1,500 optimizer steps = 10,500 4K steps; 560 fixed RULER-screen generations | Best screen only 11.67%/7.50% at 4K/8K | Rejects morphing, sparse teacher KL, answer-only LongAlign, and tested natural-retrieval curricula |
| Legacy EVQ adapters, complete RULER | Two independent training seeds x 780 generations = 1,560 | Mean 24.89%/15.19%/4.08% | Establishes the complete-suite baseline for NIAH-family adapters |
| Final task-family continuation | 276 optimizer steps, 9.04M physical 4K input tokens, then 780 generations | 37.51%/21.29%/6.13%; 39-cell macro 21.65% | Positive task-adapted capability and length-transfer result |
| Official-YaRN/operator checks | Untouched length-matched controls are promoted; additional adapter-composition runs completed remotely | Untouched control remains substantially stronger in aggregate | Prevents false claims that inference-only scaling automatically repairs a damaged adapter |
| Weak repository-ramp detour | Canary plus partial matrix stopped after 4.67 GPU-minutes | Stopped before a full rerun | GPU stop-loss; not admitted as evidence |

The locally documented packages therefore account for at least:

- **10,776 new optimizer steps** from the seven-arm search and final positive
  continuation, excluding earlier parent-adapter training and the clean
  Tulu pass;
- **3,980 autoregressive generations** from the fresh n=100 NIAH set, clean
  full matrix, seven-arm screens, two legacy full matrices, and final full
  matrix;
- additional official-YaRN/operator generations retained on the experiment
  volume but not promoted into this numeric count.

These counts describe executed work, not independent evidence. Many runs are
diagnostics or negative results.

## 2. Unified complete-RULER view

All values are official task-specific RULER macro scores. The first row uses
the already completed length-matched untouched control: Native at 4K,
official Transformers YaRN factor 2 at 8K, and factor 4 at 16K. It is an
absolute reference, not a matched-adaptation comparison.

| Model / adaptation | 4K | 8K | 16K | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Untouched Native / official-YaRN length-matched reference | **65.35%** | **58.72%** | **9.08%** | Strong pretrained capability; no EVQ conversion |
| Clean EVQ: LongAlign + Tulu, no explicit RULER rows | 9.74% | 4.01% | 2.05% | Clean-transfer negative |
| Legacy EVQ NIAH-style adapters, two-seed mean | 24.89% | 15.19% | 4.08% | Narrow task-family competence |
| Final EVQ 4K task-family adaptation, one continuation seed | **37.51%** | **21.29%** | **6.13%** | Best EVQ complete-matrix result in this cycle |

The final continuation improves over the legacy EVQ mean by
\(+12.63/+6.10/+2.06\) percentage points at 4K/8K/16K. Its overall
39-cell macro is 21.65%, compared with 14.72% for the legacy two-seed mean,
an absolute gain of 6.93 points and an approximately 47% relative gain.

It nevertheless remains below the untouched length-matched reference. That
gap cannot be hidden or interpreted away: the useful positive statement is
that short task adaptation creates EVQ capability beyond the training length,
not that the resulting adapter is the strongest model in absolute RULER
accuracy.

## 3. Final task-level result

| Task | Legacy EVQ mean 4K / 8K / 16K | Final EVQ 4K / 8K / 16K | Main change |
| --- | ---: | ---: | --- |
| NIAH single 1 | 100 / 62.5 / 0 | 100 / 45 / 0 | Some 8K regression |
| NIAH single 2 | 40 / 12.5 / 0 | 50 / 20 / 0 | Improvement |
| NIAH single 3 | 5 / 0 / 0 | 10 / 0 / 0 | Remains weak |
| NIAH multikey 1 | 45 / 27.5 / 5 | 65 / 35 / 5 | Improvement |
| NIAH multikey 2 | 22.5 / 10 / 0 | 10 / 5 / 0 | Regression |
| NIAH multikey 3 | 0 / 0 / 0 | 0 / 0 / 0 | Unsolved |
| NIAH multivalue | 18.13 / 10.63 / 2.5 | 22.5 / 17.5 / 1.25 | Better to 8K |
| NIAH multiquery | 11.25 / 9.38 / 2.5 | 17.5 / 6.25 / 2.5 | Mixed |
| VT | 0 / 0 / 0 | **68 / 31 / 0** | New 4K/8K capability |
| CWE | 0 / 0 / 0.5 | **38 / 22 / 1** | New 4K/8K capability |
| FWE | 39.17 / 30 / 27.5 | **66.67 / 60 / 50** | Strongest robust gain |
| SQuAD QA | 15 / 15 / 5 | 15 / 10 / 10 | Essentially unchanged |
| HotpotQA | 27.5 / 20 / 10 | 25 / 25 / 10 | Essentially unchanged |

The result is not a uniform solver. The aggregate gain is driven primarily by
VT, CWE, and FWE. QA is nonzero but remains too weak for a downstream-QA
headline. Sixteen-k capability is sharply task-dependent: FWE reaches 50%,
while VT remains zero.

## 4. What the negative search established

The seven-arm search is not reviewer-facing positive evidence, but it explains
why the final intervention is meaningful.

| Arm | Training hypothesis | Natural-text NLL, 4K / 8K / 16K | RULER screen, 4K / 8K | Decision |
| --- | --- | --- | --- | --- |
| A1 | Linear Native-to-EVQ morph; LongAlign full-token + Tulu | 3.0565 / 3.2559 / 3.4810 | 11.67% / 6.67% | Reject |
| A2 | A1 + sparse Native-teacher KL | 3.1331 / 3.3324 / 3.5410 | 8.33% / 4.17% | Reject |
| B1 | Linear morph; answer-only LongAlign + Tulu | 3.5799 / 3.7782 / 3.9843 | 4.17% / 2.50% | Reject |
| B2 | Immediate EVQ; same answer-only data | 3.3106 / 3.4750 / 3.6477 | 10.00% / 7.50% | Reject |
| C1 | Linear morph; independent natural-span retrieval | 3.9914 / 4.3305 / 4.6596 | 0 / 0 | Reject |
| C2 | Immediate EVQ; same natural-span curriculum | 3.4749 / 3.7542 / 4.0637 | 1.67% / 0 | Reject |
| D1 | Immediate EVQ; one-token natural retrieval | 4.7408 / 6.0438 / 7.6042 | 0 / 0 | Reject |

Consequences:

1. Gradually morphing the frequency operator does not preserve broad model
   capability under the tested LoRA budget.
2. Sparse Native-teacher KL does not repair the problem.
3. Natural-text NLL can remain finite while autoregressive task capability is
   near zero.
4. Generic or natural retrieval supervision is not interchangeable with
   learning the required task computations.
5. The successful final arm changes the missing variable directly:
   task-family coverage, while replaying NIAH and natural instruction rows.

## 5. The training recipe that worked

The final continuation starts from the stronger seed-20260725 EVQ NIAH
adapter and keeps every backward pass at physical length 4,096:

- 480 official-generator training rows: 96 each for VT, CWE, FWE, SQuAD QA,
  and HotpotQA;
- 128 paired NIAH replay rows;
- 128 LongAlign assistant replay rows;
- 20 held-out internal validation rows;
- rank-64/alpha-128 Q/K/V/O LoRA;
- three deterministic passes, 276 steps, global batch 8;
- BF16, Flash-only SDPA, fused AdamW, and TorchInductor
  `max-autotune-no-cudagraphs`.

Exact serialized training/evaluation row overlap is zero. The internal
20-row validation NLL falls from 3.0372 to 0.9881. The formal evidence is the
separate 780-generation matrix, not this internal NLL.

Training took 431.6 seconds including first compilation. Steady-state
throughput was approximately 30.3K physical token/s on RTX 5090, with 100%
observed GPU utilization and a 23.4GB peak allocation.

## 6. Reviewer-facing synthesis

### Strongest concise statement

> We evaluated capability conversion in a mature approximately 1B Instruct
> model using only 4K adaptation. On a fresh 8K NIAH set whose source-to-query
> gaps all exceed the 4K training support, Native-LoRA is 0/100 while two
> EVQ-LoRA training seeds reach 49/100 and 48/100 strict autoregressive exact.
> When a separate EVQ continuation is explicitly adapted to the complete
> RULER task families using only physical 4K sequences, it reaches
> 37.5%/21.3%/6.1% macro score at 4K/8K/16K, including 60% FWE, 31% VT, and
> 22% CWE at 8K. Training and evaluation rows are disjoint, but generator
> families are shared; we therefore claim task-adapted length transfer, not
> unseen-task generalization or pure EVQ attribution.

### What this answers

- EVQ is not limited to a small from-scratch model or teacher-forced PPL.
- A mature model can convert short-context EVQ adaptation into nonzero
  autoregressive behavior beyond the adaptation length.
- The behavior is reproduced across two EVQ training seeds on the strongest
  same-task NIAH endpoint.
- With explicit task-family coverage, capability extends beyond NIAH to
  counting, variable tracking, and frequency extraction.

### What must not be claimed

- EVQ universally beats Native or official YaRN on RULER.
- The final result is unseen-task or benchmark-independent transfer.
- The final single-seed continuation establishes statistical significance.
- QA or general downstream capability is solved.
- Natural-text NLL improvement alone proves usable long-context capability.
- The final experiment isolates pure frequency allocation: LoRA and
  task-family co-adaptation are part of the intervention.

## 7. Remaining evidence gaps

1. **Matched task-adaptation control.** Native plus the identical final
   continuation mixture was not run. Without it, the final matrix cannot
   attribute the gain uniquely to EVQ.
2. **Absolute control gap.** The final EVQ adapter remains below untouched
   Native/official-YaRN aggregate scores.
3. **QA.** Both QA tasks remain low and do not provide a persuasive downstream
   headline.
4. **Continuation-seed uncertainty.** The final broad continuation has one
   seed, although its parent NIAH behavior has two EVQ seeds.
5. **Clean transfer.** Training without explicit RULER/NIAH rows remains
   negative under all seven tested objectives.
6. **Official-YaRN adapter compositions.** These runs were completed on the
   experiment volume, but their raw packages were not promoted locally before
   shutdown. Do not quote their exact values as formal evidence until the
   retained result JSONs are copied and hashed.

## 8. Next decisions, not automatic GPU authorization

The highest-value optional follow-ups are:

1. one matched Native continuation on the identical fixed task-family view,
   if pure EVQ attribution becomes essential;
2. one second EVQ continuation seed, if uncertainty around the 21.65%
   39-cell macro becomes a reviewer issue;
3. no further generic retrieval, morphing, sparse-KL, or answer-only sweeps
   without a new falsifiable mechanism.

The completed evidence is already sufficient to state a narrow positive
capability result. Further GPU use should be triggered by the exact rebuttal
wording required, not by open-ended optimization.

## 9. Evidence index

- Same-task NIAH conversion:
  `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md`
- Fresh all-long-gap confirmation:
  `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`
- Gap decomposition:
  `OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md`
- Clean full-RULER negative:
  `OLMO2_1B_CLEAN_4K_TULU_FULL_RULER_20260728.md`
- Seven-arm negative search:
  `OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md`
- Final positive full-matrix continuation:
  `OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`
- Curated final numeric record:
  `olmo2_1b_4k_ruler_family_adaptation_20260726.json`
- Reviewer-facing selection and boundaries:
  `../01_REBUTTAL_PLAYBOOK.md`

No paper file was modified by this experiment cycle.
