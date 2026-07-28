# OLMo-2 1.485B Selective Q/K Phase Adaptation

Date: 2026-07-29
Status: `POST_SUB_RAW_HASH_BACKED`
Concerns: `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2`

## Direct answer

The large 4K loss in the earlier full Q/K/V/O task continuation is not
inevitable on multi-hop QA, but it is only partially recoverable across the
full RULER suite with this intervention. Starting independently from the
matched Native and EVQ Stage-A parents, freezing every inherited V/O LoRA
tensor, and continuing only the existing Q/K LoRA tensors recovers nearly
matched 4K multi-hop QA while retaining a large EVQ advantage at 8K and
non-zero EVQ capability at 16K. On 13-family RULER, it raises EVQ's 4K macro
from `11.09%` to `42.44%`, but Native remains higher at `72.19%`.

This is a continuation of a Q/K/V/O Stage-A adapter, not a fresh pure-Q/K
adapter. It keeps the original model weights frozen and verifies inherited V/O
LoRA tensors bitwise before and after training.

## Intervention

Both arms use:

- `OLMo-2-0425-1B-Instruct` (1.485B parameters);
- independently matched Native and EVQ Stage-A parents;
- fixed Native or EVQ frequency substrate throughout the continuation;
- rank-64, alpha-128 LoRA inherited from Stage A;
- exactly 8,388,608 trainable Q/K LoRA parameters;
- exactly 8,388,608 frozen inherited V/O LoRA parameters;
- 300 optimizer steps, global batch 8, fused AdamW, learning rate `5e-5`,
  20 warmup steps, BF16, and one declared seed per arm;
- a deterministic `phase, phase, natural` schedule;
- 1:1:2 continuous-4K, target-8K, and target-16K phase exposure within the
  phase batches;
- complete answer-plus-immediate-EOS supervision;
- physical training sequences no longer than 4K.

The only active method variable between the matched arms is the frequency
table: the original Native RoPE table versus the EVQ-Cosh table. The resulting
LoRA weights differ because they are trained under those respective frequency
tables; that divergence is an outcome of the intervention, not an additional
experimental variable.

Put plainly, the scientific comparison is original RoPE versus EVQ-Cosh. It
is not an ablation of one internal coordinate within EVQ-Cosh, and the final
trained adapters are not expected to be bitwise identical.

The long-range exposure is implemented with explicit position IDs, so this
must not be described as training without long-position exposure.

## 2WikiMultiHopQA endpoint

### Data and evaluation

The training view contains 1,536 examples and the validation view 128 examples.
The evaluation uses 200 held-out LongBench 2WikiMultiHopQA rows with zero
train/evaluation QA-identity overlap.

Evaluation is greedy autoregressive generation with a 32-token output budget.
The physical chat inputs are deterministically brought to approximately
4K/8K/16K by retaining the official prompt and appending answer-filtered
2Wiki distractor contexts where needed. The mean input lengths are
`4063.96`, `8159.86`, and `16351.86` tokens. This is a controlled
task-family long-QA protocol, not the unmodified LongBench leaderboard
protocol.

### Final matched result

Each cell contains 200 generations.

| Length | Native token-F1 | EVQ token-F1 | Native exact | EVQ exact |
| --- | ---: | ---: | ---: | ---: |
| 4K | **25.99%** | 24.84% | **22.0%** | 21.5% |
| 8K | 0.07% | **21.48%** | 0% | **17.5%** |
| 16K | 0% | **8.57%** | 0% | **4.0%** |

Terminal-EOS rates for Native/EVQ are `100%/99.5%` at 4K,
`61.0%/99.5%` at 8K, and `100%/87.0%` at 16K. Native's 16K EOS rate is not
capability: all 200 predictions are empty.

### Stage-A parent baseline

The same filled evaluation was completed on the two unadapted Stage-A
parents:

| Length | Native parent F1 | EVQ parent F1 | Native parent exact | EVQ parent exact |
| --- | ---: | ---: | ---: | ---: |
| 4K | 8.73% | 8.61% | 0% | 0% |
| 8K | 0.16% | 7.88% | 0% | 0% |
| 16K | 0.30% | 0.93% | 0% | 0% |

Thus the shared Q/K-only continuation raises 4K exact from zero to
approximately 22% for both substrates. At 8K and 16K, only the EVQ arm turns
that acquired task capability into non-zero exact match.

### Teacher-forced diagnostic

On the same 32-row validation slice, EVQ Q/K-only continuation changes
answer-plus-EOS NLL at continuous/8K-phase/16K-phase positions from
`3.631/4.084/4.323` to `0.394/0.607/0.981`. The matched Native arm changes
from `3.312/8.519/7.905` to `0.313/0.938/2.037`.

These diagnostics explain the learning trajectory but do not replace the
autoregressive QA endpoint above.

## 13-family RULER endpoint

The RULER adapters are independent of the QA adapters. Each starts from its
matching Stage-A parent and receives the same 1,248-row, 13-family training
view. Evaluation covers all 13 families at 4K/8K/16K, with 20 held-out
examples per cell and greedy autoregressive decoding.

### Official task-specific macro

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native Stage-A parent | 59.03% | 0% | 0% |
| EVQ Stage-A parent | 11.09% | 5.74% | 2.19% |
| Native Q/K phase-adapted | **72.19%** | 2.02% | 0.38% |
| EVQ Q/K phase-adapted | 42.44% | **31.63%** | **5.03%** |

### All-references-found macro

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native Q/K phase-adapted | **61.15%** | 0% | 0% |
| EVQ Q/K phase-adapted | 26.92% | **18.08%** | **1.54%** |

### All 13 official family scores

| Family | Native 4K | EVQ 4K | Native 8K | EVQ 8K | Native 16K | EVQ 16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NIAH single 1 | 100% | 95% | 0% | 90% | 0% | 0% |
| NIAH single 2 | 100% | 100% | 0% | 90% | 0% | 10% |
| NIAH single 3 | 100% | 40% | 0% | 5% | 0% | 0% |
| NIAH multikey 1 | 90% | 25% | 0% | 15% | 0% | 0% |
| NIAH multikey 2 | 95% | 0% | 0% | 0% | 0% | 0% |
| NIAH multikey 3 | 50% | 0% | 0% | 0% | 0% | 0% |
| NIAH multivalue | 93.75% | 67.5% | 0% | 48.75% | 0% | 3.75% |
| NIAH multiquery | 90% | 65% | 1.25% | 43.75% | 0% | 0% |
| VT | 64% | 59% | 0% | 28% | 0% | 0% |
| CWE | 29% | 8.5% | 0% | 14% | 0% | 0% |
| FWE | 46.67% | 56.67% | 25% | 46.67% | 5% | 41.67% |
| QA 1 | 50% | 15% | 0% | 10% | 0% | 5% |
| QA 2 | 30% | 20% | 0% | 20% | 0% | 5% |

EVQ is non-zero in 11/13 families at both 4K and 8K, and in 5/13 at 16K.
Native is non-zero in all 13 families at 4K, 2/13 at 8K, and 1/13 at 16K.
This is substantial 4K repair and strong 8K transfer, but not complete
in-window restoration or robust 16K RULER capability.

## Interpretation and boundary

The QA result supports selective phase adaptation: identical Q/K-only task
adaptation gives both arms nearly the same 4K capability, while only the fixed
EVQ substrate carries substantial capability to 8K and measurable capability
to 16K. The RULER result supports the same direction but with a material
boundary: EVQ's 4K macro improves by 31.35 percentage points over its Stage-A
parent, yet remains 29.75 points below the matched Native arm.

## Reviewer-facing QA wording

> We additionally tested whether the in-window cost of adapting a mature
> 1.485B model is avoidable. From matched Native and EVQ Stage-A parents, we
> froze the inherited V/O LoRA tensors and continued only Q/K LoRA, using
> complete answer-plus-EOS supervision and physical training sequences no
> longer than 4K. On held-out 2WikiMultiHopQA prompts, Native/EVQ obtain
> 25.99/24.84 token-F1 at 4K, but 0.07/21.48 at 8K and 0/8.57 at 16K
> (200 examples per length). Exact match is 22.0/21.5% at 4K, 0/17.5% at 8K,
> and 0/4.0% at 16K. The protocol uses deterministic 2Wiki distractor filling
> to realize the target physical lengths, so we describe this as held-out
> task-family length transfer rather than an unmodified LongBench leaderboard
> result.

## Reviewer-facing RULER wording

> We also applied the same Q/K-only phase adaptation independently to all 13
> RULER families. Native/EVQ official macro is 72.19/42.44% at 4K,
> 2.02/31.63% at 8K, and 0.38/5.03% at 16K (20 examples per family and
> length). Relative to the EVQ Stage-A parent, this raises the 4K macro from
> 11.09% to 42.44% and the 8K macro from 5.74% to 31.63%. We therefore view
> selective Q/K adaptation as a substantial in-window repair with strong 2x
> transfer, not as complete 4K restoration or robust 4x RULER capability.

It does not establish:

- clean unseen-task transfer;
- a decomposition of which individual coordinates or geometric properties
  inside the complete EVQ-Cosh frequency table cause the observed effect;
- universal long-context superiority;
- multi-seed stability;
- structural or bitwise identity to Native at 4K.

The latter would require a different Native-local plus far-only EVQ residual
attention operator. That design is not the intervention tested here and would
constitute a new attention method rather than ordinary EVQ-LoRA.

## Provenance

The machine-readable metrics and full per-example generations are in
`olmo2_qk_phase_adaptation_20260729/`.

| Artifact | SHA-256 |
| --- | --- |
| Checkpoint | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| QA training/evaluation manifest | `9abbc22bcf833762076fd895f9d26706f0cfbecdceb88a2867ced63bf2fdddf9` |
| Native QA final adapter | `d0aeaf39a9508e36eed8e7e2bf4235db245326571cf18a82ebff45017d3a76c4` |
| EVQ QA final adapter | `2e0d47b456e691be7bc880deec9477b1fb2c4257ec73a73fc74a86ffa8ddcd2e` |
| Native QA generations | `f2b1a8e3cc4b498b39b6205cdabb95745756519f41ac0d13ff8fc59ce7751a93` |
| EVQ QA generations | `434937ad0f52ff118c9299e57bd5a51b541079c990e618286b930b2745ee13e2` |
| RULER training-view manifest | `0bf1f2158340c7b6190c73c3b1aa0e07e8007e8c28f2da6f36b471528c361161` |
| RULER evaluation manifest | `886d94731af31f30698204caa9f30045887c9333d6327e2e1c33555ed57da68c` |
| Native RULER final adapter | `556db0a42d70944578ece2e51444bbce5565e83fcf6a799fb135c490578ddcde` |
| EVQ RULER final adapter | `f2d05a6ab572210bf29e628f47681a6456426d41eaab5ed6700528edfbb4b8c7` |
| Native RULER generations | `47724bdbd798c79acab156b415c1abbe855fa8c1cd25c79ccb053145fd1413cb` |
| EVQ RULER generations | `6d9cc1f5a006b92ad90d1cd49b5f7e27846386adbd33e983031b5c273c2473c6` |
