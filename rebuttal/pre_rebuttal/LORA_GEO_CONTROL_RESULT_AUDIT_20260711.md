# LoRA Geo-Control Result and Provenance Audit

> **2026-07-13 status update:** 本文对旧 LongAlign/LongAlpaca cross-protocol contrast 的否定仍成立。后来完成的 `LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` 闭环了一个新的 matched-training-pipeline seed-42 comparison，但 Geo为native endpoint、EVQ为midpoint，仍不是same-quantizer pure density-shape control；当前总边界见 `../rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`。

**Date:** 2026-07-11
**Status:** internal rebuttal analysis; result-bearing but not yet a reviewer-facing claim
**Scope:** LLaMA-3-8B-Instruct, 300-step q/k/v/o LoRA, seed 42, WikiText-2 PPL at 8K/16K/32K

## Executive verdict

The fresh native-Geo LoRA seed-42 run completed successfully and its evaluation
is internally consistent. On the new, pinned official LongAlign-10k protocol,
Geo+LoRA improves WikiText-2 NLL in every one of the five evaluated chunks at
8K, 16K, and 32K. There is no current evidence that this improvement is caused
by an evaluator, label-shift, frequency-injection, or single-chunk failure.

However, this run is **not a strict matched control for the historical EVQ-LoRA
seed-42 row in the paper**. Contemporaneous experiment records indicate
LongAlpaca-12k as the historical EVQ training corpus, but the exact historical
corpus identity is not hash-verified; the fresh Geo run uses the pinned official
LongAlign-10k release. The old downloader could write LongAlpaca content under
a file named `longalign_10k.jsonl`. Therefore the
cross-run comparisons `9.63 vs. 6.48`, `21.5 vs. 116.5`, and `104.3 vs. 1205`
do not isolate EVQ as the only changed variable.

The pre-registered Geo-control gate that compares the fresh Geo result directly
with the historical EVQ number is consequently invalid for causal attribution.
The fresh result remains useful as a verified Geo+LoRA result on official
LongAlign, but it must be paired with a fresh EVQ run on the same frozen data
before it can support an EVQ-versus-Geo rebuttal claim.

## 1. Fresh Geo+LoRA seed-42 run

### 1.1 Training protocol

| Field | Value |
| --- | --- |
| Base model | Meta-Llama-3-8B-Instruct |
| Frequency method | native geometric RoPE |
| Seed / split seed | 42 / 42 |
| Training data | pinned official `zai-org/LongAlign-10k` |
| Training objective | full-token causal LM loss |
| Maximum sequence length | 8192 |
| Accepted samples | 8,000 before the 98/2 split |
| Train / validation rows | 7,840 / 160 |
| LoRA | rank 64, alpha 128, dropout 0.05 |
| Target modules | q/k/v/o projections |
| Precision | BF16, no quantization |
| Microbatch / accumulation | 2 / 4, effective batch 8 |
| Optimizer | AdamW, LR `1e-4`, cosine, warmup 60, WD 0.01 |
| Steps | 300/300 |
| Training time | 2.333 hours |
| Aggregate train loss | 1.7589004485 |
| Status | complete |

The logged gradient norms remained finite and stable through step 300. The
last recorded gradient norm was 0.1875. The run produced complete checkpoints
at steps 100, 200, and 300 and a final adapter.

### 1.2 Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Base-model manifest | `0196fe3f3dcd932e337a7a0e91625fd12667cecbcca221020ea39428c6178210` |
| Frozen LongAlign manifest | `d7d1c62303eb150f952b86074676db8840fe7edc370258216b76801f25c8f957` |
| Frozen WikiText evaluation manifest | `60347f186a7096f2ff23a0e25d2f041108e93370a20ed536f27833a820998f88` |
| Training code | `5667cb021643d332a01035691ad085797c60288b691ff0576bebff9cd4c040bd` |
| Final adapter | `69956a9503d951b79a2ee987a32c2b59ee28ca7f2608d2d0001dd3d973498d6d` |
| Native-Geo frequency artifact | `09fab0f4da1c96bf31cf5d54dd45a935dc63f1ef8d7b1669dd5361087e4bb794` |
| Immutable run protocol | `393804558c51f1469309ce21c84b6c8b5f1fffc7cbb686b3293e3f2b7943fadd` |

The result-bearing files are `eval_base_geo.json`,
`eval_geo_longalign_s42.json`, and `legacy_geo_control_summary.json`. Their
contents were inspected on the experiment host; copies still need to be moved
into a tracked, anonymous result bundle before reviewer-facing use.

## 2. Exact evaluation results

| Context | Base-Geo NLL | Base-Geo PPL | Geo+LoRA NLL | Geo+LoRA PPL | Geo NLL delta | Geo PPL delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 2.004237 | 7.4204 | 1.868129 | 6.4762 | -0.136108 | -12.73% |
| 16K | 5.170501 | 176.0030 | 4.757525 | 116.4573 | -0.412976 | -33.83% |
| 32K | 7.572270 | 1943.5477 | 7.094225 | 1204.9877 | -0.478046 | -38.00% |

The current Base-Geo values closely reproduce the historical paper row
`7.42 / 176.3 / 1942.5`, supporting evaluation-corpus and metric compatibility.
This agreement does not establish training-data identity for the adapter runs.

### 2.1 Per-chunk consistency

Each length uses five disjoint full-context chunks from one frozen WikiText-2
token tensor. The following values are paired `Geo+LoRA NLL - Base NLL` deltas:

| Context | Chunk-level NLL deltas | Mean | Sample SD across chunks |
| --- | --- | ---: | ---: |
| 8K | -0.159675, -0.127698, -0.135472, -0.108520, -0.149175 | -0.136108 | 0.019738 |
| 16K | -0.430039, -0.394210, -0.408179, -0.448586, -0.383867 | -0.412976 | 0.026371 |
| 32K | -0.527772, -0.433123, -0.509898, -0.416183, -0.503252 | -0.478046 | 0.049919 |

All 15 paired chunk deltas favor Geo+LoRA. These chunks establish within-corpus
consistency, not training-seed replication; they must not be treated as five
independent model runs.

## 3. Critical provenance mismatch

### 3.1 Records indicate LongAlpaca; exact historical corpus identity is not hash-verified

The strongest contemporaneous records indicate:

- `internal/2026_04_run/docs/11_LoRA_EVQ_v2_实验报告_0401.md` records
  `LongAlpaca-12k, seq_len=8192` as the historical EVQ training data and later
  explicitly discusses LoRA learning the LongAlpaca response style.
- `internal/2026_04_run/docs/12_LoRA_PE_Baseline_Comparison_实验计划.md`
  specifies LongAlpaca-12k as the shared data intended for Geo/EVQ/YaRN arms
  and identifies `evq_r64_tau1414` as the existing seed-42 checkpoint.
- The historical `download_model_data.py` checks a cached LongAlpaca release
  before attempting LongAlign and writes either source into the same output
  filename, `longalign_10k.jsonl`.
- The current experiment README already warns that this downloader is not
  claim-safe and states that the historical raw hashes/runtime were not
  recovered.

The paper currently describes the historical row as LongAlign-10k. The
available artifacts do not hash-verify the exact historical corpus and therefore
cannot establish a strict matched pair. They do create a material risk of a
dataset-provenance labeling error that must be reconciled before rebuttal. The
PPL numbers themselves are not changed by this finding, but the training-corpus
description and any matched-control claim are affected.

### 3.2 Match-status table

| Component | Base vs. fresh Geo | Fresh Geo vs. historical EVQ |
| --- | --- | --- |
| Base-model family | confirmed | likely, exact historical weight hash unavailable |
| Evaluation corpus/metric | confirmed by fresh manifest; historical values reproduce closely | compatible, but historical raw evaluator artifact is not recovered |
| Seed | n/a vs. 42 | both reported as 42 |
| Steps / LoRA rank / alpha / targets | n/a | reported matched |
| Optimizer / LR / warmup / batch | n/a | reported matched |
| Training objective | n/a | reported full-token causal LM in both |
| Training data | n/a | **not established as matched: fresh official LongAlign; records indicate historical LongAlpaca, without a recovered hash** |
| Token stream/order | n/a | unresolved and not hash-verifiable |
| Runtime/code bytes | fresh hashes available | historical bytes/runtime not recovered |
| Scientific status | strict fresh comparison | **not a strict matched pair** |

## 4. Why the apparent 49% trade-off is misleading

The direct PPL ratio `9.63 / 6.476 = 1.487` is numerically correct but is not a
causal EVQ effect because current artifacts do not establish that the two
adapters form a strict matched pair, and contemporaneous records indicate
different training corpora. It should not be quoted as a matched `EVQ - Geo`
trade-off.

For scale interpretation only, PPL is exponential in NLL. If the currently
displayed values are placed side by side without making a causal claim:

| Context | Fresh Geo NLL | Historical EVQ NLL | Difference | PPL ratio |
| --- | ---: | ---: | ---: | ---: |
| 8K | 1.868 | 2.265 | +0.397 | 1.487x |
| 16K | 4.758 | 3.068 | -1.689 | 0.185x |
| 32K | 7.094 | 4.647 | -2.447 | 0.087x |

Thus a nominal 49% PPL increase at 8K is an additive increase of roughly
0.397 nat/token, whereas the displayed 32K difference is roughly 2.447
nats/token in the favorable direction. If a future matched experiment
reproduced these values, the long-range NLL gain would be about 6.2 times the
in-range NLL cost in magnitude. The current cross-corpus values cannot establish
that ratio as an EVQ effect.

The user's practical intuition is partly correct: `1205 -> 104` is a much larger
stability change than a two-to-three-point PPL shift near 8K. However, PPL 104
does not by itself demonstrate usable retrieval or generation. Historical
needle/RULER evaluations did not show corresponding extended-context task
success, so the safe interpretation is “substantially less token-level
collapse,” not “functional 32K capability.”

## 5. Historical position-bucket evidence

The contemporaneous historical report contains a more informative positional
decomposition than the aggregate paper table:

| Position bucket | Base PPL | Historical EVQ-LoRA PPL | Relative change |
| --- | ---: | ---: | ---: |
| 0-4K | 8.24 | 8.50 | approximately +3% |
| 4K-8K | 9.05 | 13.38 | approximately +48% |
| 8K-12K | 429 | 20.7 | approximately -95% |
| 12K-16K | 24,959 | 63.8 | approximately -99.7% |

This suggests that the historical in-range cost was concentrated near the
4K-8K training-window boundary rather than being a uniform 50% degradation
over all in-range positions. The sharp crossover immediately beyond 8K is
consistent with a frequency-reallocation interpretation, but it remains
single-seed and lacks the matched historical Geo+LoRA adapter.

The current evaluator reports whole-sequence averages. Its 16K result mixes
positions 0-8K and 8-16K, and its 32K result mixes all earlier position bands.
Future matched evaluation should therefore report position-bucket NLL in
addition to aggregate PPL.

## 6. What is established and what is not

### Established by the fresh run

1. A 300-step native-Geo q/k/v/o LoRA update on pinned official LongAlign-10k
   improves WikiText-2 NLL for this seed and protocol.
2. The improvement occurs in every evaluated chunk at 8K, 16K, and 32K.
3. The current evaluator uses a shared frozen token tensor, correct causal label
   shifting, canonical native-Geo frequencies, and fixed five-chunk coverage.
4. Ordinary LoRA/long-sequence adaptation can itself improve extrapolation PPL;
   any EVQ claim must be expressed as an incremental gain over a matched trained
   Geo arm.

### Not established

1. That EVQ causes a 48.7% PPL penalty relative to matched Geo at 8K.
2. That EVQ alone causes the displayed `116.5 -> 21.5` or `1205 -> 104.3`
   changes.
3. That the historical paper row was trained on official LongAlign-10k.
4. That five WikiText chunks substitute for multiple training seeds.
5. That PPL near 100 implies usable retrieval, instruction following, or
   autoregressive long-context capability.
6. That the fresh Geo result invalidates the historical EVQ result, or proves
   that a matched comparison was never executed; current artifacts establish
   neither claim and do not establish a strict matched pair.

## 7. Required next experiment

Do not spend GPU time on EVQ seeds 43/44 before establishing a valid seed-42
matched pair.

### Preferred clean path

Run one fresh EVQ-Cosh seed-42 adapter on the **same frozen official
LongAlign-10k token manifest** used by the completed Geo-42 run, changing only
the frequency method to EVQ-Cosh with tau 1.414. Evaluate both adapters with the
same frozen WikiText manifest and add position buckets:

- 0-4K;
- 4K-8K;
- 8K-12K;
- 12K-16K;
- 16K-24K;
- 24K-32K.

Report per-chunk and per-bucket NLL before exponentiating to PPL. Add at least
one task-sensitive endpoint, such as Gold-answer NLL or a pre-registered
retrieval metric. Only if the fresh seed-42 direction is useful should EVQ
seeds 43/44 be run to assess EVQ variance.

### Historical-lineage alternative

Recover the exact historical LongAlpaca artifact and historical checkpoint, or
freeze the same public LongAlpaca release as a clearly labeled best-effort
reconstruction, then train Geo-42 on that corpus. Without the original raw hash,
this path cannot be called bitwise reproduction.

## 8. Rebuttal-safe wording

Safe before a fresh matched EVQ run:

> A fresh seed-42 native-Geo LoRA control on a pinned official LongAlign-10k
> release improves WikiText-2 PPL at 8K, 16K, and 32K. This demonstrates that
> long-sequence LoRA adaptation itself can improve extrapolation and that only
> an incremental comparison against a trained Geo control can be attributed to
> EVQ. We do not compare this adapter causally with the historical EVQ-LoRA row
> because contemporaneous records indicate LongAlpaca-12k for that row, while
> the exact historical corpus identity is not hash-verified and the fresh run
> uses the newly pinned LongAlign release.

Safe after a fresh matched seed-42 pair, subject to its actual values:

> We reran native-Geo and EVQ-Cosh under the same frozen training tokens, model,
> seed, LoRA configuration, optimizer schedule, and evaluator. Base-to-Geo
> measures the contribution of long-sequence LoRA adaptation, while
> Geo-to-EVQ measures the incremental allocation effect.

Do not write:

- “The matched EVQ trade-off is 49%” from the current cross-corpus comparison.
- “Fine-tuning cannot explain any long-range improvement.”
- “The historical EVQ run used verified LongAlign-10k.”
- “PPL 104 proves usable 32K retrieval.”
- “Five chunks constitute five independent replications.”

## 9. Statistical and causal-risk scan

All 11 experiment-agent fallacy categories were checked. The material risks are:

| Risk | Severity | Finding |
| --- | --- | --- |
| Correlation/contrast interpreted as causation | RED FLAG | Fresh Geo and historical EVQ differ in training corpus and provenance. |
| Garden of forking paths | CAUTION | Aggregate PPL, positional PPL, retrieval, and multiple training/evaluation variants exist; the endpoint and comparator must be predeclared. |
| Look-elsewhere effect | CAUTION | Multiple lengths and metrics are inspected; all registered endpoints, including failures, must be reported. |
| Ecological/aggregation error | CAUTION | Aggregate 8K PPL hides a small 0-4K cost and a much larger 4-8K boundary cost. |
| Pseudoreplication | RED FLAG | Five chunks are not five training seeds. |

Simpson's paradox, Berkson bias, collider bias, base-rate neglect, regression to
the mean, survivorship bias, and reverse causality do not provide a better
explanation of the current within-run metrics. The primary blocker is the
training-corpus/provenance mismatch.
