# OLMo-2 1.485B: 4K Query-Gap Phase Routing

## Status and claim boundary

- Run state: training and registered first-number evaluation are complete;
  final capability validation is **incomplete**.
- Internal evidence tier: **AUTHOR_CONFIRMED_NOT_PROMOTED** until the stopped
  instance is explicitly restarted and its raw prediction JSONL files are
  recovered and frozen locally.
- Reviewer-promotion state: **not yet promoted**; the remote raw artifacts
  must be frozen into a sanitized local evidence package before these numbers
  enter the rebuttal or playbook.
- Supported claim: with only physical 4K training sequences, a semantic
  query-gap QKVO LoRA continuation improves real physical 8K and 16K greedy
  autoregressive **first-number answer extraction** on `niah_single_1`.
- Unsupported claims: unseen-task transfer, general RULER transfer, full
  response exact match, absence of catastrophic forgetting, multi-seed
  stability, or universal long-context capability.

The executed metric is **first-number exact**: the first complete
decimal number produced by greedy autoregressive generation must equal the
reference answer. This is an answer-extraction metric, not full generated-
string exact match, and must not be described simply as “strict exact.”

Two decision-critical checks remain unresolved:

1. full generated-string exact was not computed;
2. broad 4K retention was not evaluated, so catastrophic forgetting is not
   excluded by the 20-row NIAH result or natural-text NLL.

The prediction JSONL files were not copied locally before shutdown. No
full-string number may be reconstructed or guessed from the aggregate
receipts.

## Scientific question

The completed parent adapter learned 4K source-conditioned routing and
achieved strong 8K but weak 16K retrieval. Natural-text NLL remained finite at
16K, so the missing test was whether direct exposure to the target
source-query RoPE phases, without using physical sequences longer than 4K,
could transfer to continuous physical 8K/16K contexts.

## Registered intervention

| Field | Query-gap arm |
| --- | --- |
| Model | OLMo-2 1.485B Instruct |
| Parent | EVQ QKVO counterfactual-routing adapter, SHA-256 `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a` |
| Trainable parameters | Q/K/V/O LoRA, rank 64, alpha 128, `16,777,216` parameters |
| Physical training length | 4,096 tokens only; model input is the shifted 4,095-token context |
| Position intervention | Context/source positions remain contiguous; the final semantic query block, answer prefix, and teacher-forced answer are shifted together |
| Query offsets | Continuous deterministic coverage from `0` through `12,289`; offset is independent of row, source position, and source content |
| Pair parity | Gold-source and value-swapped variants receive identical position IDs |
| Optimizer steps | 100, with fixed `routing, routing, natural` family pattern |
| Step allocation | 67 routing steps and 33 natural-replay steps |
| Pair exposures | 268: contiguous 34, transition 33, middle 67, far 134 |
| Batch | micro-batch 4, gradient accumulation 2, global batch 8 |
| Optimization | fused AdamW, LR `5e-5`, 20 warmup steps, BF16 |
| Counterfactual loss | answer CE plus margin-1 source preference with weight 0.5 |
| Training seed | `20260728` |

The matched contiguous control starts from the same parent and uses the same
data, seed, family order, row-sampling generator, optimizer budget,
hyperparameters, and evaluation rows. Its only scientific change is the
absence of query offsets. The offset stream does not consume the training-row
RNG, so the routing and natural row order is reproducible across the two
arms. The comparison is scientifically paired, not bitwise paired: the
query-gap arm passes explicit position IDs while the control uses the model's
default contiguous IDs.

## Teacher-forced phase diagnostic

These values use physical 4K examples with virtual query offsets. They are a
mechanism diagnostic, not a capability endpoint.

| Query offset | Initial source-token exact | Final source-token exact | Initial median rank | Final median rank |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 46.88% | 70.08% | 2 | 1 |
| 8,192 | 27.08% | 65.50% | 8 | 1 |
| 12,289 | 23.96% | 54.18% | 11 | 1 |

Contiguous held-out calibration remained high: source-token exact changed
from 97.92% to 95.15%, with median rank 1 before and after.

## Real physical autoregressive answer-extraction results

All rows below use continuous physical contexts and greedy autoregressive
generation. The query-gap arm is frozen during evaluation.

### Selection screen: 20 rows per length

| Physical length | Parent | Matched contiguous +100 | Query-gap +100 |
| ---: | ---: | ---: | ---: |
| 4K | 19/20 | 20/20 | **20/20** |
| 8K | 14/20 | 16/20 | **20/20** |
| 16K | 1/20 | 0/20 | **11/20** |

The query-gap versus contiguous comparison has 4 wins, 0 losses, and 16 ties
at 8K; at 16K it has 11 wins, 0 losses, and 9 ties.

### Frozen 100-row confirmation

| Physical length | Matched contiguous +100 | Query-gap +100 | Paired wins / losses / ties |
| ---: | ---: | ---: | ---: |
| 8K | 68/100 | **95/100** | 28 / 1 / 71 |
| 16K | 0/100 | **51/100** | 51 / 0 / 49 |

On these fixed rows, exact two-sided McNemar values are
`1.12e-7` at 8K and `8.88e-16` at 16K. These quantify row-level paired
disagreement for this single trained seed; they are not training-seed
uncertainty and must not be presented as multi-seed significance.

## Source-to-generation-gap decomposition

The maximum observed gap in the original 4K routing data is 3,933 tokens.
The following table uses the real physical n=100 rows.

| Physical length | Real source-to-generation gap | Rows | Contiguous | Query-gap |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 0–3,933 | 50 | 48 | **49** |
| 8K | 3,934–7,167 | 37 | 17 | **34** |
| 8K | 7,168–8,191 | 13 | 3 | **12** |
| 16K | 0–3,933 | 34 | 0 | **25** |
| 16K | 3,934–7,167 | 12 | 0 | **9** |
| 16K | 7,168–8,191 | 5 | 0 | **2** |
| 16K | 8,192–12,287 | 20 | 0 | **9** |
| 16K | 12,288–16,079 | 29 | 0 | **6** |

The gain therefore extends beyond every source-query gap observed in the
physical 4K routing data. Performance still declines at the longest gaps;
`6/29` beyond 12,288 tokens is a material boundary and must remain adjacent
to any 16K aggregate.

## NLL is not the answer-extraction explanation

| Arm | 4K mean NLL | 8K mean NLL | 16K mean NLL |
| --- | ---: | ---: | ---: |
| Parent | 2.5480 | 2.7033 | 2.9245 |
| Matched contiguous +100 | **2.5431** | **2.6951** | **2.9094** |
| Query-gap +100 | 2.5441 | 2.6967 | 2.9142 |

The contiguous control has slightly lower natural-text NLL than the query-gap
arm at all three lengths, yet obtains `68/100` versus `95/100` at 8K and
`0/100` versus `51/100` at 16K. The first-number answer-extraction difference
therefore cannot be inferred from, or explained by, the small NLL difference.

## Narrow causal interpretation

The result supports the following bounded conclusion:

> For this EVQ QKVO LoRA parent and same-family NIAH protocol, explicit
> source-conditioned supervision at target query-source RoPE phases is the
> intervention that improves first-number answer extraction from physical 4K
> training sequences to continuous physical 8K/16K autoregressive prompts.

The matched contiguous arm rules out the explanation that another 100 steps
of the same counterfactual data are sufficient for the observed first-number
gain. The real continuous 8K/16K evaluation also shows that this gain is not
restricted to prompts containing the artificial query-boundary jump. It does
not establish full-response correctness, isolate relative phase from every
other effect of moving the entire query block, or reproduce the 16K key count
and attention competition during training.

## Mandatory limitations

1. This is one trained seed. The 100 evaluation rows reduce test-set noise but
   do not establish training-seed stability.
2. Training and evaluation use the same official numeric
   `niah_single_1` task family with disjoint rows and identities. This is
   task-family transfer across lengths, not unseen-task transfer.
3. The executed metric is first-generated-number exact, not full
   response-string exact. Full-string exact is unresolved and cannot be
   recovered from the aggregate receipts.
4. Physical training remains 4K, so the intervention teaches target relative
   phases without exposing the model to a physical 16K attention denominator
   or distractor load.
5. The longest real gaps remain the weakest stratum: `6/29` at
   12,288–16,079 tokens.
6. The paired test values describe fixed evaluation rows for a single
   training seed. Do not use multi-seed significance language.
7. 4K retention is a 20-row task-family screen plus natural NLL,
   not a broad instruction, RULER, or downstream no-harm result. Catastrophic
   forgetting therefore remains unresolved.
8. No matched Native/YaRN query-gap arm was run. This result identifies the
   intervention on the EVQ parent; it does not establish that only EVQ can
   benefit from target-phase supervision.
9. No downstream QA, multi-key RULER, variable tracking, or unseen benchmark
   result is established by this experiment.

## Raw/hash ledger

| Artifact | SHA-256 |
| --- | --- |
| Query-gap trainer | `8b3b45330f1ecd1abe7e897436ad542ffbe03280a5b2d0f50542e77c801167e7` |
| LoRA conversion module | `3934acf3c40f0a4ee18379db222e336f08b971a8590da248fef6e7869429708a` |
| Query-gap READY receipt | `da1d6c69a7a413737df4dc9b59666f27400a77bf74450fdf1a89dd757d74735c` |
| Query-gap training result | `ac66dca2aaa9e35cbf10c117eabe1ce0e293249e1f80bd0e05b0be54f9548e3e` |
| Query-gap adapter | `a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b` |
| Query-gap n=20 examples | `08796bb5a38109166d842756a36c4613133d3b252242ba033462c20c8c0411ee` |
| Query-gap n=20 result | `831fe7aff762c04dc0fab83b5b92b9d7997017f14f020a40b45c009e8784e11e` |
| Query-gap n=100 examples | `46ab4511d511aafba1214f64e38da0fe4cbc919c22e002d7e8b20f82aae21c48` |
| Query-gap n=100 result | `6c87b7eb5b07880c33225ef05bf4d1e76830297c2e2db145e3ec113b7e9c09b2` |
| Contiguous READY receipt | `4e568bcae59a0cf9d63cfbcb16a92523ce38d578f2e1ca69b565a17cccadd9fb` |
| Contiguous training result | `c5bd8f01ca5afb2cf4f85cc96b9d8edfa42e61a76781f67931960d6dcada82f2` |
| Contiguous adapter | `8e26819cf2adb9b4c613dfff4b31a2e67944fb661a1f2b062ddd7ec8127324ab` |
| Contiguous n=20 examples | `03f09ada5688fb8ccf9c061ac5b54af8364c02fbb70a81ba4673e3c701abfde0` |
| Contiguous n=20 result | `9ae3c0ffd8679f98c213afd1c9ba03820892ae43014519ccbfecce7f17af16fa` |
| Contiguous n=100 examples | `3064d0b1582d9bad71c0fac323a990cf6d692b7bb03fe85464c3aaabb3ed8283` |
| Contiguous n=100 result | `5c7a24fe367b6e1134e2895530164aa53bd59087b60f54f9d26f9f5aa70c5d5e` |
| 4K routing-data manifest | `83a745b25893a73749dd85ff66dea5b305be6e4e6dcf16a5d0e13cae935ec63b` |
| Row-matched n=20 manifest | `6e6f99457c02f55367e57b50daeff763a941291780d1d22bf369af9fb033ffc6` |
| Row-matched n=100 manifest | `274f5df85420aabde38191a3bc7da7d466eb13bff69f0fde5ad618aea3f07511` |

## Promotion gate

Before reviewer use:

1. restart the stopped instance only with explicit user authorization, recover
   the raw prediction JSONL files, and freeze them into a sanitized local
   evidence package;
2. compute and report full generated-string exact from those raw predictions;
3. run or identify a suitable broad 4K retention evaluation before making a
   no-forgetting claim;
4. verify the frozen hashes against this ledger;
5. synchronize the reviewer-usable evidence ledger and playbook;
6. state the single-seed and same-task-family boundaries adjacent to the
   result;
7. omit row-level significance language if the response cannot explain that
   it is not training-seed uncertainty.
