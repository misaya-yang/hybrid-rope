# OLMo-2 1.485B Shared QA/RULER Task Localization

Date: 2026-07-30
Status: `SCREEN COMPLETE / FULL LONG-ROUTE EVALUATION RUNNING`
Evidence tier: `POST_SUB_RAW_HASH_BACKED_SCREEN_ONLY`

## Question

The fresh generic-data Q/K experiment improves long-position natural-text
NLL but does not recover broad task capability after the EVQ frequency
replacement. This experiment tests whether one shared, short Q/K-only
continuation can localize that long-position substrate to both held-out
multi-hop QA and all 13 RULER families without training on evaluation rows.

## Matched parents

Both arms start from the completed fresh generic-data Q/K-only adapters:

| Arm | Parent adapter SHA-256 |
| --- | --- |
| Native | `3b8735c9b0cd6b65238f256ce0d9395c6e835be9ba7583fca485bc1e675c4740` |
| EVQ | `26157b7b586449bcc1a3ad5ff574ae4428640a047bc3fdad90a40e71d2653c79` |

The base checkpoint composite SHA-256 is
`36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
The only registered method variable is the active Native or EVQ frequency
table. Base weights remain frozen; the continuation updates only the existing
Q/K LoRA tensors.

## Registered protocol

- rank 64, alpha 128 Q/K-only LoRA;
- 500 optimizer steps, global batch 8;
- learning rate `5e-5`, 20 warmup steps, fused AdamW, BF16;
- fixed optimizer-step cycle:
  `2Wiki, RULER13, 2Wiki, RULER13, LongAlign replay`;
- 200 2Wiki steps, 200 RULER13 steps, and 100 replay steps;
- within every task microbatch, deterministic
  `continuous-4K : target-8K : target-16K = 1:1:2` phase exposure;
- complete answer-plus-immediate-EOS supervision;
- every physical training sequence is at most 4,096 tokens;
- one matched seed, `20260730`.

This gives each task family the same 200 phase-step dose as the corresponding
successful separate continuation while sharing one natural-replay stream.

## Data and isolation

| View | Rows / role | Manifest SHA-256 | Isolation gate |
| --- | --- | --- | --- |
| 2Wiki phase view | 1,536 train; held-out validation | `9abbc22bcf833762076fd895f9d26706f0cfbecdceb88a2867ced63bf2fdddf9` | zero train/evaluation QA-identity overlap |
| RULER13 phase view | 1,248 train, balanced over 13 families | `0bf1f2158340c7b6190c73c3b1aa0e07e8007e8c28f2da6f36b471528c361161` | generated training and evaluation rows are disjoint; zero exact-row overlap |
| Generic replay | frozen LongAlign 4K view | bound in each READY receipt | no task labels |

This is task-family adaptation on held-out rows, not clean unseen-task
transfer.

## Executed training

Both matched 500-step continuations completed:

| Field | Native | EVQ |
| --- | ---: | ---: |
| Final adapter SHA-256 | `bf0aaeaa3e4e4221f622ea2586d760c47c9c694d315847bad96eba214ff942d3` | `24f484b988a5f083c03552b60ae3f5bebe18d49760f399fa793de39710dad769` |
| Processed input tokens | 16,380,000 | 16,380,000 |
| Optimizer steps | 500 | 500 |
| Measured training seconds | 555.53 | 728.87 |
| Measured input tokens/s | 29,485.47 | 22,473.12 |
| Peak allocated bytes | 22,280,460,800 | 23,290,729,472 |

The executed arms match exactly on:

- selection stream:
  `96534060023a1b26dce504f7056efe6fac2d9e7aa9dc7ed44e4bde2ea86bb480`;
- position stream:
  `2e535b05b443393f1f98549ae2442acd1e11f29b5e67692dd24ddfee41ee46f3`;
- exposure stream:
  `df4889359fe0ecd719afee7a61b0451013a2e7817d970b91e82de4ff0a3c8a3d`;
- 200/200/100 2Wiki/RULER/natural steps;
- 800/800/1,600 contiguous/8K-phase/16K-phase sequence exposures;
- 16,380,000 processed input tokens and the complete optimizer protocol.

Each task contributes 400/400/800 contiguous/8K-phase/16K-phase sequence
exposures. The only active method difference remains the Native versus EVQ
frequency substrate and the resulting learned adapter.

The post-training 32-row teacher-forced diagnostics are:

| View | Offset | Native NLL | EVQ NLL |
| --- | ---: | ---: | ---: |
| 2Wiki | 0 | 0.3527 | 0.5219 |
| 2Wiki | 4,096 | 1.1681 | 0.7799 |
| 2Wiki | 12,289 | 2.1311 | 1.2197 |
| RULER13 | 0 | 0.3014 | 1.8482 |
| RULER13 | 4,096 | 2.0817 | 2.8493 |
| RULER13 | 12,289 | 2.6424 | 3.1437 |

These values diagnose the optimization trajectory; they are not capability
results.

## Strict autoregressive screen

The screen uses 50 held-out 2Wiki rows per length and two held-out examples
per RULER family and length. Every result passed a strict identity gate over
checkpoint, adapter SHA, frequency substrate, data manifest, evaluator hash,
protocol, cell coverage, and raw-generation hash.

| Endpoint | Native 4K | EVQ 4K | Native 8K | EVQ 8K |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki token F1 | 36.27% | 21.50% | 0.95% | 15.19% |
| 2Wiki normalized exact | 32.0% | 16.0% | 0% | 10.0% |
| 2Wiki terminal EOS | 100% | 100% | 30.0% | 100% |
| 13-family RULER official macro | 70.38% | 34.36% | 7.37% | 22.24% |

## Decision gates and stop decisions

The shared adapter is useful only if all of the following hold:

1. EVQ exceeds matched Native at 8K on both 2Wiki and complete-family RULER;
2. EVQ remains within 5 percentage points of Native 2Wiki F1 at 4K;
3. EVQ remains within 10 percentage points of Native RULER macro at 4K;
4. gains appear in autoregressive capability metrics, not only NLL;
5. Native/EVQ selection, position, exposure, family-count, and protocol
   receipts match.

The single-path gate fails because EVQ trails Native by 14.77 percentage
points on 4K 2Wiki F1 and by 36.03 points on 4K RULER. It nevertheless passes
the long-route portion of the gate: at 8K, EVQ has higher 2Wiki F1 and exact
match, 100% terminal EOS, and a 14.87-point RULER advantage over the matched
Native continuation.

No ratio, seed, rank, learning-rate, or loss sweep is authorized. The complete
4K/8K/16K single-path matrix is stopped.

## Request-length routing follow-up

The screen supports evaluating one non-training deployment alternative:

- registered total context budget at most 4K: untouched Native RoPE with the
  adapter disabled;
- evaluated 8K and 16K total context budgets: EVQ-Cosh with this one shared
  QA/RULER Q/K adapter enabled for the entire request.

The choice is made before prefill; it does not switch a KV cache mid-request,
use dual attention, or add another training intervention. It is a
deterministic two-path deployment policy, not a single pure-EVQ configuration.

A full matched 8K/16K evaluation is running for both shared Native and shared
EVQ adapters. Only after those outputs and the untouched-Native 4K component
are hash-validated may a routed composite be reported. Until then, this
owner remains screening evidence and is not reviewer-usable as a full result.
