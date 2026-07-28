# OLMo-2 1.485B Shared QA/RULER Task Localization

Date: 2026-07-30
Status: `RUNNING / NOT YET REVIEWER-USABLE`
Evidence tier: `DESIGN_ONLY_OR_PENDING`

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

## Runtime state

The EVQ arm passed READY and entered a real optimizer step:

- step 1 loss: `4.542545318603516`;
- step 1 gradient norm before clipping: `23.58408546447754`;
- peak allocated memory at step 1: `23,290,729,472` bytes;
- first-step family: `phase_2wiki`;
- loss was finite.

The matched Native arm is READY and registered to start automatically after
successful EVQ completion. Strict 4K/8K screens are registered to run
immediately after both arms complete.

## Decision gate

The shared adapter is useful only if all of the following hold:

1. EVQ exceeds matched Native at 8K on both 2Wiki and complete-family RULER;
2. EVQ remains within 5 percentage points of Native 2Wiki F1 at 4K;
3. EVQ remains within 10 percentage points of Native RULER macro at 4K;
4. gains appear in autoregressive capability metrics, not only NLL;
5. Native/EVQ selection, position, exposure, family-count, and protocol
   receipts match.

If the screen fails, stop this shared route without a ratio, seed, rank, or
loss sweep. The valid conclusion would be that separate task localization is
supported, while a single adapter has not been shown to carry both task
families.
