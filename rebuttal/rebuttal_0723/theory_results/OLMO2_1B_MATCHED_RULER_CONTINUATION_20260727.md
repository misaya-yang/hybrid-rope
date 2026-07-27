# OLMo-2 1.485B Matched 4K RULER-Family Continuation

Date: 2026-07-27
Status: `POST_SUB_RAW_HASH_BACKED`
Concerns: `RDz6s.1`, `RzWsa.3/.4`, `R27bE.2/.5`, `AC.2`

## Direct answer

Under the same physical-4K, 13-family continuation, Native-LoRA learns the
in-window task distribution more strongly, but EVQ-LoRA supplies the length
transfer: Native/EVQ official macro is `82.16%/37.51%` at 4K,
`0.08%/21.29%` at 8K, and `0%/6.13%` at 16K.

## Matched protocol

Both arms use:

- `OLMo-2-0425-1B-Instruct` (1.485B actual parameters);
- the matched seed-20260725 Native/EVQ rank-64, alpha-128 Q/K/V/O LoRA
  parents;
- the same fixed 736-row continuation view and row order: 480
  official-generator rows, 128 paired NIAH replay rows, and 128 LongAlign
  replay rows;
- physical 4K backward passes only, three deterministic passes, 276 optimizer
  steps, global batch 8, fused AdamW, learning rate `2e-5`, 14 warmup steps,
  BF16, and seed `20420726`;
- the same 13 tasks at 4K/8K/16K, 20 examples per cell, greedy autoregressive
  decoding, and official task-specific RULER metrics.

The checkpoint, training-view manifest, evaluation manifest, evaluator
protocol, and all 39 evaluation cells are identical. The fixed frequency table
is the intended scientific difference.

## Training receipt

| Arm | Validation NLL, initial → final | Validation PPL, initial → final | Steps | Physical input tokens |
| --- | ---: | ---: | ---: | ---: |
| Native-LoRA | `1.5810 → 0.5020` | `4.8596 → 1.6520` | 276 | 9,041,760 |
| EVQ-LoRA | `3.0372 → 0.9881` | not used as the claim endpoint | 276 | 9,041,760 |

Native starts closer to the supervised 4K distribution and ends with the lower
validation NLL. Its long-range failure therefore cannot be explained by a
failure to learn the continuation mixture.

## Autoregressive RULER result

### Official task-specific macro

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native-LoRA | **82.16%** | 0.08% | 0% |
| EVQ-LoRA | 37.51% | **21.29%** | **6.13%** |

### All-references-found macro

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native-LoRA | **66.54%** | 0% | 0% |
| EVQ-LoRA | 21.92% | **12.69%** | **2.31%** |

At 8K, Native's only non-zero official cell is CWE at `1%`, which produces
the `0.08%` 13-task macro. EVQ has non-zero official scores in 10/13 cells,
including VT `31%`, CWE `22%`, FWE `60%`, SQuAD QA `10%`, and HotpotQA
`25%`. At 16K, Native is zero in every cell; EVQ remains non-zero in seven
cells, including FWE `50%`, SQuAD QA `10%`, and HotpotQA `10%`.

The pooled 39-cell macro is not the relevant superiority endpoint because it
mixes the training length with 2x/4x transfer. Native's large 4K advantage and
EVQ's 8K/16K advantage must be reported together.

## Interpretation and boundary

This is a matched demonstration that the EVQ frequency substrate changes
length transfer after task-family adaptation at 1.485B scale. It complements
the matched LLaMA-3-8B RULER result: Native is stronger at the physical
training length, while EVQ retains non-zero capability beyond it.

The result remains:

- one continuation seed per arm;
- task-family-supervised adaptation, not unseen-task transfer;
- a combined frequency-substrate-plus-LoRA comparison, not pure interior-shape
  attribution;
- a 4K/8K/16K result, with no 32K capability claim.

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| Checkpoint | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| Native parent adapter | `6570ab94aec68431dd4e261eb3ef342ef72253357aa65a0d37b0a018df2f3f8d` |
| EVQ parent adapter | `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a` |
| Training-view manifest | `1bd1d35a40bbc450757c920028971df093885726153571ab02af47130681ef17` |
| Evaluation manifest | `886d94731af31f30698204caa9f30045887c9333d6327e2e1c33555ed57da68c` |
| Native final adapter | `da8111055a23ea9f56ff675399eebc2c47c5afbedb6b7079dc12598889280d80` |
| EVQ final adapter | `b18f6cfee8aad7d9004938beec0a075834cc05c69c6433aaed1943c7a603a653` |
| Native training result JSON | `3cb52dbe37956d5d3e7ec8445606a617eaac5b1353af6849da9fb3f72e133c56` |
| Native evaluation result JSON | `24eca20f7f9a98c747f775e67d0ad40ffd9eba6b2083469dbc30f5e448b00ea1` |
| Native per-example predictions | `f2b03cc08bc66f9cd60a1dae6a9f60e143cd9c7a700f0266f01f174229368ae2` |
| Native-ready receipt | `40ba81068043880f6c53fd2fb3bffe70d46f390344e0874a7be81c348e39a3c2` |
| Parameterized training implementation | `7749395dccebd17e12e7490df36bcf89c237a4c3b38724416ab655a49885e0dc` |
| Evaluator implementation | `abeb4c44d6d5a0ba5bc2caf55c48ecafe9c25c163471d6bcdc25c4e6f4b0a5a2` |
| Frozen 12-file checksum manifest | `b8b668ac1fa4fbf626d4805888d15fed5902d937fb91829753ebe9413a172830` |
