# Llama-3-8B Physical-8K RULER-Family Adaptation

## Reviewer concern

This package targets `R27bE.2`, `R27bE.5`, and `AC.2`: evidence on a mature
model and a downstream long-context capability, rather than language-model NLL
alone.

## Existing evidence

The frozen seed-42 Llama-3-8B adapters show that EVQ improves external
16K/32K language-model NLL and remote-source causal sensitivity.  They do not
yet establish broad autoregressive task use, and the existing QA aggregate is
weak.

## Smallest missing evidence

Can an already-trained EVQ adapter acquire RULER-family task behavior using
supervision confined to Llama-3-8B-Instruct's native 8K context, and retain
useful behavior when evaluated physically at 16K and 32K?

## Executable plan

- Continue the frozen seed-42 EVQ Q/K/V/O LoRA adapter.
- Every training forward is exactly 8,192 physical tokens.  No virtual gap,
  16K backward pass, or 4K substitute is admitted.
- Use all 13 official RULER task families: 96 generated training rows and four
  internal validation rows per task, plus 128 LongAlpaca instruction-replay
  rows.
- Use answer-only SFT for three epochs, global batch eight, peak LR `2e-5`,
  5% warmup, cosine decay, fused AdamW, BF16, Flash-only SDPA, fused
  linear-cross-entropy, and no activation checkpointing when it fits.
- Screen all 13 tasks at 8K/16K/32K with 20 held-out generated rows per cell.
  Extend useful cells to 100 rows.  Preserve every prediction and score.
- Run the matched Native continuation only if EVQ produces a useful 8K or 16K
  capability result.

The generated train and evaluation instances use different seeds.  QA uses
non-overlapping source question ranges.  This is supervised adaptation to the
RULER task family, not zero-shot RULER evidence.

## Stop condition

Stop after the EVQ screen if it has no material 8K capability and no directional
16K transfer.  Do not spend a second arm on a failed recipe.  Any reviewer-facing
claim must report the complete task family and clearly label task-family
supervision.

