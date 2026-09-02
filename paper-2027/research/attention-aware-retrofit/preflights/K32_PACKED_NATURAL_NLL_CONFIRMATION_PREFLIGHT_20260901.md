# K32 packed-natural NLL confirmation — preregistration

**Status:** `FROZEN_FOR_WORK_MACHINE / DATA_READY / MODEL_NOT_RUN`

The full-RULER entrance completed as `CLEAR_ADVANCE`. The model-free packed
input is hash-bound, but user instruction stops this GPU instance before model
evaluation. Execute only on the work machine under the unchanged protocol.

## Question

If the new-seed full-RULER confirmation does not reject frozen
normalized-index, does the same single static table also retain in-window
natural continuation and improve 64K likelihood relative to Native and
official-equation YaRN?

This is a likelihood double-check for an already-frozen engineering
representative. It cannot select a profile, boundary, gain, scale, dataset row,
or packing rule.

## Frozen data

- source: FineWeb-Edu `sample/10BT/001_00000.parquet`, expected SHA-256
  `3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693`;
- start row: `650000` on the 729000-row parquet;
- select source-order unique documents with no model score or benchmark read;
- create 32 non-overlapping 65536-token streams;
- insert one tokenizer EOS between complete documents;
- truncate only the final document needed to fill a stream and discard its
  unused suffix rather than reusing it;
- define each paired 32768-token input as the suffix of its 65536-token stream,
  so both lengths have identical final 256 targets;
- write complete token IDs only to the external raw owner; compact receipts
  contain hashes and source-row ranges, not content.

The shard is repository-known and has historical uses in other protocols.
“New” here means these fixed packed rows and their outcomes were not used to
select the current K32 profile; it does not mean the corpus is absent from all
past training or from the external checkpoint's unknown pretraining mixture.

## Frozen arms and execution

The checkpoint is the exact Qwen2.5-0.5B-Instruct K32 artifact used by the
RULER confirmation. Evaluate exactly:

1. Native, amplitude `1`;
2. normalized-index tensor
   `8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f`,
   amplitude `1.0512928913614359`;
3. official-equation YaRN-s2 tensor
   `d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59`,
   amplitude `1.0693147180559945`.

Every profile is loaded before inference and remains fixed for each complete
forward. Use `use_cache=False`, no compilation, no model updates, and score
only the final 256 aligned next-token targets. There is no physical arm,
per-length table, gain sweep, row exclusion, or repeated dataset draw.

## Entrance and stop

The model-free CPU data artifact may be prepared before the entrance result;
it cannot read model outcomes and does not authorize GPU evaluation. Execute
the model only if the registered full-RULER result is `CLEAR_ADVANCE` or
`COMPETITIVE_UNRESOLVED`. A full-RULER `BASELINE_LOSS`, identity mismatch,
non-finite NLL, incomplete paired grid, or profile mutation stops this branch.
No failure authorizes another natural split or profile change.

## Decision rule

Resample the same 32 stream indices jointly across all arms and lengths with
10,000 paired bootstrap replicates, seed `202609030`.

- Native compatibility: 32K index PPL retention
  `exp(NLL_native - NLL_index)` must be at least `.875`.
- Extension resolver: the 95% interval for 64K index-minus-Native NLL must have
  upper endpoint below zero.
- Baseline contrast: index is favored over YaRN only if the 64K
  index-minus-YaRN NLL interval has upper endpoint below zero; YaRN is favored
  only if its lower endpoint is above zero; otherwise unresolved.

Report all mean NLL/PPL values and both intervals. This endpoint is paired
final-256 teacher-forced continuation, not dense NLL, generated exact match,
natural QA, K causality, or a broad SOTA claim.
