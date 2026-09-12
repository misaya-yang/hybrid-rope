# 151.9M fixed-support joint-allocation experiment

This directory implements S1 from
[`docs/research/next_stage_20260912/index.md`](../../docs/research/next_stage_20260912/index.md):
fixed Geo, fixed endpoint-anchored Cosh, and Geo-initialized full-z jointly
trained with all model weights at actual sequence length 2048.

- `protocol.py` locks the two supports, three arms, optimizer semantics, and
  999,948,288-token trajectory. The available frozen 499,974,144-token source
  prefix is traversed twice with a separately seeded deterministic permutation
  per epoch; this is two epochs, not one billion unique tokens.
- `learnable_rope.py` provides 31 positive normalized gaps (30 effective
  degrees of freedom), exact fixed endpoints, exact Geo initialization, and no
  shape projection. Subtracting the common logit mean only fixes the softmax
  gauge.
- `run.py` runs a discarded ten-update qualification or one complete arm. A
  full-z qualification must observe finite nonzero allocation gradients, an
  optimizer update, finite loss, and actual throughput/memory before the
  paired block may start.
- `preflight.py` checks the remote data payload, exact Geo initialization, and
  paired model-weight identity. `queue_first_block.py` is the server-side
  state machine that waits for the final S6 endpoint job, runs the preflight
  and qualification, and only then starts the three-arm block.
- `prepare_eval_data.py` builds a new shared-endpoint 16K panel from frozen
  shard 004 while excluding the earlier 8K validation document IDs.
  `evaluate.py` records per-document whole-window and tail-128 NLL at
  2K/4K/8K/16K. `queue_evaluation.py` connects both common checkpoints from
  all three arms to that evaluation after the training supervisor completes.
- `queue_seed_block.py` serializes a preregistered additional training seed
  after the preceding evaluation. It requires the earlier paired block to be
  complete, but an evaluator engineering failure does not invalidate or cancel
  the already-declared training repeat. A 5 GiB free-space gate protects every
  new arm.

The first scheduled block is one complete `support=500000, seed=42` trio. It
is an execution block, not a three-seed conclusion; the remaining predeclared
supports and seeds are not selected according to this block's OOD outcome.
