# 151.9M fixed-support joint-allocation experiment

This directory implements S1 from
[`docs/research/next_stage_20260912/index.md`](../../docs/research/next_stage_20260912/index.md):
fixed Geo, fixed endpoint-anchored Cosh, and Geo-initialized full-z jointly
trained with all model weights at actual sequence length 2048.

## Read-only audit of the apparent historical reversal (2026-09-12)

The reported seed42 anchored-Cosh result is not a replication of historical
direct-midpoint EVQ. Do not generalize it to a failure of historical EVQ or use
full-z's improvement as evidence that the Cosh construction was invalidated.
This audit read source code and evaluated 32 scalar frequencies with Python's
standard-library math; it launched no model, GPU evaluation, training or hash
scan and changed no execution queue.

At K=32, base=500000 and tau=sqrt(2), [protocol.py](protocol.py) anchors the
midpoint quantiles to the sampled Geo endpoints. The historical
[direct-midpoint builder](../../scripts/lib/rope/schedules.py) instead uses
base^(-q) directly, as invoked by
[Phase17C](../../scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py).
The new/old frequency ratio ranges from 1.138332858 to 1.138373701 across all
32 slots. The new table is therefore almost a uniform 13.8% acceleration of
that historical table: endpoints 1 and 3.013858e-6 versus 0.878446 and
2.647607e-6. This is a confirmed construction difference, not a demonstrated
cause of the performance reversal. Removing the old table's global slowdown
does not by itself establish that the slowdown supplied the old gains.

The older successful [750M 4K continuation](../../scripts/core_text_phases/phase15_750m_2k_to_4k_continue_ckpt_eval.py)
is a separate counterexample to an anchoring-only explanation: it already uses
both sampled endpoints and r=0, with inclusive nodes and tau=1.5. Relative to
that table, the new S1 frequencies range from 0.839906 to 1 times the old
frequencies. Its success cannot be explained away as the unanchored midpoint
case. The older [2K result](../../docs/exp/2026-03/2026-03-11_phase17c_2048_continue_results.md)
and [4K result](../../docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md)
remain part of the relevant evidence.

Additional source-level differences are explicit, not identified causes:
S1 uses LR6e-4, weight decay0.01, 1525/15258 warmup updates, and two passes
over a 499974144-token prefix. The old run_evq_sweep 125M default uses LR3e-4,
weight decay0.1 and 2% warmup; Phase17C uses a staged 454M lineage rather than
a fresh 151.9M initialization. S1 evaluates shared-endpoint suffixes of 512
selected documents of at least 16385 tokens, whereas the old eval_model
samples length-specific windows from a cached concatenated stream. The new
model source also explicitly keeps residual activations and applied sin/cos
in BF16 under autocast, unlike the legacy source's FP32 residual path.

Static inspection found no obvious Cosh/Geo arm reversal, missing tau factor,
reversed frequency ordering, or incorrect next-token target shift. The queue
maps each arm to its own checkpoint; fixed arms construct the matching table
before loading, while full-z derives the table from loaded gap parameters.
These are source-level observations, not an on-device runtime validation.
The actual deployed original-root snapshot, saved frequency buffers, data
manifest and raw evaluation rows are still needed to close the run-specific
audit; local recipe defaults are not proof of a historical run's exact state.
No additional training is authorized or required by this audit note.

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
