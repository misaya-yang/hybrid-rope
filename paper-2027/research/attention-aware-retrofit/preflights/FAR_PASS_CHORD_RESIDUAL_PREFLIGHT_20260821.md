# Released-Native far-pass phase-chord residual

- **Date:** 2026-08-21
- **Status:** `NO_GPU_READY / TRAINING_DATA_COMPLETE / GPU_SMOKE_PENDING`
- **Role:** internal next-experiment owner; not manuscript evidence
- **Base:** OLMo-2-0425-1B-Instruct, Native RoPE
- **Frozen parent:** unmodified released Native checkpoint; no inherited
  adapter or historical binary artifact

## Decision

The next paid experiment is one Native-preserving far-pass chord residual
arm. It replaces the superseded full-spectrum EVQ residual design. No
whole-table morph audit, Stage-D protection diagnostic, phase-chord seed 256,
8B run, parameter sweep, or external-method gate precedes it.

The previous plan incorrectly depended on a historical RULER adapter and data
view whose binary artifacts no longer exist. The registered experiment now
starts only from assets present on the current host. This also makes the test
scientifically cleaner: every released-model parameter and the complete Native
attention path remain frozen, while a small independent residual learns from
natural documents generated without RULER or NIAH. RULER is evaluation only.

The missing evidence is whether a 4.72M-parameter attention plug-in can add
held-out long-context capability while every short request remains exactly the
released model. Historical Native/EVQ results are contextual targets, not
inputs, matched controls, or replaceable raw evidence for this run.

## Operator

For a long request, one attention layer adds

\[
s^{\rm chord}_{qt}
=g(p_q)\,\gamma\,
u_q^\top\!\left[I-R_{\Omega}(p_t-p_q)\right]v_t,
\qquad
u=A^q h,\;v=A^k h.
\]

The implementation concatenates

\[
q=[q^N,\sqrt\gamma g u,-\sqrt\gamma g R(p_q)u],\quad
k=[k^N,\sqrt\gamma v,\sqrt\gamma R(p_t)v],\quad
v=[v^N,0,0],
\]

then uses one SDPA call and one softmax with the Native `1/sqrt(128)` score
scale. Only the first 128 value coordinates pass to the frozen Native output
projection.

The scalar bilinear contains sine and `1-cos` components. The precise link to
the phase-chord discovery is that its isotropic squared perturbation energy is
proportional to `1-cos(omega*Delta)`. The residual is exactly zero at
`Delta=0`; it is not correct to claim that the scalar response itself equals
`1-cos`.

Requests with total budget at most 4096 directly call the released Native
attention path. The residual route is selected before prefill and cannot be
changed after a KV cache exists.

## Registered frequency band

Eight log-spaced wavelengths cover `[2 L_target, 10 L_target]` for
`L_target=16384`, i.e. 32768--163840 tokens. The fastest pair reaches pi phase
at 16K up to float32 roundoff, so the registered target range contains no
intentional phase wrap.

| Item | Registered value |
| --- | ---: |
| Residual pairs / dimensions | 8 / 16 |
| Augmented head dimension | 160 |
| KV-cache width relative to Native | 1.25x on long requests; unchanged on short requests |
| Mean chord energy, uniform 0--4K | `0.0333316` |
| Mean chord energy, uniform 4K--16K | `0.526731` |
| Far/near energy ratio | `15.8027` |
| Float32 frequency SHA-256 | `b824981602b75f5c7007e34d6a88f581996da7a7833f458ffd8b3aabcb5b5533` |
| Trainable parameters | `4,718,608` in 80 tensors |

This band is a fixed construction, not a sweep result or an optimality claim.

## One-arm protocol

- Freeze every released checkpoint parameter; install no parent adapter.
- Train only residual Q/K rank-64 projections and one positive gain per layer.
- Train on 1,024 independently prepared FineWeb-Edu natural-span retrieval
  rows: 896 train and 128 validation. Each row contains a unique 8-token
  natural anchor and supervises the following 8 tokens plus terminal EOS.
  The data contains zero RULER/NIAH-generated rows. Manifest SHA-256:
  `6293d0dbb15235a9ba2faf22be8f07176cab377b1490ef01c04c2163c0fb2852`.
- Run 300 optimizer steps, micro-batch 4, accumulation 2, LR `5e-5`, twenty
  warmup steps, fused AdamW, BF16, physical length at most 4096, and explicit
  target-range positions. Every micro-batch contains two 8K-phase and two
  16K-phase rows; no optimizer time is spent on the structurally inactive 4K
  residual route.
- Disable gradient checkpointing and use the repository RTX 5090 profile:
  Flash-only SDPA, `max-autotune-no-cudagraphs`, expandable segments, and a
  persistent TorchInductor cache. The GPU smoke compiles the exact registered
  training shape, runs one cache-warming forward/backward followed by one
  timed steady-state forward/backward, and records throughput plus peak
  memory.
- Supervise complete answer tokens plus immediate EOS.
- Do not train a second arm or tune rank, gain, band, pair count, steps, or
  threshold.

## GPU smoke and stop rules

The smoke must pass before training:

1. short-route logits are bitwise equal to the loaded Native parent;
2. the registered frequency hash and 4,718,608-parameter scope match;
3. BF16 Flash-only SDPA forward/backward accepts head dimension 160 with
   math, memory-efficient, and cuDNN fallbacks disabled;
4. the equal-position chord residual is numerically zero;
5. residual gradients are finite and nonzero;
6. augmented prefill and decode use one fixed 160-wide cache.

Any failure stops without changing the backend or scientific protocol.

After training, generate a fresh independent-seed RULER matrix from the pinned
official repository and run the complete 13 families with 20 rows per family.
Evaluate released Native at 4K/8K/16K and the residual at 8K/16K on identical
rows. A separate candidate 4K generation pass is unnecessary because the
short route is the exact released model and the GPU smoke checks its logits
bitwise. RULER generation and evaluation are held out from training by both
generator and rows.

The matched causal comparison is released Native versus released Native plus
the residual on the fresh rows. A headline-worthy result needs a material 8K
macro gain, non-zero gains beyond a single retrieval family, and measurable
16K capability, all while preserving the short path exactly. Historical EVQ
values (`31.63%` at 8K and `5.03%` at 16K under its own task-family-adapted
protocol) are stretch context only and must not be called matched thresholds.

## Prepared code and current asset boundary

The existing compatibility paths now implement the registered operator:

- method/adapter: `far_only_evq_residual.py`;
- no-GPU preflight: `preflight_4k_far_only_evq_residual.py`;
- GPU smoke/training: `train_4k_far_only_evq_residual.py`;
- strict evaluation: `evaluate_instruct_ruler_transfer.py` and
  `evaluate_2wiki_phase_adaptation.py`;
- exact remote entry points:
  `run_far_pass_chord_released_native.sh`;
- focused tests: `tests/test_olmo2_far_only_evq_residual.py`.

The current no-GPU host now has the exact released checkpoint, checkpoint
receipt, pretokenized FineWeb-Edu source, complete natural-span phase-training
view, pinned official RULER source, runtime dependencies, and a passing no-GPU
prepared receipt. No historical adapter or historical data view is required.
The platform exposes only about half a CPU in no-card mode, so the full fresh
RULER evaluation JSONL generation is registered for the first CPU-capable
window rather than burning hours before GPU activation. GPU work remains
blocked only on the real D160 Flash smoke and explicit authorisation.
