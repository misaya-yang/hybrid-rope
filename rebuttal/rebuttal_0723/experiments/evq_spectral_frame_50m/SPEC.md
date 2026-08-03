# EVQ Spectral-Frame 50M Experiment

Date: 2026-07-29

Status: `CPU_READY_GPU_RUNTIME_UNVERIFIED_NOT_TRAINED`

## Research question

Can a from-scratch, single-path RoPE model trained only at physical length
\(L=128\) retain Native in-window language modeling while improving raw
perplexity at \(2L\) and \(4L\)?

This is a next-method experiment, not completed rebuttal evidence. Perplexity
does not establish retrieval, QA, RULER, complete-string generation, or
terminal EOS capability.

## Existing evidence and smallest missing evidence

The completed full-band EVQ-LeRoPE runs used the same 50.928M architecture,
15M-token FineWeb-Edu tensor, and 128/256/512 evaluation:

- seed 42 EVQ-LeRoPE minus Native NLL:
  `-0.00998/-0.05550/-0.12149`;
- seed 43:
  `+0.02059/-0.01081/-0.07490`.

The extrapolation direction is consistent, but Native in-window
non-inferiority is not. The smallest missing evidence is whether three
frequency constructions that separately address short-window frequency
drift, long-range average coherence, and shared-grid capacity reduce that
remaining trade-off. The experiment also includes Native-LeRoPE and a
frequency-multiset-matched head control so a gain is not automatically
attributed to EVQ initialization or head factorization.

## Scientific contract

All arms preserve:

- architecture: 50.928M decoder-only Transformer, 6 layers, 8 heads,
  `d_head=64`;
- tokenizer/vocabulary: the existing 50,304-token FineWeb-Edu artifacts;
- training: physical length 128, 15,000,000 requested tokens, global batch
  128, 915 optimizer steps;
- optimizer semantics: AdamW, peak LR `6e-4`, betas `0.9/0.95`, weight decay
  `0.1`, identical schedule and data order within seed;
- common endpoints where applicable: base `500,000`, paper-midpoint EVQ-Cosh
  `tau=5`; the integer-period grid deliberately changes the observed-band
  internal shape and is not an exact endpoint/range-matched EVQ ablation;
- evaluation: the same held-out validation tensor, fixed paired offsets,
  lengths 128/256/512, 32 chunks per length;
- raw inference: no YaRN, position interpolation, length routing, or virtual
  position gaps. The integer-period grid is designed once for the registered
  4x study range and is then fixed across 128/256/512; it is not
  target-range agnostic.

Seeds are `42/43/44`. A seed interrupted before its result is written is not a
result.

## Arms

### Existing controls

1. `native`: fixed standard geometric RoPE.
2. `evq_fixed`: fixed paper-midpoint EVQ-Cosh.
3. `native_lerope`: Native initialization with all 32 shared per-band
   log-frequency residuals learnable.
4. `evq_lerope`: EVQ initialization with all 32 shared per-band log-frequency
   residuals learnable. This reproduces the method identity of the completed
   two-seed arm.

### Candidate A: `phase_observed_evq`

\[
\omega_k=\omega_k^{EVQ}\exp\left(
 m_kr_k\tanh(\alpha_k)\right),\qquad
m_k=\mathbf 1\left[\frac{L\omega_k^{EVQ}}{2\pi}\ge1\right].
\]

Only bands completing at least one cycle in training may learn. At the
registered configuration, bands `0..21` learn and `22..31` stay exactly EVQ.
Here \(r_k\) is 0.45 times the smaller adjacent EVQ log-frequency gap
(the sole adjacent gap at an endpoint). This makes each learned residual
bounded and preserves strict ordering by construction. “One cycle” is a
registered heuristic mask, not a proof of statistical observability. This is
the observed-band hypothesis under a fresh three-seed comparison.

### Candidate B: `integer_period_coherence_evq`

Start from EVQ wavelengths. For the 22 bands completing at least one training
cycle, select strictly increasing integer-token periods by deterministic
coordinate descent. The objective is:

\[
\max_{d\in[L,4L]} |K(d)|
+\frac12\mathbb E_{d\in[L,4L]}K(d)^2
+2\,\mathbb E_{d\in[1,L)}(K(d)-K_{EVQ}(d))^2,
\]

where \(K(d)=K^{-1}\sum_k\cos(\omega_kd)\). The remaining ten slow bands stay
exactly EVQ. The generated grid must improve both external maximum coherence
and external mean-squared coherence relative to fixed EVQ, keep local-kernel
MSE below `0.02`, remain strictly decreasing, and be recorded by hash before
training.

This is a fixed, zero-learned-parameter schedule. It is not an equiripple
solution: coordinate descent minimizes the stated discrete objective only.
Integer periods also create exact aliases within individual bands, so the
average kernel may hide sub-band concentration. The kernel objective is a
design diagnostic, not a claim that it predicts trained-model loss.

### Candidate C: `head_factorized`

Use a fixed direct sum across the eight heads:

- heads `0..3`: Native geometric grid;
- heads `4..5`: fixed EVQ-Cosh grid;
- heads `6..7`: fixed integer-period coherence grid.

There is one ordinary attention and one KV cache. No request-length routing,
additional branch, learned frequency, or extra model dimension is introduced.
The arm tests whether separate heads can learn local, intermediate, and 4x
roles without forcing every head through one compromise grid.

### Attribution control: `within_head_mixed_control`

This arm uses the exact same global frequency multiset as `head_factorized`:
four Native, two EVQ, and two integer-period copies of every band across eight
heads. The sources are distributed cyclically within each head and each
head's final grid is sorted. It removes head-level grouping while preserving
the total frequency budget, allowing the result to distinguish head
factorization from merely adding the mixed spectrum.

## Gates

The 50M experiment is a frequency-screening study.

Per seed:

1. 128 NLL must be finite and no worse than Native by more than `+0.02`.
2. 256 and 512 NLL must both be below Native.
3. No frequency grid may be non-finite or unordered within a head. Fixed
   grids must retain their registered hash; learned grids must retain their
   registered frozen bands.

Aggregate promotion requires:

1. all three seeds completed under one immutable contract;
2. mean candidate-minus-Native NLL at 128 no greater than `+0.01`;
3. all three seeds improve over Native at 512;
4. paired bootstrap intervals are reported separately within each seed over
   fixed evaluation chunks; seeds are summarized by mean and range rather
   than pooled as independent chunks;
5. the winning final frequency identities are preserved.

Passing these gates justifies a larger capability experiment. It does not by
itself justify “capability maintained through 4x.”

## Runtime and cost discipline

The launcher uses:

- BF16 autocast;
- Flash-only SDPA;
- fused AdamW;
- TF32 matmul;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- persistent TorchInductor cache;
- `torch.compile(mode="max-autotune-no-cudagraphs")` when the runtime probe
  is selected only if a five-step cold/steady comparison estimates lower
  total runtime than eager execution;
- evaluation batches capped by total input tokens rather than one example per
  forward, after CUDA batch-1 versus registered-batch NLL parity is verified.

Global batch and data order do not change when micro-batch/accumulation is
adjusted. The first GPU use must be a discarded runtime probe that compares
eager and compiled cold/steady costs, records finite loss, peak memory,
throughput, Flash eligibility, and the exact selected micro-batch. No GPU
launch is authorized by this file.

## Stop condition

Stop this 50M line if none of the candidates passes the registered three-seed
PPL gates. Do not rescue it by changing the evaluation metric, adding YaRN,
using target-length position exposure, or reporting only the best seed.
