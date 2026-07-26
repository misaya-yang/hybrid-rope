# Post-Hoc RoPE Frequency Transplant: Exact Compensation Obstruction

Date: 2026-07-26

Status: `INTERNAL_DIAGNOSTIC_NOT_A_REBUTTAL_CLAIM`

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

Evidence role: explains the boundary of the mature-model LoRA result; it is not
a new paper theorem or a completed method

## Reviewer question, existing evidence, and stop condition

1. **Concern addressed.** Can EVQ be inserted into a mature pretrained model
   and retain useful capabilities after a small LoRA adaptation?
2. **Existing evidence.** Under matched 4K-only Q/K/V/O LoRA, EVQ reaches
   69/100 and 67/100 strict exact on the trained RULER `niah_single_1` family
   at 8K, whereas Native reaches 0/100. On held-out 4K RULER tasks, however,
   untouched Native retains 55% UUID retrieval and 25% variable tracking,
   while EVQ immediately after injection, after 20M natural tokens, and after
   the final routing stage scores 0% on both.
3. **Smallest missing explanation.** Determine whether a post-hoc change of
   RoPE frequencies is merely a constant reparameterization that a
   position-independent Q/K adapter can exactly undo.
4. **Smallest executable diagnostic.** Prove the all-content/all-position
   linear case and measure the actual Native-to-EVQ phase displacement and
   effective LoRA update norms. No new training is required.
5. **Stop condition.** Do not start another rank, loss, hybrid, or schedule
   sweep. If exact constant compensation is obstructed and the completed
   held-out task gate is negative, keep the mature-model claim narrow and
   defer any new architecture to a separately authorized experiment.

## Result in one sentence

Full post-hoc EVQ replacement is not a harmless change of coordinates: except
for frequency permutations/sign aliases, no fixed position-independent Q/K
maps can exactly preserve the original RoPE bilinear form at every relative
position. The actual OLMo-2 Native-to-EVQ replacement changes 63 of 64 rotary
pairs and produces large phase displacement already inside 4K.

This result explains why natural-text NLL can recover while pre-existing
routing behavior does not. It does **not** prove that a nonlinear transformer
cannot approximately relearn a finite task distribution.

## Exact linear obstruction

Consider one rotary pair. Absorb the original content projections into
\(x,y\), and write the Native relative-position bilinear form as

\[
f_{\omega}(x,y,\Delta)
  = x^\top R(\omega\Delta)y,
\]

where \(R(\theta)\) is the two-dimensional rotation matrix. After replacing
\(\omega\) by \(\omega'\), grant the adapted model arbitrary invertible,
position-independent maps \(A,B\), which is more freedom than an ordinary
low-rank additive LoRA update:

\[
\tilde f(x,y,\Delta)
  = x^\top A^\top R(\omega'\Delta)B y.
\]

### Proposition

If

\[
A^\top R(\omega'\Delta)B = R(\omega\Delta)
\]

for every content vector and every real \(\Delta\), then
\(|\omega'|=|\omega|\). For integer-only positions the corresponding condition
is equality up to sign and \(2\pi\) aliasing.

### Proof

At \(\Delta=0\),

\[
A^\top B=I,
\qquad B=A^{-\top}.
\]

Therefore

\[
A^\top R(\omega'\Delta)A^{-\top}=R(\omega\Delta).
\]

The two rotations must be similar for every \(\Delta\). Similarity preserves
trace, hence

\[
2\cos(\omega'\Delta)=2\cos(\omega\Delta)
\]

for every \(\Delta\), which implies \(|\omega'|=|\omega|\). Equivalently,
differentiating the one-parameter groups at zero requires their skew-symmetric
generators to be similar and therefore to have the same eigenvalues
\(\{\pm i\omega\}\).

For multiple rotary pairs, the same argument applies to the block-diagonal
generator. Universal constant compensation is possible only when the
frequency multiset is preserved up to permutation, sign, and—on integer
positions—\(2\pi\) aliases. OLMo-2 frequencies lie in \((0,1]\) radians per
token, so the integer-position alias does not identify the Native and EVQ
tables.

### Scope of the proposition

The proposition assumes universal equality over rotary content and positions.
A real transformer can exploit restricted activation subspaces, nonlinear
layers, V/O updates, residual paths, and a finite training distribution.
Consequently:

- it is an obstruction to **exact function preservation by a fixed Q/K
  reparameterization**;
- it is not an impossibility theorem for approximate retraining;
- it does not identify which layer or projection causally produces an observed
  error.

## Measured phase displacement

The audit recomputed the exact frozen frequency identities:

| Frequency table | SHA-256 |
| --- | --- |
| Native float32 | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| EVQ float32 | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |

EVQ preserves pair 0 and changes every other pair. Among 64 pairs, the number
whose maximum **unwrapped** phase displacement over
\(\Delta=0,\ldots,4095\) stays below a threshold is:

| Threshold | Pairs |
| ---: | ---: |
| 0.1 rad | 8/64 |
| 0.5 rad | 13/64 |
| 1.0 rad | 15/64 |
| \(\pi\) rad | 19/64 |
| \(2\pi\) rad | 22/64 |

At fixed relative distances:

| \(\Delta\) | Mean absolute wrapped error | Median | P90 | Pairs below 0.1 rad | Mean phase-alignment cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.8833 | 0.5697 | 2.4078 | 20/64 | 0.4716 |
| 512 | 1.2394 | 0.8831 | 2.7480 | 15/64 | 0.2094 |
| 1,024 | 1.0565 | 0.8906 | 2.4678 | 13/64 | 0.3702 |
| 2,048 | 1.3559 | 1.2948 | 2.8791 | 12/64 | 0.1363 |
| 4,095 | 1.2130 | 0.9526 | 2.6424 | 11/64 | 0.2462 |
| 8,191 | 1.2980 | 1.0905 | 2.8986 | 9/64 | 0.2010 |
| 16,383 | 1.3115 | 1.3826 | 2.8344 | 7/64 | 0.1817 |

Here “mean phase-alignment cosine” is
\(\frac1K\sum_k\cos((\omega'_k-\omega_k)\Delta)\). It is a frequency-table
diagnostic, not a measured attention-score similarity.

The displacement is highly nonuniform. For example, EVQ-to-Native frequency
ratios at pair indices 8, 24, 32, and 40 are respectively 2.23, 7.49, 10.40,
and 10.93. Their 4K unwrapped phase displacements are approximately 973, 194,
54, and 11 radians. Thus the transplant is already a large change in the
model's trained coordinate system within the original 4K window.

## Effective LoRA-weight audit

For each adapter module, the audit computes
\(\|\frac{\alpha}{r}BA\|_F/\|W_{\text{base}}\|_F\) without materializing a
full dense delta. Values below are means across the 16 layers.

| Adapter | Q | K | V | O | Q/K fraction of LoRA delta energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Stage A | 0.0521 | 0.0665 | 0.0318 | 0.0390 | 66.70% |
| Native final | 0.0545 | 0.0691 | 0.0338 | 0.0413 | 65.88% |
| EVQ Stage A, seed 20260725 | 0.0934 | 0.0979 | 0.0346 | 0.0398 | 81.86% |
| EVQ final, seed 20260725 | 0.0969 | 0.1013 | 0.0367 | 0.0421 | 81.26% |
| EVQ Stage A, seed 20260726 | 0.0937 | 0.0989 | 0.0351 | 0.0401 | 81.82% |
| EVQ final, seed 20260726 | 0.0968 | 0.1024 | 0.0371 | 0.0424 | 81.19% |

The replicated EVQ runs independently place substantially more effective
adapter energy in Q/K than Native. Stage A already contains most of the final
update. The median Stage-A-to-final effective-matrix cosine is 0.97–0.99 in
every projection, so the routing stage is mostly an aligned refinement rather
than a wholesale replacement of the Stage-A adapter.

These norm results are descriptive. They are consistent with the EVQ arms
spending capacity on the changed positional coordinate system, but they do not
prove that Q/K compensation causes the held-out-task failures.

The CPU diagnostic JSON has SHA-256
`14894531c6480f3f663fcfe189765384b59a7aa9438fe428b55e6ae66177cb96`.
It records the checkpoint and all six adapter hashes, per-module rows, phase
statistics, and the explicit non-causal claim boundary.

The exact analysis-script SHA-256 is
`72197841cc2dac1a2944500f3138ff76c80e19833e9a716d92cc3bd2dbfd02aa`.
The read-only two-file evidence archive, copied back and verified with
`tar -tzf`, has SHA-256
`7ca367ea9078d0f8e4c6ec9d111b0a6f6612b5e8c84f8f279dded6e7b5facfd5`.

## Maturity-stage decision rule

| Model state | Scientifically clean use of EVQ | Required evidence |
| --- | --- | --- |
| From scratch | Install EVQ before step 1 so all weights co-adapt to the frequency allocation | Matched full-training Native/EVQ, seeds, PPL plus capability metrics |
| Early checkpoint | Full transplant is a continuation intervention, not a pure initialization comparison | Fixed recipe across maturity anchors; in-range behavior-retention gate plus long-context gain |
| Mature Base/Instruct | Do not assume short LoRA exactly preserves functions after full transplant | Untouched Native capability controls, task-diverse retention, and autoregressive long-context evaluation |

For the current mature Instruct model, adding more LoRA rank or more examples
from the same `niah_single_1` family would not answer the observed failure.
The positive result is already stable across two seeds; the missing evidence is
cross-task function retention, where the completed gate is negative.

## Deferred design space, not execution authorization

A future post-hoc method would need to be function-preserving at
initialization—for example, retaining the Native branch and adding a
zero-initialized gated EVQ branch—rather than directly replacing some or all
frequencies. Such a method changes parameters and runtime and would require its
own matched control. It is not the submitted EVQ method and was not run here.

Any future run must first pass a CPU unit test proving exact equality to the
untouched Native model at gate value zero. A paid-GPU experiment would then
need pre-registered 4K capability-retention and natural-text NLL gates before
testing 8K. A DC subspace, larger rank, and further single-task training are
not authorized by this diagnostic.

### Correction: historical direct hybrids were not actually hybrids

The historical zero-training partial-pair, log-blend, and per-head probes
cannot be used to reject direct hybridization. The evaluator retained a view
of the CPU float32 Native `inv_freq` buffer, then patched that buffer to EVQ in
place. Both the “Native” and EVQ variables consequently referred to the
post-patch values.

An independent reconstruction from each receipt's declared Native/EVQ arrays
and partition metadata found:

| Audit item | Result |
| --- | ---: |
| Historical hybrid receipts | 28 |
| Recorded active hash matches declared hybrid | 0/28 |
| Recorded active hash matches alias-bug reconstruction | 28/28 |

The old zero scores therefore measure the alias-bug frequency tensors, not the
declared pair/head/blend hybrids. They are invalid method evidence. The fixed
evaluator clones both snapshots and has 12 passing CPU identity tests. No
corrected GPU rerun was launched.

The audit JSON SHA-256 is
`82a58d0e019bce450dba5c5197133d1c5519c80ff57b08258752fde244eb68c3`.
The frozen audit package—28 historical receipts, buggy and fixed code,
auditor, and regression test—has SHA-256
`eae8efd0015ab1a0ae5114ba9fc6367da5aa9f023408063bce8441e1f90b1d6c`.

## Reviewer-safe use

The mature-model result may safely support:

> Under a matched 4K-only LoRA protocol, EVQ enables reproducible 2x
> autoregressive retrieval on the trained task family (69/100 and 67/100
> across two training seeds versus 0/100 Native).

It must be accompanied by:

> Full post-hoc frequency replacement does not preserve all existing
> capabilities: two held-out 4K RULER tasks fall to zero despite strong
> natural-text NLL. We therefore treat mature-model adaptation as bounded
> supporting evidence, not general downstream or no-forgetting closure.

The paper's cleanest EVQ claim remains training-time frequency allocation in
models that co-adapt from initialization. This audit narrows the mature-model
extension; it does not weaken the controlled from-scratch allocation result.
