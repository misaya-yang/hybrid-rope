# Research timeline

This is a chronological ledger for cold start. It answers **what we did, what
we observed, how our interpretation changed, and where the work stopped**.
It is not a fourth authority: rules remain in [`../../../AGENTS.md`](../../../AGENTS.md),
owner routing and durable agenda remain in [`../../../INDEX.md`](../../../INDEX.md),
and volatile state remains in [`../../HANDOFF.md`](../../HANDOFF.md). Every
number below defers to its linked owner.

## 2026-02 — early finite-frequency evidence

- **Did:** established short-context baselines, then ran 50M/125M finite
  EVQ-Cosh tau grids on from-scratch language modelling. Historical reports are
  grouped under [`docs/exp/2026-02/`](../../../docs/exp/2026-02/).
- **Observed:** the finite grid was non-monotone; the useful point was not a
  smooth monotone continuation from small tau. Cross-seed direction was more
  stable than effect size.
- **Changed our view:** tau became a fallible finite-grid operating prior, not a
  continuous optimum certificate.
- **Evidence status:** historical/system evidence; current causal ownership is
  later exact-range work.

## 2026-03 — scale, tasks, composition, and failure conversion

- **Did:** ran 454M passkey/PPL composition, L=256 extrapolation, 125M
  Kerple+MLP compatibility, 750M continuation, the 99-run formula sweep,
  QuALITY diagnostics, Video-DiT, and GQA/MLA channel-scarcity studies. Reports
  are under [`docs/exp/2026-03/`](../../../docs/exp/2026-03/).
- **Observed:** long PPL, teacher-forced retrieval, autoregressive exactness,
  and downstream QA did not convert automatically into one another. The same
  range operator had substrate-dependent leverage. The 750M run preserved a
  small in-window/large-OOD crossover, while the QuALITY pilot exposed protocol
  and scoring mismatch rather than a clean downstream verdict. Scarce-channel
  gains were large in the extreme MLA cell but not monotone across architecture
  cells.
- **Changed our view:** systems breadth is valuable, but each endpoint and
  operator identity must remain protocol-specific. Formula-centred grids show
  a tested basin, not near-optimality over a continuum.
- **Evidence status:** 750M persistence and 432M MLA breadth remain supporting
  owners; early claims and labels are bounded by current provenance audits.

## 2026-04 to 2026-06 — submission hardening and provenance repair

- **Did:** prepared the NeurIPS submission, tightened figures/tables, and built
  code/result/provenance manifests under [`docs/overview/`](../../../docs/overview/).
- **Observed:** several historical labels overstated implementation identity or
  seed scope; nearby scripts/results were not automatically matched evidence.
- **Changed our view:** a plan, script, checkpoint inventory, or paper row is
  not a result owner. Reviewer-safe numbers require explicit method, data,
  endpoint, seed, and hash/provenance boundaries.

## 2026-07 — rebuttal, mature models, and exact-range controls

- **Did:** audited 8B LoRA temporal NLL, 151.9M Native/EVQ controls, official
  versus repository YaRN-style components, M4 exact-range factorials, and the
  OLMo-2 1.485B mature-checkpoint route. Historical experiment reports are in
  [`docs/exp/2026-07/`](../../../docs/exp/2026-07/); rebuttal owners are under
  [`rebuttal/rebuttal_0723/theory_results/`](../../../rebuttal/rebuttal_0723/theory_results/).
- **Observed:** fixed-support interior allocation remained active in M4; a
  matched non-Cosh analytic shape was competitive, so Cosh uniqueness did not
  follow. Mature-model adaptation produced protocol-specific long capability.
  Exact position-independent invertible Q/K compensation is obstructed for
  unequal frequency multisets.
- **Changed our view:** pure allocation identification, systems composition,
  mature adaptation, and frozen retrofit are separate estimands. EVQ-Cosh is a
  closed-form construction and intervention, not a universal optimum.

## 2026-08-19 to 2026-08-20 — ICLR causal core

- **Did:** completed the full-RoPE sin/cos geometry audit and the raw-hash-
  receipted 151.9M three-seed exact-range replication. Owners:
  [`foundations/FULL_ROPE...`](../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
  and [`evidence/EXACT_RANGE...`](../evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md).
- **Observed:** at fixed sampled support, moving only 30 interior frequencies
  changes trained behaviour with the same OOD direction in 3/3 seeds; when
  both grids are target-matched, the tested ordering reverses. Static
  collision/stable-rank objectives admit Fourier/ordering counterexamples and
  do not rank LM quality.
- **Changed our view:** support `(a,R)` and normalized allocation `z` are
  distinct but interacting causal coordinates. The paper's geometry claim is
  phase-invariant redundancy/effective dimension, not an extrapolation
  predictor.
- **Closed routes:** cosine-only collision ranking, collision/logdet as a
  behavioural target, attention-Fisher ordering, and the LeRoPE `w^(1/3)`
  profile oracle.

## 2026-08-21 to 2026-08-26 — mature-checkpoint intervention ladder

- **Did:** ran phase-chord from-training controls, released-model residual and
  table interventions, same-support Qwen/OLMo controls, fresh FineWeb NLL,
  static single-table gates, co-adaptive recovery, allocation dose response,
  and supporting Video-DiT confirmation. Owners are grouped under
  [`attention-aware-retrofit/results/`](../attention-aware-retrofit/results/).
- **Observed:** pure same-support `z` changes frozen mature behaviour; detailed
  derived profiles often did not separate from a coarse nearest ramp. Two
  analytic Native-support tables improved long NLL while failing Native-window
  retention. Co-adaptation recovered tail NLL but traded against full-sequence
  NLL and did not materially separate 2Wiki capability. The dose curve showed
  graded tail effects without a jointly passing point.
- **Changed our view:** table/weight compatibility and endpoint-specific
  readout dominate static geometry scores. NLL-only positives do not become
  capability or deployment positives.

## 2026-08-27 to 2026-08-31 — zero-training portfolio and ordered coupling

- **Did:** preregistered and closed a success-first portfolio, then evaluated a
  scale-consistent exponent-space table, gain controls, cross-checkpoint
  transport, and same-multiset permutations. The historical portfolio lives in
  [`preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md`](../attention-aware-retrofit/preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md);
  the result owner is
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](../attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).
- **Observed:** one static table could pass declared OLMo 1x gates and improve
  longer endpoints, and the construction transferred long capability to Qwen.
  Preserving the exact frequency multiset while permuting slot assignment
  caused decisive collapse.
- **Changed our view:** the mature object is the ordered pairing between learned
  rotary subspaces and frequencies/dilations, not an unordered spectrum. A
  successful table remains one checkpoint/protocol result, not a universal law.

## 2026-09-01 — cross-K, reference-length, and breadth confirmation

- **Did:** evaluated two-parameter coupling across Qwen K32/K64 and Gemma K128,
  repaired the Native reference-length identification protocol, completed
  reference-correct s2/s4 tables, ran fresh K32/K128 confirmations, and opened
  full RULER-13 only after registered gates. See
  [`attention-aware-retrofit/results/`](../attention-aware-retrofit/results/)
  and receipts under [`evidence/`](../attention-aware-retrofit/evidence/).
- **Observed:** physical/index ordering did not replicate at K32; normalized
  index was more Native-compatible and later won the independent K128 contrast.
  Reference- and request-scale correction recovered Gemma 8K/16K, while the old
  reference control remained zero. K and checkpoint still co-vary, so no K
  causality follows.
- **Changed our view:** normalized index is the best-supported tested transport
  coordinate, not a canonical or universal one. Cell averaging and conditional
  Native-Q/K diagnostics failed their entrance gates.

## 2026-09-02 — natural-text/QA closure, theory-only synthesis, benchmark

- **Did:** completed Qwen packed-natural NLL, far-evidence QA/source-use
  diagnostics, a bounded headwise factorization ladder, and a first-principles
  retrofit memo. We also froze a 16-episode
  [`theory falsification benchmark`](../../../falsification_benchmark/README.md).
- **Observed:** static pure-`z` improves long NLL, RULER/NIAH, and
  source-conditioned answer likelihood, but natural autoregressive QA and EOS
  conversion remain unresolved. Headwise clocks improve the long-task Pareto,
  yet tested log-start and Native-start arms remain on opposite sides of the
  Native-retention/long-QA frontier. The newest Qwen natural/QA raw remote files
  were not recovered, so those session-level numbers remain internal.
- **Changed our view:** useful structure is proven only in parts; current data do
  not identify a canonical nonlinear flow or a shared Native-compatible and
  long-capable direction.
- **Lifecycle:** `PURE_Z_LONG_SIGNAL_ESTABLISHED /
  NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA /
  GPU_METHOD_DEVELOPMENT_STOPPED`.

## What a new AI should retain

1. **Established:** fixed-support allocation is causal during training; frozen
   mature checkpoints are sensitive to ordered allocation; support policy,
   table, gain/routing, adaptation, and endpoint are separate estimands.
2. **Supporting breadth:** 432M MLA, 750M continuation, 1.485B from-initialisation
   and adaptation, 8B adaptation, and Video-DiT each keep their own protocol
   scope.
3. **Not established:** universal optimum, SOTA, continuous basin bounds,
   arbitrary-scale law, natural-generation QA conversion, or K causality.
4. **Do not repeat:** closed routes remain listed in `INDEX.md`; historical
   preflights and plans are not queues.
5. **Current next actions:** read [`../../HANDOFF.md`](../../HANDOFF.md). At this
   snapshot they are manuscript claim selection, missing-raw recovery if a copy
   exists, supplement rebuild on the work machine, and owner-by-owner number
   review—not new GPU method development.
